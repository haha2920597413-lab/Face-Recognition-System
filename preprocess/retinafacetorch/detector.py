"""
本地 .pth 推理（与 biubug6/Pytorch_Retinaface 权重兼容）。
对 224x224 BGR 人脸 crop 做前向，NMS 后取最高分框的 5 点 landmark，形状 (5,2)，顺序与 WIDER/RetinaFace 一致：
左眼、右眼、鼻尖、左嘴角、右嘴角。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .box_utils_torch import decode, decode_landm
from .config import CFG_MNET, CFG_RE50
from .prior_box import PriorBox
from .py_cpu_nms import py_cpu_nms
from .retinaface_net import RetinaFace


def _remove_prefix(state_dict: dict, prefix: str) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _load_checkpoint_dict(pth_path: Union[str, Path]) -> dict:
    p = Path(pth_path)
    if not p.is_file():
        raise FileNotFoundError(f"RetinaFace 权重不存在: {p.resolve()}")
    try:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(p), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"无法解析权重字典: {p}")
    return _remove_prefix(sd, "module.")


class RetinaFaceTorchDetector:
    """加载 mobilenet0.25 / Resnet50 Final .pth，在单张 BGR 图上提取 5 点。"""

    def __init__(
        self,
        pth_path: Union[str, Path],
        backbone: str = "mobile0.25",
        device: Optional[str] = None,
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.4,
        top_k: int = 500,
    ):
        bb = (backbone or "mobile0.25").lower().replace("-", "")
        if bb in ("mobile0.25", "mobilenet0.25", "mnet"):
            self.cfg = CFG_MNET
        elif bb in ("resnet50", "re50", "r50"):
            self.cfg = CFG_RE50
        else:
            raise ValueError(f"不支持的 retinaface_backbone: {backbone}")

        self.net = RetinaFace(cfg=self.cfg, phase="test")
        sd = _load_checkpoint_dict(pth_path)
        self.net.load_state_dict(sd, strict=False)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net.to(self.device)
        self.net.eval()

        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.top_k = int(top_k)

        self._mean = np.array([104, 117, 123], dtype=np.float32)

    @torch.inference_mode()
    def landmarks_5(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        img_bgr: HxWx3, BGR；任意尺寸均可（与 PriorBox 一致）。预处理流水线里一般为 224。
        返回 float32 (5,2)，失败返回 None。
        """
        if img_bgr is None or img_bgr.size == 0:
            return None
        h, w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
        if h < 2 or w < 2:
            return None

        im = np.asarray(img_bgr, dtype=np.float32)
        im -= self._mean
        chw = np.transpose(im, (2, 0, 1))
        img = torch.from_numpy(chw).unsqueeze(0).to(self.device, dtype=torch.float32)

        loc, conf, landms = self.net(img)

        priorbox = PriorBox(self.cfg, image_size=(h, w))
        priors = priorbox.forward().to(self.device)

        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
        boxes = decode(loc.squeeze(0), priors, self.cfg["variance"])
        boxes = boxes * scale

        scores = conf.squeeze(0)[:, 1]
        landms_dec = decode_landm(landms.squeeze(0), priors, self.cfg["variance"])
        scale_lm = torch.tensor([w, h] * 5, dtype=torch.float32, device=self.device)
        landms_dec = landms_dec * scale_lm

        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        landms_np = landms_dec.cpu().numpy()

        inds = np.where(scores_np > self.conf_thresh)[0]
        if inds.size == 0:
            return None

        boxes_np = boxes_np[inds]
        landms_np = landms_np[inds]
        scores_np = scores_np[inds]

        order = scores_np.argsort()[::-1][: self.top_k]
        boxes_np = boxes_np[order]
        landms_np = landms_np[order]
        scores_np = scores_np[order]

        dets = np.hstack((boxes_np, scores_np[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_thresh)
        if len(keep) == 0:
            return None

        best = int(keep[0])
        lm = landms_np[best].reshape(5, 2).astype(np.float32, copy=False)
        return lm
