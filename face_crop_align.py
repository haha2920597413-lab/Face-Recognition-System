import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# =========================
# ArcFace / MobileFaceNet 112x112 标准五点模板
# 顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
# =========================
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def load_yaml(config_path: Path) -> dict:
    """读取 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    支持中文路径的 OpenCV 读图。
    普通 cv2.imread 在部分 Windows 中文路径下可能失败。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """
    支持中文路径的 OpenCV 写图。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix not in [".jpg", ".jpeg", ".png", ".bmp"]:
            suffix = ".jpg"

        success, encoded_img = cv2.imencode(suffix, img)
        if not success:
            return False

        encoded_img.tofile(str(path))
        return True
    except Exception:
        return False


def collect_images(raw_dir: Path, image_exts: List[str]) -> List[Path]:
    """
    遍历 raw 目录，收集所有身份文件夹下的图片。

    期望输入结构：
        raw/
          101024/
            101024_王金霞_000.jpg
          101038/
            101038_侯旭_000.jpg
    """
    image_exts = {ext.lower() for ext in image_exts}
    image_paths = []

    for person_dir in sorted(raw_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        for img_path in sorted(person_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in image_exts:
                image_paths.append(img_path)

    return image_paths


def detect_best_face_yolo(
    img: np.ndarray,
    yolo_model: YOLO,
    conf_thresh: float,
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    使用 YOLOv8-face 检测人脸，返回最高置信度的人脸框。

    返回：
        x1, y1, x2, y2, confidence
    """
    if img is None or img.size == 0:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    results = yolo_model(img, conf=conf_thresh, verbose=False)

    if len(results) == 0:
        return None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = boxes.conf.argmax().item()

    xyxy = boxes.xyxy[best_idx].cpu().numpy()
    conf = float(boxes.conf[best_idx].cpu().numpy())

    x1, y1, x2, y2 = map(int, xyxy)

    return x1, y1, x2, y2, conf


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int, int],
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    对 YOLO 检测框进行外扩，避免裁得太紧。
    """
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return x1, y1, x2, y2

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    new_w = box_w * (1.0 + expand_ratio)
    new_h = box_h * (1.0 + expand_ratio)

    nx1 = int(cx - new_w / 2.0)
    ny1 = int(cy - new_h / 2.0)
    nx2 = int(cx + new_w / 2.0)
    ny2 = int(cy + new_h / 2.0)

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    return nx1, ny1, nx2, ny2


def crop_by_bbox(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """
    根据 bbox 裁剪图像。
    """
    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop


class RetinaFaceLandmarkExtractor:
    """
    使用本地 PyTorch RetinaFace 权重提取 5 点关键点。

    依赖：
        你的项目中需要有 preprocess/retinafacetorch.py
        并且其中包含 RetinaFaceTorchDetector 类。

    该类需要支持：
        detector = RetinaFaceTorchDetector(...)
        detector.landmarks_5(img_bgr)
    """

    def __init__(
        self,
        project_root: Path,
        weight_path: Path,
        backbone: str = "resnet50",
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        device: Optional[str] = None,
    ):
        self.project_root = project_root
        self.weight_path = weight_path
        self.backbone = backbone
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.device = device

        if not self.weight_path.exists():
            raise FileNotFoundError(f"未找到 RetinaFace 权重文件：{self.weight_path}")

        # 把项目根目录加入 Python 路径，方便导入 preprocess.retinafacetorch
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        try:
            from preprocess.retinafacetorch import RetinaFaceTorchDetector
        except ImportError as e:
            raise ImportError(
                "无法导入 preprocess.retinafacetorch.RetinaFaceTorchDetector。\n"
                "请确认你的项目中存在：preprocess/retinafacetorch.py\n"
                "并且其中定义了 RetinaFaceTorchDetector 类。\n"
                "如果你之前项目里有这个文件，请复制到当前项目的 preprocess/ 目录下。"
            ) from e

        self.detector = RetinaFaceTorchDetector(
            self.weight_path,
            backbone=self.backbone,
            device=self.device,
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
        )

    def extract(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        输入 BGR 图像，输出 5 点关键点。
        返回 shape = [5, 2]，顺序：
            左眼、右眼、鼻尖、左嘴角、右嘴角
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        try:
            landmarks = self.detector.landmarks_5(img_bgr)
        except Exception:
            return None

        if landmarks is None:
            return None

        landmarks = np.asarray(landmarks, dtype=np.float32)

        if landmarks.shape != (5, 2):
            return None

        return landmarks


def align_face_by_landmarks(
    crop_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: int,
) -> Optional[np.ndarray]:
    """
    使用 5 点关键点将人脸对齐到 ArcFace / MobileFaceNet 标准模板。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    if landmarks is None or landmarks.shape != (5, 2):
        return None

    if output_size == 112:
        dst_points = ARCFACE_TEMPLATE_112.copy()
    else:
        scale = output_size / 112.0
        dst_points = ARCFACE_TEMPLATE_112.copy() * scale

    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks,
        dst_points,
        method=cv2.LMEDS,
    )

    if matrix is None:
        return None

    aligned = cv2.warpAffine(
        crop_bgr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    return aligned


def fallback_yolo_crop_resize(
    crop_bgr: np.ndarray,
    output_size: int,
) -> Optional[np.ndarray]:
    """
    当 RetinaFace 关键点失败时，退化为 YOLO 裁剪 + resize。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    return cv2.resize(crop_bgr, (output_size, output_size))


def process_one_image(
    img_path: Path,
    raw_dir: Path,
    output_dir: Path,
    yolo_model: YOLO,
    landmark_extractor: RetinaFaceLandmarkExtractor,
    cfg: dict,
) -> Tuple[str, Dict]:
    """
    处理单张图片。

    输入：
        raw/101024/101024_王金霞_000.jpg

    输出：
        identities/101024/101024_王金霞_000.jpg

    返回：
        status_type: "success" / "failed" / "skipped"
        record: 字典记录
    """
    preprocess_cfg = cfg["preprocess"]

    output_size = int(preprocess_cfg.get("input_size", 112))
    yolo_conf = float(preprocess_cfg.get("yolo_conf", 0.5))
    expand_ratio = float(preprocess_cfg.get("face_expand_ratio", 0.20))

    fallback_enabled = bool(preprocess_cfg.get("fallback_to_yolo_crop", True))
    skip_if_exists = bool(preprocess_cfg.get("skip_if_exists", True))
    overwrite = bool(preprocess_cfg.get("overwrite", False))

    file_id = img_path.parent.name
    relative_path = img_path.relative_to(raw_dir)
    out_path = output_dir / relative_path

    # =========================
    # 断点续跑机制
    # =========================
    if out_path.exists() and skip_if_exists and not overwrite:
        return "skipped", {
            "file": file_id,
            "source_image": str(img_path),
            "output_image": str(out_path),
            "status": "skipped_exists",
            "reason": "output_already_exists",
        }

    img = imread_unicode(img_path)

    if img is None:
        return "failed", {
            "file": file_id,
            "source_image": str(img_path),
            "reason": "image_read_failed",
        }

    # =========================
    # Step 1: YOLO 检测人脸框
    # =========================
    yolo_result = detect_best_face_yolo(
        img=img,
        yolo_model=yolo_model,
        conf_thresh=yolo_conf,
    )

    if yolo_result is None:
        return "failed", {
            "file": file_id,
            "source_image": str(img_path),
            "reason": "yolo_face_not_detected",
        }

    x1, y1, x2, y2, yolo_score = yolo_result

    # =========================
    # Step 2: 人脸框外扩
    # =========================
    expanded_bbox = expand_bbox(
        bbox=(x1, y1, x2, y2),
        img_shape=img.shape,
        expand_ratio=expand_ratio,
    )

    # =========================
    # Step 3: 按外扩框裁剪
    # =========================
    crop = crop_by_bbox(img, expanded_bbox)

    if crop is None:
        return "failed", {
            "file": file_id,
            "source_image": str(img_path),
            "reason": "crop_failed",
            "bbox": [x1, y1, x2, y2],
            "expanded_bbox": list(expanded_bbox),
        }

    # =========================
    # Step 4: RetinaFace 提取 5 点关键点
    # 关键点坐标基于 crop 图像
    # =========================
    landmarks = landmark_extractor.extract(crop)

    # =========================
    # Step 5: 如果关键点成功，仿射对齐到 112x112
    # =========================
    if landmarks is not None:
        aligned = align_face_by_landmarks(
            crop_bgr=crop,
            landmarks=landmarks,
            output_size=output_size,
        )

        if aligned is not None:
            save_ok = imwrite_unicode(out_path, aligned)

            if not save_ok:
                return "failed", {
                    "file": file_id,
                    "source_image": str(img_path),
                    "output_image": str(out_path),
                    "reason": "image_write_failed",
                }

            return "success", {
                "file": file_id,
                "source_image": str(img_path),
                "output_image": str(out_path),
                "status": "success_aligned",
                "yolo_score": yolo_score,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "expanded_x1": expanded_bbox[0],
                "expanded_y1": expanded_bbox[1],
                "expanded_x2": expanded_bbox[2],
                "expanded_y2": expanded_bbox[3],
                "landmarks": landmarks.tolist(),
                "output_size": output_size,
            }

    # =========================
    # Step 6: 如果关键点失败，是否使用 YOLO 裁剪 resize 兜底
    # =========================
    if fallback_enabled:
        fallback_img = fallback_yolo_crop_resize(
            crop_bgr=crop,
            output_size=output_size,
        )

        if fallback_img is None:
            return "failed", {
                "file": file_id,
                "source_image": str(img_path),
                "reason": "fallback_resize_failed",
            }

        save_ok = imwrite_unicode(out_path, fallback_img)

        if not save_ok:
            return "failed", {
                "file": file_id,
                "source_image": str(img_path),
                "output_image": str(out_path),
                "reason": "image_write_failed",
            }

        return "success", {
            "file": file_id,
            "source_image": str(img_path),
            "output_image": str(out_path),
            "status": "success_yolo_crop_fallback",
            "yolo_score": yolo_score,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "expanded_x1": expanded_bbox[0],
            "expanded_y1": expanded_bbox[1],
            "expanded_x2": expanded_bbox[2],
            "expanded_y2": expanded_bbox[3],
            "output_size": output_size,
            "reason": "retinaface_landmark_failed_but_fallback_used",
        }

    return "failed", {
        "file": file_id,
        "source_image": str(img_path),
        "reason": "retinaface_landmark_failed",
        "yolo_score": yolo_score,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "expanded_x1": expanded_bbox[0],
        "expanded_y1": expanded_bbox[1],
        "expanded_x2": expanded_bbox[2],
        "expanded_y2": expanded_bbox[3],
    }


def main():
    # 默认从当前运行目录读取 config.yaml
    config_path = Path("config.yaml").resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件：{config_path}")

    cfg = load_yaml(config_path)

    project_root = Path(cfg["project"]["root"]).expanduser().resolve()

    raw_dir = project_root / cfg["paths"]["raw_dir"]
    output_dir = project_root / cfg["paths"]["output_dir"]
    log_dir = project_root / cfg["paths"].get("log_dir", "log")

    yolo_weight = project_root / cfg["weights"]["yolo_face"]
    retinaface_weight = project_root / cfg["weights"]["retinaface"]

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir 不存在：{raw_dir}")

    if not yolo_weight.exists():
        raise FileNotFoundError(f"YOLO 权重不存在：{yolo_weight}")

    if not retinaface_weight.exists():
        raise FileNotFoundError(f"RetinaFace 权重不存在：{retinaface_weight}")

    preprocess_cfg = cfg["preprocess"]

    image_exts = preprocess_cfg.get(
        "image_exts",
        [".jpg", ".jpeg", ".png", ".bmp"],
    )

    print("=" * 80)
    print("Face Crop & Align")
    print("=" * 80)
    print(f"项目根目录：{project_root}")
    print(f"输入目录 raw：{raw_dir}")
    print(f"输出目录 identities：{output_dir}")
    print(f"日志目录 log：{log_dir}")
    print(f"YOLO 权重：{yolo_weight}")
    print(f"RetinaFace 权重：{retinaface_weight}")
    print(f"输出尺寸：{preprocess_cfg.get('input_size', 112)}")
    print(f"断点续跑 skip_if_exists：{preprocess_cfg.get('skip_if_exists', True)}")
    print(f"覆盖已有 overwrite：{preprocess_cfg.get('overwrite', False)}")
    print("=" * 80)

    # =========================
    # 加载 YOLOv8-face
    # =========================
    print("\n正在加载 YOLOv8-face 模型...")
    yolo_model = YOLO(str(yolo_weight))

    # =========================
    # 加载 PyTorch RetinaFace
    # =========================
    print("正在加载 RetinaFace 关键点模型...")
    landmark_extractor = RetinaFaceLandmarkExtractor(
        project_root=project_root,
        weight_path=retinaface_weight,
        backbone=str(preprocess_cfg.get("retinaface_backbone", "resnet50")),
        conf_thresh=float(preprocess_cfg.get("retinaface_confidence", 0.5)),
        nms_thresh=float(preprocess_cfg.get("retinaface_nms", 0.4)),
        device=preprocess_cfg.get("retinaface_device", None),
    )

    # =========================
    # 收集图片
    # =========================
    image_paths = collect_images(raw_dir, image_exts)

    print(f"\n共找到 {len(image_paths)} 张待处理图片。")

    if len(image_paths) == 0:
        print("没有找到图片，请检查 raw 目录结构和 image_exts 配置。")
        return

    success_records = []
    failed_records = []
    skipped_records = []

    for img_path in tqdm(image_paths, desc="Face crop and align"):
        status_type, record = process_one_image(
            img_path=img_path,
            raw_dir=raw_dir,
            output_dir=output_dir,
            yolo_model=yolo_model,
            landmark_extractor=landmark_extractor,
            cfg=cfg,
        )

        if status_type == "success":
            success_records.append(record)
        elif status_type == "failed":
            failed_records.append(record)
        elif status_type == "skipped":
            skipped_records.append(record)
        else:
            failed_records.append({
                "source_image": str(img_path),
                "reason": f"unknown_status_type_{status_type}",
            })

    # =========================
    # 保存报告
    # =========================
    success_csv = log_dir / cfg["output"].get(
        "success_csv",
        "face_crop_align_success.csv",
    )
    failed_csv = log_dir / cfg["output"].get(
        "failed_csv",
        "face_crop_align_failed.csv",
    )
    skipped_csv = log_dir / cfg["output"].get(
        "skipped_csv",
        "face_crop_align_skipped.csv",
    )

    pd.DataFrame(success_records).to_csv(
        success_csv,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(failed_records).to_csv(
        failed_csv,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(skipped_records).to_csv(
        skipped_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n处理完成。")
    print(f"成功处理：{len(success_records)}")
    print(f"跳过已有：{len(skipped_records)}")
    print(f"处理失败：{len(failed_records)}")
    print(f"\n成功记录：{success_csv}")
    print(f"跳过记录：{skipped_csv}")
    print(f"失败记录：{failed_csv}")

    if failed_records:
        print("\n存在失败样本，请检查 failed_csv。")
        print("常见原因：YOLO 未检测到人脸、RetinaFace 关键点失败、图片损坏。")


if __name__ == "__main__":
    main()