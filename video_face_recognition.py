import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


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


def load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"配置文件为空: {config_path}")

    return cfg


def resolve_path(project_root: Path, path_str: Union[str, int]) -> Path:
    p = Path(str(path_str))

    if p.is_absolute():
        return p

    return project_root / p


def parse_video_source(source: Any) -> Union[int, str]:
    if isinstance(source, int):
        return source

    source_str = str(source).strip()

    if source_str.isdigit():
        return int(source_str)

    return source_str


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)

    if norm < eps:
        return x

    return x / norm


def _try_create_session(model_path: Path, providers: List[str]) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers,
    )


def create_onnx_session(model_path: Path, device: str = "auto") -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX 模型文件不存在: {model_path}")

    device = str(device).lower().strip()
    available_providers = ort.get_available_providers()

    if device == "cpu":
        providers = ["CPUExecutionProvider"]
        print(f"[INFO] ONNX Runtime providers: {providers}")
        return _try_create_session(model_path, providers)

    if device == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "当前环境没有 CUDAExecutionProvider。请检查 onnxruntime-gpu / CUDA / cuDNN，"
                "或将 video.device 设置为 cpu。"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"[INFO] ONNX Runtime providers: {providers}")
        return _try_create_session(model_path, providers)

    if device == "auto":
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"[INFO] ONNX Runtime providers: {providers}")

            try:
                return _try_create_session(model_path, providers)
            except Exception as e:
                print("[WARN] CUDAExecutionProvider 创建失败，将回退到 CPUExecutionProvider。")
                print(f"[WARN] CUDA 原始错误: {e}")

        providers = ["CPUExecutionProvider"]
        print(f"[INFO] ONNX Runtime providers: {providers}")
        return _try_create_session(model_path, providers)

    raise ValueError(f"不支持的 device 配置: {device}，请使用 auto / cuda / cpu。")


def inspect_model(session: ort.InferenceSession) -> Tuple[str, List[Any]]:
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    print(f"[INFO] ONNX input name: {input_name}")
    print(f"[INFO] ONNX input shape: {input_shape}")

    return input_name, input_shape


def load_multi_template_gallery(
    features_path: Path,
    labels_path: Path,
    person_index_path: Path,
    embedding_dim: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, List[int]]]:
    if not features_path.exists():
        raise FileNotFoundError(f"gallery_features.npy 不存在: {features_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"gallery_labels.json 不存在: {labels_path}")

    if not person_index_path.exists():
        raise FileNotFoundError(f"gallery_person_index.json 不存在: {person_index_path}")

    gallery_features = np.load(str(features_path)).astype(np.float32)

    with open(labels_path, "r", encoding="utf-8") as f:
        gallery_labels = json.load(f)

    with open(person_index_path, "r", encoding="utf-8") as f:
        person_index = json.load(f)

    if len(gallery_features.shape) != 2:
        raise ValueError(f"gallery_features 维度异常: {gallery_features.shape}")

    if gallery_features.shape[1] != embedding_dim:
        raise ValueError(
            f"gallery_features 特征维度异常，期望 {embedding_dim}, 实际 {gallery_features.shape[1]}"
        )

    if not isinstance(gallery_labels, list):
        raise ValueError("gallery_labels.json 格式异常：多模板版本应为 list[dict]。")

    if len(gallery_labels) > 0 and not isinstance(gallery_labels[0], dict):
        raise ValueError("gallery_labels.json 看起来是旧版平均特征库格式，请重新构建多模板 gallery。")

    if gallery_features.shape[0] != len(gallery_labels):
        raise ValueError(
            f"gallery 文件不匹配: features 行数={gallery_features.shape[0]}, "
            f"labels 数量={len(gallery_labels)}"
        )

    normalized_person_index: Dict[str, List[int]] = {}

    for identity_id, indices in person_index.items():
        normalized_indices = [int(i) for i in indices]

        for i in normalized_indices:
            if i < 0 or i >= gallery_features.shape[0]:
                raise ValueError(
                    f"gallery_person_index 中存在越界下标: identity_id={identity_id}, index={i}"
                )

        normalized_person_index[str(identity_id)] = normalized_indices

    norms = np.linalg.norm(gallery_features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    gallery_features = gallery_features / norms

    print(f"[INFO] gallery_features shape: {gallery_features.shape}")
    print(f"[INFO] gallery_labels length: {len(gallery_labels)}")
    print(f"[INFO] gallery_person_index identities: {len(normalized_person_index)}")

    return gallery_features.astype(np.float32), gallery_labels, normalized_person_index


def load_id_name_map(label_csv_path: Path) -> Dict[str, str]:
    if not label_csv_path.exists():
        print(f"[WARN] label.csv 不存在，将只输出 gallery 中的姓名或工号: {label_csv_path}")
        return {}

    last_error = None
    df = None

    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            df = pd.read_csv(label_csv_path, encoding=enc)
            print(f"[INFO] label.csv 使用编码读取成功: {enc}")
            break
        except Exception as e:
            last_error = e

    if df is None:
        print(f"[WARN] label.csv 读取失败，将只输出 gallery 中的姓名或工号，错误: {last_error}")
        return {}

    if "file" not in df.columns or "name" not in df.columns:
        print(f"[WARN] label.csv 缺少 file/name 列，当前列名: {list(df.columns)}")
        return {}

    id_to_name: Dict[str, str] = {}

    for _, row in df.iterrows():
        employee_id = str(row["file"]).strip()
        name = str(row["name"]).strip()

        if employee_id == "" or employee_id.lower() == "nan":
            continue

        if name == "" or name.lower() == "nan":
            name = "UnknownName"

        id_to_name[employee_id] = name

    print(f"[INFO] 工号-姓名映射数量: {len(id_to_name)}")

    return id_to_name


class RetinaFaceLandmarkExtractor:
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

        if not self.weight_path.exists():
            raise FileNotFoundError(f"RetinaFace 权重不存在: {self.weight_path}")

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from preprocess.retinafacetorch import RetinaFaceTorchDetector
        except ImportError as e:
            raise ImportError(
                "无法导入 preprocess.retinafacetorch.RetinaFaceTorchDetector。\n"
                "请确认项目中存在 preprocess/retinafacetorch.py。"
            ) from e

        self.detector = RetinaFaceTorchDetector(
            self.weight_path,
            backbone=backbone,
            device=device,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
        )

    def extract(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
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


def detect_all_faces_yolo(
    frame: np.ndarray,
    yolo_model: YOLO,
    conf_thresh: float,
) -> List[Dict[str, Any]]:
    if frame is None or frame.size == 0:
        return []

    results = yolo_model(frame, conf=conf_thresh, verbose=False)

    if len(results) == 0:
        return []

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return []

    faces = []

    xyxys = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    for xyxy, conf in zip(xyxys, confs):
        x1, y1, x2, y2 = map(int, xyxy)

        if x2 <= x1 or y2 <= y1:
            continue

        faces.append({
            "bbox": [x1, y1, x2, y2],
            "det_conf": float(conf),
        })

    faces = sorted(
        faces,
        key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]),
        reverse=True,
    )

    return faces


def expand_bbox(
    bbox: List[int],
    img_shape: Tuple[int, int, int],
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
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

    nx1 = max(0, int(cx - new_w / 2.0))
    ny1 = max(0, int(cy - new_h / 2.0))
    nx2 = min(w, int(cx + new_w / 2.0))
    ny2 = min(h, int(cy + new_h / 2.0))

    return nx1, ny1, nx2, ny2


def crop_by_bbox(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop


def align_face_by_landmarks(
    crop_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: int,
) -> Optional[np.ndarray]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    if landmarks is None or landmarks.shape != (5, 2):
        return None

    if output_size == 112:
        dst_points = ARCFACE_TEMPLATE_112.copy()
    else:
        dst_points = ARCFACE_TEMPLATE_112.copy() * (output_size / 112.0)

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


def align_face_from_bbox_in_memory(
    frame: np.ndarray,
    bbox: List[int],
    landmark_extractor: RetinaFaceLandmarkExtractor,
    input_size: int,
    expand_ratio: float,
    fallback_to_yolo_crop: bool,
) -> Tuple[Optional[np.ndarray], str, Tuple[int, int, int, int]]:
    expanded_bbox = expand_bbox(
        bbox=bbox,
        img_shape=frame.shape,
        expand_ratio=expand_ratio,
    )

    crop = crop_by_bbox(frame, expanded_bbox)

    if crop is None:
        return None, "crop_failed", expanded_bbox

    landmarks = landmark_extractor.extract(crop)

    if landmarks is not None:
        aligned = align_face_by_landmarks(
            crop_bgr=crop,
            landmarks=landmarks,
            output_size=input_size,
        )

        if aligned is not None:
            return aligned, "aligned", expanded_bbox

    if fallback_to_yolo_crop:
        try:
            fallback = cv2.resize(
                crop,
                (input_size, input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            return fallback, "fallback_yolo_crop", expanded_bbox
        except Exception:
            return None, "fallback_resize_failed", expanded_bbox

    return None, "retinaface_landmark_failed", expanded_bbox


def preprocess_aligned_bgr(
    aligned_bgr: np.ndarray,
    input_size: int,
) -> Optional[np.ndarray]:
    if aligned_bgr is None or aligned_bgr.size == 0:
        return None

    img = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


def extract_feature_from_aligned_bgr(
    session: ort.InferenceSession,
    input_name: str,
    aligned_bgr: np.ndarray,
    input_size: int,
    embedding_dim: int,
) -> Optional[np.ndarray]:
    input_blob = preprocess_aligned_bgr(aligned_bgr, input_size=input_size)

    if input_blob is None:
        return None

    outputs = session.run(None, {input_name: input_blob})

    if outputs is None or len(outputs) == 0:
        return None

    embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    if embedding.shape[0] != embedding_dim:
        raise ValueError(
            f"模型输出维度异常，期望 {embedding_dim}, 实际 {embedding.shape[0]}"
        )

    embedding = l2_normalize(embedding).astype(np.float32)

    return embedding


def aggregate_scores_for_person(
    template_scores: np.ndarray,
    method: str = "max",
    top_k: int = 3,
) -> float:
    if template_scores.size == 0:
        return -1.0

    method = method.lower().strip()

    if method == "max":
        return float(np.max(template_scores))

    if method == "topk_mean":
        k = min(int(top_k), template_scores.size)
        sorted_scores = np.sort(template_scores)[::-1]
        return float(np.mean(sorted_scores[:k]))

    raise ValueError(f"不支持的 aggregate_method: {method}")


def compute_topk_multi_template(
    gallery_features: np.ndarray,
    gallery_labels: List[Dict[str, Any]],
    person_index: Dict[str, List[int]],
    query_feature: np.ndarray,
    id_to_name: Dict[str, str],
    top_k: int,
    aggregate_method: str,
    aggregate_top_k: int,
) -> List[Dict[str, Any]]:
    all_template_scores = gallery_features @ query_feature

    person_results: List[Dict[str, Any]] = []

    for identity_id, indices in person_index.items():
        if len(indices) == 0:
            continue

        idx_array = np.asarray(indices, dtype=np.int64)
        template_scores = all_template_scores[idx_array]

        person_score = aggregate_scores_for_person(
            template_scores,
            method=aggregate_method,
            top_k=aggregate_top_k,
        )

        best_local_pos = int(np.argmax(template_scores))
        best_feature_index = int(idx_array[best_local_pos])
        best_template_score = float(template_scores[best_local_pos])

        label_record = gallery_labels[best_feature_index]
        name = id_to_name.get(identity_id, label_record.get("name", "UnknownName"))

        person_results.append({
            "employee_id": identity_id,
            "name": name,
            "score": person_score,
            "best_template_score": best_template_score,
            "best_template_image": label_record.get("image_name", ""),
            "best_template_path": label_record.get("image_path", ""),
            "num_templates": len(indices),
        })

    person_results = sorted(person_results, key=lambda x: x["score"], reverse=True)

    return [
        dict(item, rank=rank)
        for rank, item in enumerate(person_results[: min(top_k, len(person_results))], start=1)
    ]


def recognize_aligned_face(
    aligned_bgr: np.ndarray,
    session: ort.InferenceSession,
    input_name: str,
    gallery_features: np.ndarray,
    gallery_labels: List[Dict[str, Any]],
    person_index: Dict[str, List[int]],
    id_to_name: Dict[str, str],
    input_size: int,
    embedding_dim: int,
    threshold: float,
    top_k: int,
    aggregate_method: str,
    aggregate_top_k: int,
) -> Dict[str, Any]:
    query_feature = extract_feature_from_aligned_bgr(
        session=session,
        input_name=input_name,
        aligned_bgr=aligned_bgr,
        input_size=input_size,
        embedding_dim=embedding_dim,
    )

    if query_feature is None:
        return {
            "status": "Failed",
            "employee_id": "Failed",
            "name": "Failed",
            "score": 0.0,
            "best_template_score": 0.0,
            "reason": "feature_extract_failed",
        }

    topk_results = compute_topk_multi_template(
        gallery_features=gallery_features,
        gallery_labels=gallery_labels,
        person_index=person_index,
        query_feature=query_feature,
        id_to_name=id_to_name,
        top_k=top_k,
        aggregate_method=aggregate_method,
        aggregate_top_k=aggregate_top_k,
    )

    if len(topk_results) == 0:
        return {
            "status": "Failed",
            "employee_id": "Failed",
            "name": "Failed",
            "score": 0.0,
            "best_template_score": 0.0,
            "reason": "no_topk_result",
        }

    top1 = topk_results[0]
    score = float(top1["score"])

    if score >= threshold:
        return {
            "status": "Known",
            "employee_id": str(top1["employee_id"]),
            "name": str(top1["name"]),
            "score": score,
            "best_template_score": float(top1.get("best_template_score", 0.0)),
            "best_template_image": top1.get("best_template_image", ""),
            "reason": "known",
        }

    return {
        "status": "Unknown",
        "employee_id": "Unknown",
        "name": "Unknown",
        "score": score,
        "best_template_score": float(top1.get("best_template_score", 0.0)),
        "best_template_image": top1.get("best_template_image", ""),
        "reason": "below_threshold",
    }


def get_text_font(font_path: Optional[str], font_size: int) -> Optional[ImageFont.FreeTypeFont]:
    candidate_paths = []

    if font_path:
        candidate_paths.append(font_path)

    candidate_paths.extend([
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])

    for p in candidate_paths:
        try:
            path = Path(p)
            if path.exists():
                return ImageFont.truetype(str(path), font_size)
        except Exception:
            continue

    return None


def draw_text_with_pil(
    frame_bgr: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    font: Optional[ImageFont.FreeTypeFont],
) -> np.ndarray:
    if font is None:
        cv2.putText(
            frame_bgr,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        return frame_bgr

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(position, text, font=font, fill=color_rgb)

    out_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr


def draw_face_results(
    frame: np.ndarray,
    face_results: List[Dict[str, Any]],
    draw_score: bool = True,
    draw_status: bool = True,
    draw_det_conf: bool = False,
    font: Optional[ImageFont.FreeTypeFont] = None,
) -> np.ndarray:
    output = frame.copy()

    for result in face_results:
        bbox = result.get("bbox", None)

        if bbox is None:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        status = result.get("status", "Failed")
        name = result.get("name", status)
        score = float(result.get("score", 0.0))
        det_conf = result.get("det_conf", None)

        if status == "Known":
            color = (0, 255, 0)
        elif status == "Unknown":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        text_parts = [name]

        if draw_score:
            text_parts.append(f"{score:.2f}")

        if draw_status and status not in ["Known"]:
            text_parts.append(status)

        if draw_det_conf and det_conf is not None:
            text_parts.append(f"det={float(det_conf):.2f}")

        text = " ".join(text_parts)

        text_x = x1
        text_y = max(0, y1 - 28)

        bg_x1 = x1
        bg_y1 = max(0, y1 - 34)
        bg_x2 = min(output.shape[1] - 1, x1 + max(160, len(text) * 18))
        bg_y2 = max(0, y1 - 4)

        cv2.rectangle(output, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        output = draw_text_with_pil(
            output,
            text,
            (text_x + 4, text_y),
            (0, 0, 0),
            font,
        )

    return output


class VideoRunLogger:
    """
    每次运行独立保存日志：
        log_root/YYYYMMDD_HHMMSS/video_result.jsonl
        log_root/YYYYMMDD_HHMMSS/run_meta.json

    使用 JSONL + buffer，减少频繁磁盘写入。
    """

    def __init__(
        self,
        project_root: Path,
        video_cfg: Dict[str, Any],
        run_meta: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = bool(video_cfg.get("save_log", True))

        self.log_format = str(video_cfg.get("log_format", "jsonl")).lower().strip()
        if self.log_format != "jsonl":
            raise ValueError("当前版本建议使用 log_format: jsonl")

        self.buffer_size = int(video_cfg.get("log_buffer_size", 50))
        self.flush_interval = float(video_cfg.get("log_flush_interval_seconds", 1.0))

        run_name = str(video_cfg.get("log_run_name", "auto")).strip()
        if run_name == "" or run_name.lower() == "auto":
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_root = resolve_path(
            project_root,
            video_cfg.get("log_root", "log/video_recognition"),
        )

        self.run_dir = log_root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        base_name = str(video_cfg.get("log_file_name", "video_result")).strip()
        if base_name == "":
            base_name = "video_result"

        self.log_path = self.run_dir / f"{base_name}.jsonl"
        self.meta_path = self.run_dir / "run_meta.json"

        self.buffer: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()
        self.last_log_time: Dict[str, float] = {}

        self.log_cooldown_seconds = float(video_cfg.get("log_cooldown_seconds", 2.0))
        self.log_unknown = bool(video_cfg.get("log_unknown", True))
        self.unknown_log_cooldown_seconds = float(
            video_cfg.get("unknown_log_cooldown_seconds", 2.0)
        )

        self.file = open(self.log_path, "a", encoding="utf-8")

        if bool(video_cfg.get("save_run_meta", True)):
            self.save_run_meta(run_meta or {})

        print(f"[INFO] 本次运行日志目录: {self.run_dir}")
        print(f"[INFO] 本次运行日志文件: {self.log_path}")

    def save_run_meta(self, run_meta: Dict[str, Any]) -> None:
        meta = dict(run_meta)
        meta["run_dir"] = str(self.run_dir)
        meta["log_path"] = str(self.log_path)
        meta["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def can_write(self, result: Dict[str, Any]) -> bool:
        status = result.get("status", "Failed")
        employee_id = str(result.get("employee_id", ""))

        if status == "Failed":
            return False

        if status == "Unknown":
            if not self.log_unknown:
                return False

            log_key = "Unknown"
            cooldown = self.unknown_log_cooldown_seconds
        else:
            log_key = employee_id
            cooldown = self.log_cooldown_seconds

        now = time.time()
        last_time = self.last_log_time.get(log_key, -1e18)

        if now - last_time >= cooldown:
            self.last_log_time[log_key] = now
            return True

        return False

    def append(
        self,
        timestamp: str,
        frame_id: int,
        face_results: List[Dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return

        if len(face_results) == 0:
            return

        for result in face_results:
            if not self.can_write(result):
                continue

            bbox = result.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)

            record = {
                "timestamp": timestamp,
                "frame_id": int(frame_id),
                "face_index": result.get("face_index", ""),
                "status": result.get("status", ""),
                "employee_id": result.get("employee_id", ""),
                "name": result.get("name", ""),
                "score": float(result.get("score", 0.0)),
                "det_conf": float(result.get("det_conf", 0.0)),
                "align_status": result.get("align_status", ""),
                "bbox": [x1, y1, x2, y2],
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "best_template_score": float(result.get("best_template_score", 0.0)),
                "best_template_image": result.get("best_template_image", ""),
                "reason": result.get("reason", ""),
            }

            self.buffer.append(record)

        now = time.time()

        if len(self.buffer) >= self.buffer_size or now - self.last_flush_time >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return

        if len(self.buffer) == 0:
            return

        for record in self.buffer:
            self.file.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.file.flush()
        self.buffer.clear()
        self.last_flush_time = time.time()

    def close(self) -> None:
        if not self.enabled:
            return

        self.flush()
        self.file.close()


def process_frame_faces(
    frame: np.ndarray,
    yolo_model: YOLO,
    landmark_extractor: RetinaFaceLandmarkExtractor,
    session: ort.InferenceSession,
    input_name: str,
    gallery_features: np.ndarray,
    gallery_labels: List[Dict[str, Any]],
    person_index: Dict[str, List[int]],
    id_to_name: Dict[str, str],
    yolo_conf: float,
    expand_ratio: float,
    fallback_to_yolo_crop: bool,
    input_size: int,
    embedding_dim: int,
    threshold: float,
    top_k: int,
    aggregate_method: str,
    aggregate_top_k: int,
) -> List[Dict[str, Any]]:
    faces = detect_all_faces_yolo(
        frame=frame,
        yolo_model=yolo_model,
        conf_thresh=yolo_conf,
    )

    face_results: List[Dict[str, Any]] = []

    for face_index, face in enumerate(faces):
        bbox = face["bbox"]
        det_conf = face["det_conf"]

        aligned_bgr, align_status, expanded_bbox = align_face_from_bbox_in_memory(
            frame=frame,
            bbox=bbox,
            landmark_extractor=landmark_extractor,
            input_size=input_size,
            expand_ratio=expand_ratio,
            fallback_to_yolo_crop=fallback_to_yolo_crop,
        )

        if aligned_bgr is None:
            face_results.append({
                "face_index": face_index,
                "bbox": bbox,
                "expanded_bbox": list(expanded_bbox),
                "det_conf": det_conf,
                "status": "Failed",
                "employee_id": "Failed",
                "name": "Failed",
                "score": 0.0,
                "align_status": align_status,
                "reason": align_status,
            })
            continue

        recog_result = recognize_aligned_face(
            aligned_bgr=aligned_bgr,
            session=session,
            input_name=input_name,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            person_index=person_index,
            id_to_name=id_to_name,
            input_size=input_size,
            embedding_dim=embedding_dim,
            threshold=threshold,
            top_k=top_k,
            aggregate_method=aggregate_method,
            aggregate_top_k=aggregate_top_k,
        )

        result = {
            "face_index": face_index,
            "bbox": bbox,
            "expanded_bbox": list(expanded_bbox),
            "det_conf": det_conf,
            "align_status": align_status,
            **recog_result,
        }

        face_results.append(result)

    return face_results


def open_video_writer(
    output_path: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    return writer


def run_video_face_recognition(config_path: str) -> None:
    cfg = load_config(config_path)

    project_root = Path(cfg["project"]["root"]).expanduser().resolve()

    weights_cfg = cfg["weights"]
    preprocess_cfg = cfg.get("preprocess", {})
    video_cfg = cfg["video"]

    source = parse_video_source(video_cfg.get("source", 0))

    yolo_weight = resolve_path(project_root, weights_cfg["yolo_face"])
    retinaface_weight = resolve_path(project_root, weights_cfg["retinaface"])
    recog_model_path = resolve_path(project_root, weights_cfg["arcface_onnx"])

    label_csv_path = resolve_path(project_root, video_cfg["label_csv"])
    gallery_features_path = resolve_path(project_root, video_cfg["gallery_features"])
    gallery_labels_path = resolve_path(project_root, video_cfg["gallery_labels"])
    gallery_person_index_path = resolve_path(project_root, video_cfg["gallery_person_index"])

    process_interval = int(video_cfg.get("process_interval", 3))
    process_interval = max(1, process_interval)

    draw_last_results = bool(video_cfg.get("draw_last_results", True))

    show_window = bool(video_cfg.get("show_window", True))
    window_name = str(video_cfg.get("window_name", "Video Face Recognition"))
    display_scale = float(video_cfg.get("display_scale", 1.0))

    save_output_video = bool(video_cfg.get("save_output_video", False))
    output_video_path = resolve_path(project_root, video_cfg.get("output_video_path", "outputs/video_result.mp4"))

    save_log = bool(video_cfg.get("save_log", True))

    yolo_conf = float(video_cfg.get("yolo_conf", preprocess_cfg.get("yolo_conf", 0.5)))
    expand_ratio = float(video_cfg.get("face_expand_ratio", preprocess_cfg.get("face_expand_ratio", 0.3)))

    threshold = float(video_cfg.get("threshold", 0.4))
    top_k = int(video_cfg.get("top_k", 5))
    aggregate_method = str(video_cfg.get("aggregate_method", "max"))
    aggregate_top_k = int(video_cfg.get("aggregate_top_k", 3))

    input_size = int(video_cfg.get("input_size", preprocess_cfg.get("input_size", 112)))
    embedding_dim = int(video_cfg.get("embedding_dim", 512))
    device = str(video_cfg.get("device", "cpu"))

    fallback_to_yolo_crop = bool(video_cfg.get("fallback_to_yolo_crop", preprocess_cfg.get("fallback_to_yolo_crop", True)))

    draw_score = bool(video_cfg.get("draw_score", True))
    draw_status = bool(video_cfg.get("draw_status", True))
    draw_det_conf = bool(video_cfg.get("draw_det_conf", False))
    draw_fps = bool(video_cfg.get("draw_fps", True))

    quit_key = str(video_cfg.get("quit_key", "q"))

    font_path = video_cfg.get("font_path", None)
    font_size = int(video_cfg.get("font_size", 22))
    font = get_text_font(font_path, font_size)

    retinaface_backbone = str(preprocess_cfg.get("retinaface_backbone", "resnet50"))
    retinaface_confidence = float(preprocess_cfg.get("retinaface_confidence", 0.5))
    retinaface_nms = float(preprocess_cfg.get("retinaface_nms", 0.4))
    retinaface_device = preprocess_cfg.get("retinaface_device", None)

    if retinaface_device == "null":
        retinaface_device = None

    print("=" * 80)
    print("[INFO] 启动视频流多人脸识别")
    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] source: {source}")
    print(f"[INFO] process_interval: {process_interval}")
    print(f"[INFO] yolo_weight: {yolo_weight}")
    print(f"[INFO] retinaface_weight: {retinaface_weight}")
    print(f"[INFO] recognition_model: {recog_model_path}")
    print(f"[INFO] gallery_features: {gallery_features_path}")
    print(f"[INFO] threshold: {threshold}")
    print(f"[INFO] aggregate_method: {aggregate_method}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] save_log: {save_log}")
    print("=" * 80)

    for p, name in [
        (yolo_weight, "YOLO 权重"),
        (retinaface_weight, "RetinaFace 权重"),
        (recog_model_path, "识别 ONNX 模型"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name}不存在: {p}")

    print("\n[INFO] 加载 YOLOv8-face...")
    yolo_model = YOLO(str(yolo_weight))

    print("[INFO] 加载 RetinaFace...")
    landmark_extractor = RetinaFaceLandmarkExtractor(
        project_root=project_root,
        weight_path=retinaface_weight,
        backbone=retinaface_backbone,
        conf_thresh=retinaface_confidence,
        nms_thresh=retinaface_nms,
        device=retinaface_device,
    )

    print("[INFO] 加载识别 ONNX 模型...")
    session = create_onnx_session(recog_model_path, device=device)
    input_name, _ = inspect_model(session)

    print("[INFO] 加载多模板 gallery...")
    gallery_features, gallery_labels, person_index = load_multi_template_gallery(
        features_path=gallery_features_path,
        labels_path=gallery_labels_path,
        person_index_path=gallery_person_index_path,
        embedding_dim=embedding_dim,
    )

    id_to_name = load_id_name_map(label_csv_path)

    logger = None
    if save_log:
        run_meta = {
            "source": source,
            "process_interval": process_interval,
            "threshold": threshold,
            "aggregate_method": aggregate_method,
            "aggregate_top_k": aggregate_top_k,
            "device": device,
            "yolo_weight": str(yolo_weight),
            "retinaface_weight": str(retinaface_weight),
            "recognition_model": str(recog_model_path),
            "gallery_features": str(gallery_features_path),
            "gallery_labels": str(gallery_labels_path),
            "gallery_person_index": str(gallery_person_index_path),
        }
        logger = VideoRunLogger(
            project_root=project_root,
            video_cfg=video_cfg,
            run_meta=run_meta,
        )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 1:
        src_fps = 25.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None

    if save_output_video:
        writer = open_video_writer(
            output_path=output_video_path,
            fps=src_fps,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    last_face_results: List[Dict[str, Any]] = []

    frame_id = 0
    fps_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    print("[INFO] 视频流已启动。按 q 退出。")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("[INFO] 视频读取结束或摄像头断开。")
                break

            frame_id += 1
            fps_count += 1

            now = time.time()
            elapsed = now - fps_start_time

            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_start_time = now

            should_process = (frame_id % process_interval == 0)

            if should_process:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                face_results = process_frame_faces(
                    frame=frame,
                    yolo_model=yolo_model,
                    landmark_extractor=landmark_extractor,
                    session=session,
                    input_name=input_name,
                    gallery_features=gallery_features,
                    gallery_labels=gallery_labels,
                    person_index=person_index,
                    id_to_name=id_to_name,
                    yolo_conf=yolo_conf,
                    expand_ratio=expand_ratio,
                    fallback_to_yolo_crop=fallback_to_yolo_crop,
                    input_size=input_size,
                    embedding_dim=embedding_dim,
                    threshold=threshold,
                    top_k=top_k,
                    aggregate_method=aggregate_method,
                    aggregate_top_k=aggregate_top_k,
                )

                last_face_results = face_results

                if logger is not None:
                    logger.append(
                        timestamp=timestamp,
                        frame_id=frame_id,
                        face_results=face_results,
                    )

            if draw_last_results:
                draw_results = last_face_results
            else:
                draw_results = last_face_results if should_process else []

            display_frame = draw_face_results(
                frame=frame,
                face_results=draw_results,
                draw_score=draw_score,
                draw_status=draw_status,
                draw_det_conf=draw_det_conf,
                font=font,
            )

            if draw_fps:
                fps_text = f"FPS: {current_fps:.1f}  Frame: {frame_id}"
                cv2.putText(
                    display_frame,
                    fps_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(display_frame)

            if show_window:
                if display_scale != 1.0:
                    display_frame_show = cv2.resize(
                        display_frame,
                        None,
                        fx=display_scale,
                        fy=display_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    display_frame_show = display_frame

                cv2.imshow(window_name, display_frame_show)

                key = cv2.waitKey(1) & 0xFF

                if key == ord(quit_key):
                    print("[INFO] 用户退出。")
                    break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if logger is not None:
            logger.close()

        if show_window:
            cv2.destroyAllWindows()

        print("=" * 80)
        print("[INFO] 视频识别结束")
        print(f"[INFO] 总帧数: {frame_id}")

        if logger is not None:
            print(f"[INFO] 本次运行日志目录: {logger.run_dir}")
            print(f"[INFO] 本次运行日志文件: {logger.log_path}")

        if save_output_video:
            print(f"[INFO] 输出视频: {output_video_path}")

        print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video face recognition with YOLOv8-face, RetinaFace alignment, ONNX recognition, and multi-template gallery."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        run_video_face_recognition(args.config)
    except Exception as e:
        import traceback

        print("\n[ERROR] 程序运行失败：")
        print(e)
        traceback.print_exc()
        input("\n按 q 键退出...")