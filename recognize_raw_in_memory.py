import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml
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


def resolve_path(project_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def read_image_unicode(image_path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            print(f"[WARN] 图片文件为空或无法读取字节: {image_path}")
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cv2.imdecode 解码失败: {image_path}")
            return None
        return img
    except Exception as e:
        print(f"[WARN] 图片读取异常: {image_path}, 原因: {e}")
        return None


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
    return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)


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
                "或将 recognition.device 设置为 cpu。"
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


def detect_best_face_yolo(img: np.ndarray, yolo_model: YOLO, conf_thresh: float) -> Optional[Tuple[int, int, int, int, float]]:
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


def expand_bbox(bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int, int], expand_ratio: float) -> Tuple[int, int, int, int]:
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


def crop_by_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


class RetinaFaceLandmarkExtractor:
    """
    使用项目内 preprocess/retinafacetorch.py 的 RetinaFaceTorchDetector 提取 5 点关键点。
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


def align_face_by_landmarks(crop_bgr: np.ndarray, landmarks: np.ndarray, output_size: int) -> Optional[np.ndarray]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if landmarks is None or landmarks.shape != (5, 2):
        return None
    if output_size == 112:
        dst_points = ARCFACE_TEMPLATE_112.copy()
    else:
        dst_points = ARCFACE_TEMPLATE_112.copy() * (output_size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, dst_points, method=cv2.LMEDS)
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


def align_raw_face_in_memory(
    image_path: Path,
    yolo_model: YOLO,
    landmark_extractor: RetinaFaceLandmarkExtractor,
    input_size: int,
    yolo_conf: float,
    expand_ratio: float,
    fallback_to_yolo_crop: bool,
) -> Tuple[Optional[np.ndarray], str, str]:
    """
    原始图 -> YOLO 检测 -> bbox 扩张裁剪 -> RetinaFace 5 点关键点 -> ArcFace 对齐。
    不保存中间图片，只返回内存中的 aligned_bgr。
    返回: aligned_bgr, align_status, detect_info
    """
    img = read_image_unicode(image_path)
    if img is None:
        return None, "image_read_failed", ""

    yolo_result = detect_best_face_yolo(img=img, yolo_model=yolo_model, conf_thresh=yolo_conf)
    if yolo_result is None:
        return None, "yolo_face_not_detected", ""

    x1, y1, x2, y2, conf = yolo_result
    expanded_bbox = expand_bbox((x1, y1, x2, y2), img.shape, expand_ratio)
    crop = crop_by_bbox(img, expanded_bbox)
    detect_info = f"yolo_conf={conf:.4f};bbox=({x1},{y1},{x2},{y2});expanded_bbox={expanded_bbox}"

    if crop is None:
        return None, "crop_failed", detect_info

    landmarks = landmark_extractor.extract(crop)
    if landmarks is not None:
        aligned = align_face_by_landmarks(crop_bgr=crop, landmarks=landmarks, output_size=input_size)
        if aligned is not None:
            return aligned, "aligned", detect_info

    if fallback_to_yolo_crop:
        try:
            fallback = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            return fallback, "fallback_yolo_crop", detect_info
        except Exception:
            return None, "fallback_resize_failed", detect_info

    return None, "retinaface_landmark_failed", detect_info


def preprocess_aligned_bgr(aligned_bgr: np.ndarray, input_size: int) -> Optional[np.ndarray]:
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
    try:
        outputs = session.run(None, {input_name: input_blob})
        if outputs is None or len(outputs) == 0:
            return None
        embedding = np.array(outputs[0], dtype=np.float32, copy=True).reshape(-1)
        if embedding.shape[0] != embedding_dim:
            raise ValueError(f"模型输出维度异常，期望 {embedding_dim}, 实际 {embedding.shape[0]}")
        embedding = l2_normalize(embedding).astype(np.float32, copy=False)
        return embedding.copy()
    finally:
        try:
            del input_blob
            del outputs
        except Exception:
            pass


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
        raise ValueError(f"gallery_features 特征维度异常，期望 {embedding_dim}, 实际 {gallery_features.shape[1]}")
    if not isinstance(gallery_labels, list):
        raise ValueError("gallery_labels.json 格式异常：多模板版本应为 list[dict]。")
    if len(gallery_labels) > 0 and not isinstance(gallery_labels[0], dict):
        raise ValueError("gallery_labels.json 看起来是旧版平均特征库格式，请重新构建多模板 gallery。")
    if gallery_features.shape[0] != len(gallery_labels):
        raise ValueError(f"gallery 文件不匹配: features 行数={gallery_features.shape[0]}, labels 数量={len(gallery_labels)}")

    normalized_person_index: Dict[str, List[int]] = {}
    for identity_id, indices in person_index.items():
        normalized_indices = [int(i) for i in indices]
        for i in normalized_indices:
            if i < 0 or i >= gallery_features.shape[0]:
                raise ValueError(f"gallery_person_index 中存在越界下标: identity_id={identity_id}, index={i}")
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
        print(f"[WARN] label.csv 不存在，将只输出工号: {label_csv_path}")
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
        print(f"[WARN] label.csv 读取失败，将只输出工号，错误: {last_error}")
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


def collect_raw_images(raw_dir: Path, image_exts: List[str], recursive: bool = True) -> List[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"待识别原始图片目录不存在: {raw_dir}")
    image_exts = [ext.lower() for ext in image_exts]
    iterator = raw_dir.rglob("*") if recursive else raw_dir.iterdir()
    image_paths = [p for p in iterator if p.is_file() and p.suffix.lower() in image_exts]
    return sorted(image_paths, key=lambda x: str(x))


def aggregate_scores_for_person(template_scores: np.ndarray, method: str = "max", top_k: int = 3) -> float:
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
    aggregate_method: str = "max",
    aggregate_top_k: int = 3,
) -> List[Dict[str, Any]]:
    all_template_scores = gallery_features @ query_feature
    person_results: List[Dict[str, Any]] = []
    for identity_id, indices in person_index.items():
        if len(indices) == 0:
            continue
        idx_array = np.asarray(indices, dtype=np.int64)
        template_scores = all_template_scores[idx_array]
        person_score = aggregate_scores_for_person(template_scores, aggregate_method, aggregate_top_k)
        best_local_pos = int(np.argmax(template_scores))
        best_feature_index = int(idx_array[best_local_pos])
        best_template_score = float(template_scores[best_local_pos])
        label_record = gallery_labels[best_feature_index]
        name = id_to_name.get(identity_id, label_record.get("name", "UnknownName"))
        person_results.append(
            {
                "employee_id": identity_id,
                "name": name,
                "score": person_score,
                "best_template_score": best_template_score,
                "best_feature_index": best_feature_index,
                "best_template_image": label_record.get("image_name", ""),
                "best_template_path": label_record.get("image_path", ""),
                "num_templates": len(indices),
            }
        )
    person_results = sorted(person_results, key=lambda x: x["score"], reverse=True)
    return [dict(item, rank=rank) for rank, item in enumerate(person_results[: min(top_k, len(person_results))], start=1)]


def format_topk_for_csv(topk_results: List[Dict[str, Any]]) -> str:
    items = []
    for item in topk_results:
        items.append(
            f"{item['rank']}:"
            f"{item['employee_id']}:"
            f"{item['name']}:"
            f"{item['score']:.4f}:"
            f"best_template={item.get('best_template_image', '')}:"
            f"best_template_score={item.get('best_template_score', 0.0):.4f}"
        )
    return " | ".join(items)


def get_final_prediction(top1_result: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    best_score = float(top1_result["score"])
    if best_score >= threshold:
        return {
            "pred_status": "Known",
            "pred_id": top1_result["employee_id"],
            "pred_name": top1_result["name"],
            "score": best_score,
        }
    return {"pred_status": "Unknown", "pred_id": "Unknown", "pred_name": "Unknown", "score": best_score}


def print_recognition_result(image_path: Path, topk_results: List[Dict[str, Any]], threshold: float, align_status: str) -> None:
    print("=" * 80)
    print(f"图片: {image_path}")
    print(f"对齐状态: {align_status}")
    if len(topk_results) == 0:
        print("识别失败: 无 Top-K 结果")
        print("=" * 80)
        return
    top1 = topk_results[0]
    final_pred = get_final_prediction(top1, threshold)
    print("\nTop-1:")
    print(f"工号: {final_pred['pred_id']}")
    print(f"姓名: {final_pred['pred_name']}")
    print(f"相似度: {final_pred['score']:.4f}")
    print(f"阈值: {threshold:.4f}")
    print(f"判定: {final_pred['pred_status']}")
    print("\nTop-K 候选:")
    for item in topk_results:
        print(
            f"{item['rank']}. {item['employee_id']} {item['name']} "
            f"person_score={item['score']:.4f} "
            f"best_template={item.get('best_template_image', '')} "
            f"best_template_score={item.get('best_template_score', 0.0):.4f} "
            f"templates={item.get('num_templates', '')}"
        )
    print("=" * 80)


def save_results_csv(records: List[Dict[str, Any]], result_csv_path: Path) -> None:
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "image_name",
        "align_status",
        "detect_info",
        "pred_status",
        "pred_id",
        "pred_name",
        "person_score",
        "top1_candidate_id",
        "top1_candidate_name",
        "top1_candidate_score",
        "top1_best_template",
        "top1_best_template_score",
        "top1_num_templates",
        "threshold",
        "aggregate_method",
        "aggregate_top_k",
        "topk",
    ]
    with open(result_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print(f"[INFO] 识别结果已保存: {result_csv_path}")


def recognize_raw_images(config_path: str) -> None:
    cfg = load_config(config_path)
    project_root = Path(cfg["project"]["root"]).expanduser().resolve()
    weights_cfg = cfg["weights"]
    preprocess_cfg = cfg.get("preprocess", {})
    recognition_cfg = cfg["recognition"]

    # 默认复用 recognition.infer_dir，但这里 infer_dir 应该放原始图，不再放预处理后的 112x112 图。
    raw_infer_dir = resolve_path(project_root, recognition_cfg.get("raw_infer_dir", recognition_cfg["infer_dir"]))
    output_dir = resolve_path(project_root, recognition_cfg["output_dir"])
    result_csv_path = output_dir / recognition_cfg.get("result_csv", "recognition_results.csv")

    yolo_weight = resolve_path(project_root, weights_cfg["yolo_face"])
    retinaface_weight = resolve_path(project_root, weights_cfg["retinaface"])
    recog_model_path = resolve_path(project_root, weights_cfg["arcface_onnx"])

    label_csv_path = resolve_path(project_root, recognition_cfg["label_csv"])
    gallery_features_path = resolve_path(project_root, recognition_cfg["gallery_features"])
    gallery_labels_path = resolve_path(project_root, recognition_cfg["gallery_labels"])
    gallery_person_index_path = resolve_path(project_root, recognition_cfg["gallery_person_index"])

    input_size = int(recognition_cfg.get("input_size", 112))
    embedding_dim = int(recognition_cfg.get("embedding_dim", 512))
    device = str(recognition_cfg.get("device", "auto"))
    threshold = float(recognition_cfg.get("threshold", 0.25))
    top_k = int(recognition_cfg.get("top_k", 5))
    aggregate_method = str(recognition_cfg.get("aggregate_method", "max"))
    aggregate_top_k = int(recognition_cfg.get("aggregate_top_k", 3))
    image_exts = recognition_cfg.get("image_exts", [".jpg", ".jpeg", ".png", ".bmp"])
    recursive = bool(recognition_cfg.get("recursive", True))

    fallback_to_yolo_crop = bool(recognition_cfg.get("fallback_to_yolo_crop", preprocess_cfg.get("fallback_to_yolo_crop", True)))
    yolo_conf = float(recognition_cfg.get("yolo_conf", preprocess_cfg.get("yolo_conf", 0.5)))
    expand_ratio = float(recognition_cfg.get("face_expand_ratio", preprocess_cfg.get("face_expand_ratio", 0.3)))
    retinaface_backbone = str(recognition_cfg.get("retinaface_backbone", preprocess_cfg.get("retinaface_backbone", "resnet50")))
    retinaface_confidence = float(recognition_cfg.get("retinaface_confidence", preprocess_cfg.get("retinaface_confidence", 0.5)))
    retinaface_nms = float(recognition_cfg.get("retinaface_nms", preprocess_cfg.get("retinaface_nms", 0.4)))
    retinaface_device = recognition_cfg.get("retinaface_device", preprocess_cfg.get("retinaface_device", None))
    if retinaface_device == "null":
        retinaface_device = None

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[INFO] 开始批量人脸识别：原始图片 -> 内存裁剪/关键点对齐 -> 特征提取 -> 多模板匹配")
    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] raw_infer_dir: {raw_infer_dir}")
    print(f"[INFO] yolo_weight: {yolo_weight}")
    print(f"[INFO] retinaface_weight: {retinaface_weight}")
    print(f"[INFO] recognition_model: {recog_model_path}")
    print(f"[INFO] gallery_features: {gallery_features_path}")
    print(f"[INFO] threshold: {threshold}")
    print(f"[INFO] aggregate_method: {aggregate_method}")
    print(f"[INFO] aggregate_top_k: {aggregate_top_k}")
    print(f"[INFO] fallback_to_yolo_crop: {fallback_to_yolo_crop}")
    print("=" * 80)

    for p, name in [(yolo_weight, "YOLO 权重"), (retinaface_weight, "RetinaFace 权重"), (recog_model_path, "识别 ONNX 模型")]:
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
    raw_images = collect_raw_images(raw_infer_dir, image_exts, recursive=recursive)
    print(f"[INFO] 原始图片目录下发现待识别图片数量: {len(raw_images)}")
    if len(raw_images) == 0:
        print("[WARN] 原始图片目录下没有找到待识别图片。")
        return

    records: List[Dict[str, Any]] = []
    align_success = 0
    fallback_count = 0
    align_failed = 0
    feature_failed = 0
    known_count = 0
    unknown_count = 0

    for idx, image_path in enumerate(raw_images, start=1):
        print(f"[INFO] [{idx}/{len(raw_images)}] 识别原始图片: {image_path}")
        base_record = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "align_status": "",
            "detect_info": "",
            "pred_status": "Failed",
            "pred_id": "",
            "pred_name": "",
            "person_score": "",
            "top1_candidate_id": "",
            "top1_candidate_name": "",
            "top1_candidate_score": "",
            "top1_best_template": "",
            "top1_best_template_score": "",
            "top1_num_templates": "",
            "threshold": f"{threshold:.4f}",
            "aggregate_method": aggregate_method,
            "aggregate_top_k": aggregate_top_k,
            "topk": "",
        }

        try:
            aligned_bgr, align_status, detect_info = align_raw_face_in_memory(
                image_path=image_path,
                yolo_model=yolo_model,
                landmark_extractor=landmark_extractor,
                input_size=input_size,
                yolo_conf=yolo_conf,
                expand_ratio=expand_ratio,
                fallback_to_yolo_crop=fallback_to_yolo_crop,
            )
            base_record["align_status"] = align_status
            base_record["detect_info"] = detect_info

            if aligned_bgr is None:
                align_failed += 1
                print(f"[WARN] 人脸裁剪/对齐失败: {image_path}, 原因: {align_status}")
                records.append(base_record)
                continue

            align_success += 1
            if align_status == "fallback_yolo_crop":
                fallback_count += 1

            query_feature = extract_feature_from_aligned_bgr(
                session=session,
                input_name=input_name,
                aligned_bgr=aligned_bgr,
                input_size=input_size,
                embedding_dim=embedding_dim,
            )

            if query_feature is None:
                feature_failed += 1
                print(f"[WARN] 特征提取失败: {image_path}")
                records.append(base_record)
                continue

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
                records.append(base_record)
                continue

            top1 = topk_results[0]
            final_pred = get_final_prediction(top1, threshold)
            if final_pred["pred_status"] == "Known":
                known_count += 1
            else:
                unknown_count += 1

            print_recognition_result(image_path, topk_results, threshold, align_status)

            base_record.update(
                {
                    "pred_status": final_pred["pred_status"],
                    "pred_id": final_pred["pred_id"],
                    "pred_name": final_pred["pred_name"],
                    "person_score": f"{final_pred['score']:.4f}",
                    "top1_candidate_id": top1["employee_id"],
                    "top1_candidate_name": top1["name"],
                    "top1_candidate_score": f"{top1['score']:.4f}",
                    "top1_best_template": top1.get("best_template_image", ""),
                    "top1_best_template_score": f"{top1.get('best_template_score', 0.0):.4f}",
                    "top1_num_templates": top1.get("num_templates", ""),
                    "topk": format_topk_for_csv(topk_results),
                }
            )
            records.append(base_record)

        except Exception as e:
            print(f"[ERROR] 图片处理异常: {image_path}, 原因: {e}")
            base_record["align_status"] = base_record.get("align_status", "") or "exception"
            base_record["detect_info"] = f"exception={e}"
            records.append(base_record)
            continue

    save_results_csv(records, result_csv_path)

    print("=" * 80)
    print("[INFO] 原始图片批量识别完成")
    print(f"[INFO] 总图片数: {len(raw_images)}")
    print(f"[INFO] 人脸裁剪/对齐成功: {align_success}")
    print(f"[INFO] RetinaFace 失败但 YOLO crop 兜底: {fallback_count}")
    print(f"[INFO] 人脸裁剪/对齐失败: {align_failed}")
    print(f"[INFO] 特征提取失败: {feature_failed}")
    print(f"[INFO] Known: {known_count}")
    print(f"[INFO] Unknown: {unknown_count}")
    print(f"[INFO] 结果文件: {result_csv_path}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize raw face images with in-memory YOLO crop, RetinaFace alignment, ArcFace ONNX, and multi-template gallery."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recognize_raw_images(args.config)
