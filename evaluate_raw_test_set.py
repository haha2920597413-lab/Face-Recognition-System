import argparse
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


def load_config(config_path: str) -> Dict:
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
            return None

        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return None

        return img

    except Exception:
        return None


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def create_onnx_session(model_path: Path, device: str = "auto") -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX 模型文件不存在: {model_path}")

    available_providers = ort.get_available_providers()

    if device == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "当前环境不支持 CUDAExecutionProvider，请检查 onnxruntime-gpu 是否安装正确。"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    elif device == "cpu":
        providers = ["CPUExecutionProvider"]

    elif device == "auto":
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    else:
        raise ValueError(f"不支持的 device 配置: {device}")

    print(f"[INFO] ONNX Runtime providers: {providers}")

    return ort.InferenceSession(str(model_path), providers=providers)


def inspect_model(session: ort.InferenceSession) -> Tuple[str, List]:
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    print(f"[INFO] ONNX input name: {input_name}")
    print(f"[INFO] ONNX input shape: {input_shape}")

    return input_name, input_shape


def detect_best_face_yolo(
    img: np.ndarray,
    yolo_model: YOLO,
    conf_thresh: float,
) -> Optional[Tuple[int, int, int, int, float]]:
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

    要求项目中存在：
        preprocess/retinafacetorch.py

    且其中包含：
        RetinaFaceTorchDetector
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


def align_raw_face_in_memory(
    image_path: Path,
    yolo_model: YOLO,
    landmark_extractor: RetinaFaceLandmarkExtractor,
    input_size: int,
    yolo_conf: float,
    expand_ratio: float,
    fallback_to_yolo_crop: bool,
) -> Tuple[Optional[np.ndarray], str]:
    """
    对原始测试图片做人脸检测、裁剪、关键点对齐。

    不保存任何中间图片，只返回内存中的 aligned face。
    """
    img = read_image_unicode(image_path)

    if img is None:
        return None, "image_read_failed"

    yolo_result = detect_best_face_yolo(
        img=img,
        yolo_model=yolo_model,
        conf_thresh=yolo_conf,
    )

    if yolo_result is None:
        return None, "yolo_face_not_detected"

    x1, y1, x2, y2, _ = yolo_result

    expanded_bbox = expand_bbox(
        bbox=(x1, y1, x2, y2),
        img_shape=img.shape,
        expand_ratio=expand_ratio,
    )

    crop = crop_by_bbox(img, expanded_bbox)

    if crop is None:
        return None, "crop_failed"

    landmarks = landmark_extractor.extract(crop)

    if landmarks is not None:
        aligned = align_face_by_landmarks(
            crop_bgr=crop,
            landmarks=landmarks,
            output_size=input_size,
        )

        if aligned is not None:
            return aligned, "aligned"

    if fallback_to_yolo_crop:
        try:
            fallback = cv2.resize(crop, (input_size, input_size))
            return fallback, "fallback_yolo_crop"
        except Exception:
            return None, "fallback_resize_failed"

    return None, "retinaface_landmark_failed"


def preprocess_aligned_bgr(aligned_bgr: np.ndarray, input_size: int) -> Optional[np.ndarray]:
    """
    aligned_bgr 是内存中的 112x112 BGR 人脸图。
    转成 ONNX 输入。
    """
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

    embedding = np.asarray(outputs[0]).reshape(-1)

    if embedding.shape[0] != embedding_dim:
        raise ValueError(
            f"模型输出维度异常，期望 {embedding_dim}, 实际 {embedding.shape[0]}"
        )

    embedding = embedding.astype(np.float32)
    embedding = l2_normalize(embedding)

    return embedding


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
            f"gallery_features 特征维度异常，"
            f"期望 {embedding_dim}, 实际 {gallery_features.shape[1]}"
        )

    if not isinstance(gallery_labels, list):
        raise ValueError("gallery_labels.json 格式异常：多模板版本应为 list[dict]。")

    if len(gallery_labels) > 0 and not isinstance(gallery_labels[0], dict):
        raise ValueError(
            "gallery_labels.json 看起来是旧版平均特征库格式。"
            "请重新运行多模板版 build_gallery.py。"
        )

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

    id_to_name = {}

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


def collect_test_images(test_dir: Path, image_exts: List[str]) -> List[Path]:
    if not test_dir.exists():
        raise FileNotFoundError(f"测试目录不存在: {test_dir}")

    image_exts = [ext.lower() for ext in image_exts]

    image_paths = []

    for p in sorted(test_dir.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in image_exts:
            image_paths.append(p)

    return image_paths


def get_true_id_from_filename(image_path: Path) -> str:
    """
    test/101024.jpg -> 101024
    如果文件名是 101024_xxx.jpg，也会取第一段 101024。
    """
    stem = image_path.stem.strip()
    if "_" in stem:
        return stem.split("_")[0]
    return stem


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
    """
    返回按身份聚合后的 Top-K。
    每个身份的 score 是最终预测分数：
        max 模式：最高模板分数
        topk_mean 模式：最高 k 个模板分数平均
    """
    all_template_scores = gallery_features @ query_feature

    person_results = []

    for identity_id, indices in person_index.items():
        if len(indices) == 0:
            continue

        idx_array = np.asarray(indices, dtype=np.int64)
        template_scores = all_template_scores[idx_array]

        final_person_score = aggregate_scores_for_person(
            template_scores=template_scores,
            method=aggregate_method,
            top_k=aggregate_top_k,
        )

        best_local_pos = int(np.argmax(template_scores))
        best_feature_index = int(idx_array[best_local_pos])
        best_template_score = float(template_scores[best_local_pos])

        label_record = gallery_labels[best_feature_index]
        name = id_to_name.get(identity_id, label_record.get("name", "UnknownName"))

        person_results.append({
            "identity_id": identity_id,
            "name": name,
            "score": final_person_score,
            "best_template_score": best_template_score,
            "best_template_image": label_record.get("image_name", ""),
            "best_template_path": label_record.get("image_path", ""),
            "num_templates": len(indices),
        })

    person_results = sorted(person_results, key=lambda x: x["score"], reverse=True)

    top_k = min(top_k, len(person_results))

    topk_results = []

    for rank, item in enumerate(person_results[:top_k], start=1):
        item = dict(item)
        item["rank"] = rank
        topk_results.append(item)

    return topk_results


def describe_scores(scores: List[float]) -> Dict[str, Optional[float]]:
    if len(scores) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "q25": None,
            "q75": None,
        }

    arr = np.asarray(scores, dtype=np.float32)

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def print_score_stats(title: str, scores: List[float]) -> None:
    stats = describe_scores(scores)

    print(f"\n---------------- {title} ----------------")
    print(f"count : {stats['count']}")

    if stats["count"] == 0:
        print("无数据")
        return

    print(f"mean  : {stats['mean']:.6f}")
    print(f"median: {stats['median']:.6f}")
    print(f"min   : {stats['min']:.6f}")
    print(f"max   : {stats['max']:.6f}")
    print(f"std   : {stats['std']:.6f}")
    print(f"q25   : {stats['q25']:.6f}")
    print(f"q75   : {stats['q75']:.6f}")


def print_examples(title: str, records: List[Dict[str, Any]], top_n: int) -> None:
    print(f"\n---------------- {title} Top {top_n} ----------------")

    if len(records) == 0:
        print("无")
        return

    for i, r in enumerate(records[:top_n], start=1):
        print(
            f"{i}. image={r['image_name']} "
            f"true={r['true_id']}({r.get('true_name', '')}) "
            f"pred={r.get('pred_id', '')}({r.get('pred_name', '')}) "
            f"score={r.get('score', ''):.6f} "
            f"best_template={r.get('best_template_image', '')} "
            f"best_template_score={r.get('best_template_score', 0.0):.6f} "
            f"reason={r.get('reason', '')}"
        )


def evaluate_raw_test_set(config_path: str) -> None:
    cfg = load_config(config_path)

    project_root = Path(cfg["project"]["root"]).expanduser().resolve()

    weights_cfg = cfg["weights"]
    preprocess_cfg = cfg.get("preprocess", {})
    test_cfg = cfg["test_evaluation"]

    raw_test_dir = resolve_path(project_root, test_cfg["raw_test_dir"])
    label_csv_path = resolve_path(project_root, test_cfg["label_csv"])

    yolo_weight = resolve_path(project_root, weights_cfg["yolo_face"])
    retinaface_weight = resolve_path(project_root, weights_cfg["retinaface"])
    recog_model_path = resolve_path(project_root, weights_cfg["arcface_onnx"])

    gallery_features_path = resolve_path(project_root, test_cfg["gallery_features"])
    gallery_labels_path = resolve_path(project_root, test_cfg["gallery_labels"])
    gallery_person_index_path = resolve_path(project_root, test_cfg["gallery_person_index"])

    input_size = int(test_cfg.get("input_size", 112))
    embedding_dim = int(test_cfg.get("embedding_dim", 512))
    device = str(test_cfg.get("device", "auto"))

    threshold = float(test_cfg.get("threshold", 0.25))
    top_k = int(test_cfg.get("top_k", 5))
    aggregate_method = str(test_cfg.get("aggregate_method", "max"))
    aggregate_top_k = int(test_cfg.get("aggregate_top_k", 3))

    image_exts = test_cfg.get("image_exts", [".jpg", ".jpeg", ".png", ".bmp"])
    print_top_n = int(test_cfg.get("print_top_n", 10))

    fallback_to_yolo_crop = bool(test_cfg.get("fallback_to_yolo_crop", True))

    yolo_conf = float(preprocess_cfg.get("yolo_conf", 0.5))
    expand_ratio = float(preprocess_cfg.get("face_expand_ratio", 0.20))

    retinaface_backbone = str(preprocess_cfg.get("retinaface_backbone", "resnet50"))
    retinaface_confidence = float(preprocess_cfg.get("retinaface_confidence", 0.5))
    retinaface_nms = float(preprocess_cfg.get("retinaface_nms", 0.4))
    retinaface_device = preprocess_cfg.get("retinaface_device", None)

    if retinaface_device == "null":
        retinaface_device = None

    print("=" * 80)
    print("[INFO] 原始测试集评估：不保存任何中间文件")
    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] raw_test_dir: {raw_test_dir}")
    print(f"[INFO] yolo_weight: {yolo_weight}")
    print(f"[INFO] retinaface_weight: {retinaface_weight}")
    print(f"[INFO] recognition_model: {recog_model_path}")
    print(f"[INFO] gallery_features: {gallery_features_path}")
    print(f"[INFO] gallery_labels: {gallery_labels_path}")
    print(f"[INFO] gallery_person_index: {gallery_person_index_path}")
    print(f"[INFO] aggregate_method: {aggregate_method}")
    print(f"[INFO] aggregate_top_k: {aggregate_top_k}")
    print(f"[INFO] threshold: {threshold}")
    print("=" * 80)

    if not yolo_weight.exists():
        raise FileNotFoundError(f"YOLO 权重不存在: {yolo_weight}")

    if not retinaface_weight.exists():
        raise FileNotFoundError(f"RetinaFace 权重不存在: {retinaface_weight}")

    if not recog_model_path.exists():
        raise FileNotFoundError(f"识别 ONNX 模型不存在: {recog_model_path}")

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

    test_images = collect_test_images(raw_test_dir, image_exts)

    total_images = len(test_images)

    print(f"\n[INFO] test 目录下发现图片数量: {total_images}")

    if total_images == 0:
        print("[WARN] test 目录下没有图片。")
        return

    align_success = 0
    align_failed = 0
    fallback_count = 0

    feature_success = 0
    feature_failed = 0

    correct_count = 0
    wrong_count = 0
    unknown_count = 0

    all_scores: List[float] = []
    correct_scores: List[float] = []
    wrong_scores: List[float] = []
    unknown_scores: List[float] = []

    wrong_records: List[Dict[str, Any]] = []
    unknown_records: List[Dict[str, Any]] = []
    align_failed_records: List[Dict[str, Any]] = []
    feature_failed_records: List[Dict[str, Any]] = []
    low_correct_records: List[Dict[str, Any]] = []
    high_wrong_records: List[Dict[str, Any]] = []

    wrong_true_ids = set()
    unknown_true_ids = set()
    align_failed_true_ids = set()
    feature_failed_true_ids = set()

    for idx, image_path in enumerate(test_images, start=1):
        true_id = get_true_id_from_filename(image_path)
        true_name = id_to_name.get(true_id, "UnknownName")

        print(f"[INFO] [{idx}/{total_images}] 评估: {image_path.name}")

        aligned_bgr, align_status = align_raw_face_in_memory(
            image_path=image_path,
            yolo_model=yolo_model,
            landmark_extractor=landmark_extractor,
            input_size=input_size,
            yolo_conf=yolo_conf,
            expand_ratio=expand_ratio,
            fallback_to_yolo_crop=fallback_to_yolo_crop,
        )

        if aligned_bgr is None:
            align_failed += 1
            align_failed_true_ids.add(true_id)

            align_failed_records.append({
                "image_name": image_path.name,
                "true_id": true_id,
                "true_name": true_name,
                "pred_id": "",
                "pred_name": "",
                "score": 0.0,
                "best_template_image": "",
                "best_template_score": 0.0,
                "reason": align_status,
            })

            continue

        align_success += 1

        if align_status == "fallback_yolo_crop":
            fallback_count += 1

        try:
            query_feature = extract_feature_from_aligned_bgr(
                session=session,
                input_name=input_name,
                aligned_bgr=aligned_bgr,
                input_size=input_size,
                embedding_dim=embedding_dim,
            )
        except Exception as e:
            query_feature = None
            align_status = f"feature_exception: {e}"

        if query_feature is None:
            feature_failed += 1
            feature_failed_true_ids.add(true_id)

            feature_failed_records.append({
                "image_name": image_path.name,
                "true_id": true_id,
                "true_name": true_name,
                "pred_id": "",
                "pred_name": "",
                "score": 0.0,
                "best_template_image": "",
                "best_template_score": 0.0,
                "reason": "feature_extract_failed",
            })

            continue

        feature_success += 1

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
            feature_failed += 1
            feature_failed_true_ids.add(true_id)
            continue

        top1 = topk_results[0]

        pred_id = str(top1["identity_id"])
        pred_name = str(top1["name"])

        # 关键点：
        # final_score 是多模板聚合后的最终预测分数。
        # max 时是最高模板分数；
        # topk_mean 时是最高 k 个模板分数均值。
        final_score = float(top1["score"])

        best_template_image = top1.get("best_template_image", "")
        best_template_score = float(top1.get("best_template_score", 0.0))

        all_scores.append(final_score)

        record = {
            "image_name": image_path.name,
            "true_id": true_id,
            "true_name": true_name,
            "pred_id": pred_id,
            "pred_name": pred_name,
            "score": final_score,
            "best_template_image": best_template_image,
            "best_template_score": best_template_score,
            "reason": "",
        }

        if final_score < threshold:
            unknown_count += 1
            unknown_true_ids.add(true_id)
            unknown_scores.append(final_score)

            record["reason"] = "below_threshold"
            unknown_records.append(record)

        elif pred_id == true_id:
            correct_count += 1
            correct_scores.append(final_score)

            record["reason"] = "correct"
            low_correct_records.append(record)

        else:
            wrong_count += 1
            wrong_true_ids.add(true_id)
            wrong_scores.append(final_score)

            record["reason"] = "wrong_prediction"
            wrong_records.append(record)
            high_wrong_records.append(record)

    evaluated = feature_success
    total = total_images

    overall_accuracy = correct_count / total if total > 0 else 0.0
    evaluated_accuracy = correct_count / evaluated if evaluated > 0 else 0.0

    known_total = correct_count + wrong_count
    known_accuracy = correct_count / known_total if known_total > 0 else 0.0

    wrong_rate = wrong_count / total if total > 0 else 0.0
    unknown_rate = unknown_count / total if total > 0 else 0.0
    align_failed_rate = align_failed / total if total > 0 else 0.0
    feature_failed_rate = feature_failed / total if total > 0 else 0.0

    low_correct_records = sorted(low_correct_records, key=lambda x: x["score"])
    high_wrong_records = sorted(high_wrong_records, key=lambda x: x["score"], reverse=True)
    unknown_records_sorted = sorted(unknown_records, key=lambda x: x["score"], reverse=True)
    align_failed_records = align_failed_records[:print_top_n]
    feature_failed_records = feature_failed_records[:print_top_n]

    print("\n" + "=" * 80)
    print("原始测试集评估结果")
    print("=" * 80)

    print(f"测试集总图片数 total_images: {total_images}")
    print(f"人脸裁剪对齐成功数 align_success: {align_success}")
    print(f"人脸裁剪对齐失败数 align_failed: {align_failed}")
    print(f"RetinaFace 失败但使用 YOLO 裁剪兜底数 fallback_count: {fallback_count}")

    print(f"有效识别数 evaluated: {evaluated}")
    print(f"特征提取失败数 feature_failed: {feature_failed}")

    print(f"\n预测正确数 correct_count: {correct_count}")
    print(f"预测错误数 wrong_count: {wrong_count}")
    print(f"Unknown 数 unknown_count: {unknown_count}")

    print(f"\n被识别错的人数 unique_wrong_people: {len(wrong_true_ids)}")
    print(f"被判定为 Unknown 的人数 unique_unknown_people: {len(unknown_true_ids)}")
    print(f"裁剪对齐失败人数 unique_align_failed_people: {len(align_failed_true_ids)}")
    print(f"特征提取失败人数 unique_feature_failed_people: {len(feature_failed_true_ids)}")

    print("\n---------------- 指标 ----------------")
    print(f"整体准确率 overall_accuracy = correct / total: {overall_accuracy:.6f}")
    print(f"有效样本准确率 evaluated_accuracy = correct / evaluated: {evaluated_accuracy:.6f}")
    print(f"Known 样本准确率 known_accuracy = correct / (correct + wrong): {known_accuracy:.6f}")
    print(f"误识率 wrong_rate = wrong / total: {wrong_rate:.6f}")
    print(f"拒识率 unknown_rate = unknown / total: {unknown_rate:.6f}")
    print(f"裁剪失败率 align_failed_rate = align_failed / total: {align_failed_rate:.6f}")
    print(f"特征失败率 feature_failed_rate = feature_failed / total: {feature_failed_rate:.6f}")

    print(f"\n聚合方式 aggregate_method: {aggregate_method}")
    print(f"聚合 top_k aggregate_top_k: {aggregate_top_k}")
    print(f"Unknown 阈值 threshold: {threshold:.6f}")
    print("注意：以下分数统计全部基于最终预测分数 final_score，即 top1_result['score']。")

    print_score_stats("全部有效样本 final_score", all_scores)
    print_score_stats("预测正确样本 final_score", correct_scores)
    print_score_stats("预测错误样本 final_score", wrong_scores)
    print_score_stats("Unknown 样本 final_score", unknown_scores)

    print_examples("最高分错误样本", high_wrong_records, print_top_n)
    print_examples("最低分正确样本", low_correct_records, print_top_n)
    print_examples("接近阈值但被 Unknown 的样本", unknown_records_sorted, print_top_n)
    print_examples("裁剪对齐失败样本", align_failed_records, print_top_n)
    print_examples("特征提取失败样本", feature_failed_records, print_top_n)

    print("\n" + "=" * 80)
    print("评估完成：未保存任何中间图片、CSV 或 JSON 文件。")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate raw test images with in-memory face crop alignment and multi-template gallery."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_raw_test_set(args.config)