import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml


def load_config(config_path: str) -> Dict:
    """
    读取 config.yaml 配置文件。
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"配置文件为空: {config_path}")

    return config


def resolve_path(project_root: Path, path_str: str) -> Path:
    """
    将配置文件中的路径解析为绝对路径。
    """
    p = Path(path_str)

    if p.is_absolute():
        return p

    return project_root / p


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    对 embedding 做 L2 normalize。
    """
    norm = np.linalg.norm(vector)

    if norm < eps:
        return vector

    return vector / norm


def read_image_unicode(image_path: Path) -> Optional[np.ndarray]:
    """
    支持中文路径/中文文件名的 OpenCV 图片读取函数。
    """
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)

        if image_bytes.size == 0:
            print(f"[WARN] 图片文件为空或无法读取字节: {image_path}")
            return None

        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            print(f"[WARN] cv2.imdecode 解码失败: {image_path}")
            return None

        return img

    except Exception as e:
        print(f"[WARN] 图片读取异常: {image_path}, 原因: {e}")
        return None


def _try_create_session(model_path: Path, providers: List[str]) -> ort.InferenceSession:
    """
    按指定 provider 创建 ONNX Runtime session。
    单独封装是为了在 auto 模式下 CUDA 失败时可以回退到 CPU。
    """
    return ort.InferenceSession(
        str(model_path),
        providers=providers
    )


def create_onnx_session(model_path: Path, device: str = "auto") -> ort.InferenceSession:
    """
    创建 ONNX Runtime 推理会话。

    device:
        - "auto": 优先使用 CUDA；如果 CUDA DLL / cuDNN / CUDA 版本不匹配，则自动回退 CPU
        - "cuda": 强制使用 CUDA；如果 CUDA 创建失败，直接报错
        - "cpu": 强制使用 CPU；不会加载 CUDAExecutionProvider，因此不会触发 CUDA DLL 报错
    """
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
                "当前环境没有 CUDAExecutionProvider。请检查是否安装 onnxruntime-gpu，"
                "以及 CUDA / cuDNN / MSVC Runtime 是否匹配；或者将 gallery.device 设置为 cpu。"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"[INFO] ONNX Runtime providers: {providers}")

        try:
            return _try_create_session(model_path, providers)
        except Exception as e:
            raise RuntimeError(
                "强制使用 CUDAExecutionProvider 失败。\n"
                "常见原因：缺少 cublasLt64_12.dll、CUDA 12 / cuDNN 9 未安装或未加入 PATH、"
                "onnxruntime-gpu 版本与 CUDA/cuDNN 不匹配。\n"
                "如果暂时不使用 GPU，请将 config.yaml 中 gallery.device 设置为 cpu。\n"
                f"原始错误: {e}"
            ) from e

    if device == "auto":
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"[INFO] ONNX Runtime providers: {providers}")

            try:
                return _try_create_session(model_path, providers)
            except Exception as e:
                print("[WARN] CUDAExecutionProvider 创建失败，将自动回退到 CPUExecutionProvider。")
                print(f"[WARN] CUDA 原始错误: {e}")

        providers = ["CPUExecutionProvider"]
        print(f"[INFO] ONNX Runtime providers: {providers}")
        return _try_create_session(model_path, providers)

    raise ValueError(f"不支持的 gallery.device 配置: {device}，请使用 auto / cuda / cpu。")


def preprocess_image(image_path: Path, input_size: int = 112) -> Optional[np.ndarray]:
    """
    读取并预处理单张人脸图像。

    处理流程：
        1. 支持中文路径读取 BGR 图像
        2. BGR -> RGB
        3. resize 到 input_size x input_size
        4. 转 float32
        5. 归一化到 [-1, 1]
        6. HWC -> CHW
        7. 增加 batch 维度

    返回:
        np.ndarray, shape = [1, 3, 112, 112]
    """
    img = read_image_unicode(image_path)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)

    # InsightFace / ArcFace 常用归一化方式
    img = (img - 127.5) / 127.5

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # [3, H, W] -> [1, 3, H, W]
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


def extract_embedding(
    session: ort.InferenceSession,
    input_name: str,
    image_path: Path,
    input_size: int,
    embedding_dim: int
) -> Optional[np.ndarray]:
    """
    对单张图片提取 embedding，并做 L2 normalize。
    """
    input_blob = preprocess_image(image_path, input_size=input_size)

    if input_blob is None:
        return None

    outputs = session.run(None, {input_name: input_blob})

    if outputs is None or len(outputs) == 0:
        return None

    embedding = outputs[0]
    embedding = np.asarray(embedding).reshape(-1)

    if embedding.shape[0] != embedding_dim:
        raise ValueError(
            f"模型输出维度异常: {image_path}, "
            f"期望 {embedding_dim}, 实际 {embedding.shape[0]}"
        )

    embedding = embedding.astype(np.float32)
    embedding = l2_normalize(embedding)

    return embedding


def collect_identity_images(
    identity_dir: Path,
    image_exts: List[str]
) -> List[Path]:
    """
    收集某个身份文件夹下的所有图片，并按文件名排序。
    """
    image_exts = [ext.lower() for ext in image_exts]

    image_paths = []

    for p in identity_dir.iterdir():
        if p.is_file() and p.suffix.lower() in image_exts:
            image_paths.append(p)

    image_paths = sorted(image_paths, key=lambda x: x.name)

    return image_paths


def parse_name_from_image(identity_id: str, image_path: Path) -> str:
    """
    从文件名中解析姓名。

    期望文件名格式：
        101024_王金霞_000.jpg

    如果解析失败，返回空字符串。
    """
    stem = image_path.stem
    parts = stem.split("_")

    if len(parts) >= 3 and parts[0] == identity_id:
        return parts[1]

    if len(parts) >= 2:
        return parts[1]

    return ""


def build_identity_templates(
    session: ort.InferenceSession,
    input_name: str,
    identity_id: str,
    identity_dir: Path,
    image_exts: List[str],
    input_size: int,
    embedding_dim: int,
    expected_images: int
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict]:
    """
    为单个身份构建多模板特征。

    不再求平均，而是：
        每张图片 -> 一个 embedding
        一个身份 -> 多个 template embedding
    """
    image_paths = collect_identity_images(identity_dir, image_exts)

    log_info = {
        "identity_id": identity_id,
        "identity_dir": str(identity_dir),
        "num_images_found": len(image_paths),
        "num_images_success": 0,
        "num_images_failed": 0,
        "num_templates_added": 0,
        "status": "",
        "reason": ""
    }

    if len(image_paths) == 0:
        log_info["status"] = "skipped"
        log_info["reason"] = "no_images"
        return [], [], log_info

    embeddings: List[np.ndarray] = []
    label_records: List[Dict[str, Any]] = []

    for local_template_index, image_path in enumerate(image_paths):
        try:
            embedding = extract_embedding(
                session=session,
                input_name=input_name,
                image_path=image_path,
                input_size=input_size,
                embedding_dim=embedding_dim
            )

            if embedding is None:
                log_info["num_images_failed"] += 1
                continue

            name = parse_name_from_image(identity_id, image_path)

            # feature_index 会在主流程 append 时统一补充
            label_record = {
                "identity_id": identity_id,
                "name": name,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "template_index_in_identity": local_template_index
            }

            embeddings.append(embedding.astype(np.float32))
            label_records.append(label_record)
            log_info["num_images_success"] += 1

        except Exception as e:
            log_info["num_images_failed"] += 1
            print(f"[WARN] 图片处理失败: {image_path}, 原因: {e}")

    if len(embeddings) == 0:
        log_info["status"] = "failed"
        log_info["reason"] = "all_images_failed"
        return [], [], log_info

    log_info["status"] = "success"
    log_info["num_templates_added"] = len(embeddings)

    if log_info["num_images_found"] < expected_images:
        log_info["reason"] = "incomplete_but_used"
    elif log_info["num_images_success"] < log_info["num_images_found"]:
        log_info["reason"] = "partial_images_used"
    else:
        log_info["reason"] = "ok"

    return embeddings, label_records, log_info


def atomic_write_json(obj: Any, path: Path) -> None:
    """
    原子写入 JSON 文件。

    先写入同目录 .tmp 文件，flush + fsync 确认落盘后，再用 os.replace 替换正式文件。
    这样程序中断时，正式 JSON 文件不会变成半截文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_npy(array: np.ndarray, path: Path) -> None:
    """
    原子写入 .npy 文件。

    注意：不能直接 np.save("xxx.npy.tmp", array)，否则 numpy 可能生成 xxx.npy.tmp.npy。
    因此这里使用文件句柄写入。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_json_with_clear_error(path: Path, expected_type: type, description: str) -> Any:
    """
    读取 JSON，并在 JSON 损坏时给出明确提示。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{description} 已损坏，通常是上一次运行中断时写入不完整导致的。\n"
            f"文件路径: {path}\n"
            "处理方法：\n"
            "  1. 删除 gallery 目录下的 gallery_features.npy、gallery_labels.json、"
            "gallery_person_index.json 后重新运行；或\n"
            "  2. 将 config.yaml 中 gallery.overwrite 设置为 true 后重新运行。\n"
            f"原始 JSON 错误: {e}"
        ) from e

    except FileNotFoundError:
        raise

    except Exception as e:
        raise RuntimeError(
            f"{description} 读取失败。\n"
            f"文件路径: {path}\n"
            f"原始错误: {e}"
        ) from e

    if not isinstance(obj, expected_type):
        raise ValueError(
            f"{description} 格式异常：期望类型 {expected_type.__name__}，"
            f"实际类型 {type(obj).__name__}。文件路径: {path}"
        )

    return obj


def load_existing_multi_template_gallery(
    features_path: Path,
    labels_path: Path,
    person_index_path: Path,
    embedding_dim: int
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, List[int]]]:
    """
    读取已有多模板 gallery，用于断点续跑。

    只有三个文件同时存在时才恢复：
        gallery_features.npy
        gallery_labels.json
        gallery_person_index.json

    如果文件不完整或损坏，会给出明确错误，避免继续在坏文件上追加。
    """
    existing_files = [
        features_path.exists(),
        labels_path.exists(),
        person_index_path.exists()
    ]

    if not any(existing_files):
        return [], [], {}

    if not all(existing_files):
        missing = []
        if not features_path.exists():
            missing.append(str(features_path))
        if not labels_path.exists():
            missing.append(str(labels_path))
        if not person_index_path.exists():
            missing.append(str(person_index_path))

        raise RuntimeError(
            "检测到 gallery 断点文件不完整，不能安全续跑。\n"
            f"缺失文件: {missing}\n"
            "请删除 gallery 目录下残留的 gallery_features.npy、gallery_labels.json、"
            "gallery_person_index.json 后重新运行，或设置 overwrite=true。"
        )

    try:
        features = np.load(str(features_path))
    except Exception as e:
        raise RuntimeError(
            f"gallery_features.npy 读取失败，可能是上一次运行中断导致文件损坏。\n"
            f"文件路径: {features_path}\n"
            "请删除 gallery 目录下已有结果后重新运行，或设置 overwrite=true。\n"
            f"原始错误: {e}"
        ) from e

    labels = load_json_with_clear_error(
        labels_path,
        expected_type=list,
        description="gallery_labels.json"
    )

    person_index = load_json_with_clear_error(
        person_index_path,
        expected_type=dict,
        description="gallery_person_index.json"
    )

    if len(features.shape) != 2:
        raise ValueError(f"已有 gallery_features.npy 维度异常: {features.shape}")

    if features.shape[1] != embedding_dim:
        raise ValueError(
            f"已有 gallery_features.npy 特征维度异常，"
            f"期望 {embedding_dim}, 实际 {features.shape[1]}"
        )

    if features.shape[0] != len(labels):
        raise ValueError(
            f"已有 gallery 文件不匹配: features 行数={features.shape[0]}, "
            f"labels 数量={len(labels)}。\n"
            "这通常是上一次运行中断时，三个断点文件没有保持一致导致的。\n"
            "请删除 gallery 目录下已有结果后重新运行，或设置 overwrite=true。"
        )

    if len(labels) > 0 and not isinstance(labels[0], dict):
        raise ValueError(
            "当前 gallery_labels.json 看起来是旧版平均特征库格式。"
            "请设置 gallery.overwrite=true 重新构建多模板特征库。"
        )

    feature_list = [features[i].astype(np.float32) for i in range(features.shape[0])]

    # 确保 person_index 的 value 是 int list
    normalized_person_index: Dict[str, List[int]] = {}
    for identity_id, indices in person_index.items():
        if not isinstance(indices, list):
            raise ValueError(
                f"gallery_person_index.json 格式异常：身份 {identity_id} 对应的 value 不是 list。"
            )
        normalized_person_index[str(identity_id)] = [int(i) for i in indices]

    # 进一步校验 person_index 里的索引是否越界
    num_features = len(feature_list)
    for identity_id, indices in normalized_person_index.items():
        for i in indices:
            if i < 0 or i >= num_features:
                raise ValueError(
                    f"gallery_person_index.json 中存在越界索引："
                    f"identity_id={identity_id}, index={i}, num_features={num_features}。"
                )

    print(
        f"[INFO] 已加载已有多模板 gallery: "
        f"{len(normalized_person_index)} 人，{len(feature_list)} 个模板"
    )

    return feature_list, labels, normalized_person_index


def save_multi_template_gallery(
    features: List[np.ndarray],
    labels: List[Dict[str, Any]],
    person_index: Dict[str, List[int]],
    features_path: Path,
    labels_path: Path,
    person_index_path: Path
) -> None:
    """
    保存多模板特征库：
        gallery_features.npy
        gallery_labels.json
        gallery_person_index.json

    重要改动：
        使用原子写入，避免程序中途被中断时把 JSON 写成半截文件。
    """
    if len(features) == 0:
        raise ValueError("没有可保存的 gallery 特征。")

    if len(features) != len(labels):
        raise ValueError(
            f"待保存的 gallery 数据不一致: features={len(features)}, labels={len(labels)}"
        )

    features_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    person_index_path.parent.mkdir(parents=True, exist_ok=True)

    features_array = np.stack(features, axis=0).astype(np.float32)

    # 先写临时文件，再原子替换正式文件。
    # 即使运行中断，正式 JSON 文件也不会变成半截 JSON。
    atomic_write_npy(features_array, features_path)
    atomic_write_json(labels, labels_path)
    atomic_write_json(person_index, person_index_path)

    print(f"[INFO] gallery_features 保存完成: {features_path}")
    print(f"[INFO] gallery_labels 保存完成: {labels_path}")
    print(f"[INFO] gallery_person_index 保存完成: {person_index_path}")
    print(f"[INFO] gallery_features shape: {features_array.shape}")
    print(f"[INFO] gallery_labels length: {len(labels)}")
    print(f"[INFO] gallery_person_index identities: {len(person_index)}")


def save_log(log_records: List[Dict], log_path: Path) -> None:
    """
    保存构建日志 CSV。
    """
    if len(log_records) == 0:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "identity_id",
        "identity_dir",
        "num_images_found",
        "num_images_success",
        "num_images_failed",
        "num_templates_added",
        "status",
        "reason"
    ]

    tmp_path = log_path.with_name(log_path.name + ".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in log_records:
                row = {key: record.get(key, "") for key in fieldnames}
                writer.writerow(row)

            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, log_path)

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    print(f"[INFO] 构建日志保存完成: {log_path}")


def inspect_model(session: ort.InferenceSession) -> Tuple[str, List]:
    """
    查看 ONNX 模型输入信息。
    """
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    print(f"[INFO] ONNX input name: {input_name}")
    print(f"[INFO] ONNX input shape: {input_shape}")

    return input_name, input_shape


def build_gallery(config_path: str) -> None:
    """
    主流程：构建多模板人脸 gallery。

    输出：
        gallery_features.npy:
            shape = [N_templates, embedding_dim]

        gallery_labels.json:
            list[dict]，每个 template 一个记录

        gallery_person_index.json:
            dict，identity_id -> feature_index list
    """
    config = load_config(config_path)

    project_root = Path(config["project"]["root"])

    weights_cfg = config["weights"]
    gallery_cfg = config["gallery"]

    gallery_mode = str(gallery_cfg.get("mode", "multi_template"))

    if gallery_mode != "multi_template":
        raise ValueError(
            f"当前 build_gallery.py 是多模板版本，"
            f"请将 gallery.mode 设置为 multi_template，当前为: {gallery_mode}"
        )

    identities_dir = resolve_path(project_root, gallery_cfg["input_dir"])
    gallery_dir = resolve_path(project_root, gallery_cfg["output_dir"])
    model_path = resolve_path(project_root, weights_cfg["arcface_onnx"])

    log_dir = resolve_path(project_root, config["paths"]["log_dir"])
    log_path = log_dir / gallery_cfg.get("log_file", "build_gallery_log.csv")

    features_path = gallery_dir / gallery_cfg.get("features_file", "gallery_features.npy")
    labels_path = gallery_dir / gallery_cfg.get("labels_file", "gallery_labels.json")
    person_index_path = gallery_dir / gallery_cfg.get(
        "person_index_file",
        "gallery_person_index.json"
    )

    input_size = int(gallery_cfg.get("input_size", 112))
    embedding_dim = int(gallery_cfg.get("embedding_dim", 512))
    expected_images = int(gallery_cfg.get("expected_images_per_identity", 40))

    image_exts = gallery_cfg.get("image_exts", [".jpg", ".jpeg", ".png", ".bmp"])

    # 你现在使用 device: cpu 是正确的。
    # 本代码在 device=cpu 时只启用 CPUExecutionProvider，不会再尝试加载 CUDA DLL。
    device = gallery_cfg.get("device", "auto")

    resume = bool(gallery_cfg.get("resume", True))
    overwrite = bool(gallery_cfg.get("overwrite", False))
    skip_existing_identity = bool(gallery_cfg.get("skip_existing_identity", True))
    skip_empty_identity = bool(gallery_cfg.get("skip_empty_identity", True))
    use_available_images_if_incomplete = bool(
        gallery_cfg.get("use_available_images_if_incomplete", True)
    )

    if not identities_dir.exists():
        raise FileNotFoundError(f"身份图片目录不存在: {identities_dir}")

    gallery_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[INFO] 开始构建多模板人脸 gallery")
    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] identities_dir: {identities_dir}")
    print(f"[INFO] gallery_dir: {gallery_dir}")
    print(f"[INFO] model_path: {model_path}")
    print(f"[INFO] gallery mode: {gallery_mode}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] resume: {resume}")
    print(f"[INFO] overwrite: {overwrite}")
    print("=" * 80)

    session = create_onnx_session(model_path=model_path, device=device)
    input_name, _ = inspect_model(session)

    gallery_features: List[np.ndarray] = []
    gallery_labels: List[Dict[str, Any]] = []
    gallery_person_index: Dict[str, List[int]] = {}

    if resume and not overwrite:
        gallery_features, gallery_labels, gallery_person_index = load_existing_multi_template_gallery(
            features_path=features_path,
            labels_path=labels_path,
            person_index_path=person_index_path,
            embedding_dim=embedding_dim
        )
    else:
        print("[INFO] overwrite=true 或 resume=false，将重新构建多模板 gallery。")

    existing_identity_set = set(gallery_person_index.keys())

    identity_dirs = [
        p for p in identities_dir.iterdir()
        if p.is_dir()
    ]

    identity_dirs = sorted(identity_dirs, key=lambda x: x.name)

    print(f"[INFO] identities 下发现身份数量: {len(identity_dirs)}")

    log_records = []

    num_success_new = 0
    num_skipped_existing = 0
    num_skipped_empty = 0
    num_failed = 0
    num_incomplete_used = 0
    num_templates_added_total = 0

    for idx, identity_dir in enumerate(identity_dirs, start=1):
        identity_id = identity_dir.name

        print(f"[INFO] [{idx}/{len(identity_dirs)}] 处理身份: {identity_id}")

        if skip_existing_identity and identity_id in existing_identity_set:
            num_skipped_existing += 1

            existing_templates = len(gallery_person_index.get(identity_id, []))

            log_records.append({
                "identity_id": identity_id,
                "identity_dir": str(identity_dir),
                "num_images_found": "",
                "num_images_success": "",
                "num_images_failed": "",
                "num_templates_added": existing_templates,
                "status": "skipped",
                "reason": "already_exists_in_gallery"
            })

            print(f"[INFO] 跳过已有身份: {identity_id}, 已有模板数: {existing_templates}")
            continue

        image_paths = collect_identity_images(identity_dir, image_exts)

        if len(image_paths) == 0 and skip_empty_identity:
            num_skipped_empty += 1

            log_records.append({
                "identity_id": identity_id,
                "identity_dir": str(identity_dir),
                "num_images_found": 0,
                "num_images_success": 0,
                "num_images_failed": 0,
                "num_templates_added": 0,
                "status": "skipped",
                "reason": "no_images"
            })

            print(f"[WARN] 身份 {identity_id} 没有图片，跳过。")
            continue

        if len(image_paths) < expected_images:
            if use_available_images_if_incomplete:
                print(
                    f"[WARN] 身份 {identity_id} 图片数量不足: "
                    f"{len(image_paths)}/{expected_images}，将使用现有图片构建多模板。"
                )
                num_incomplete_used += 1
            else:
                log_records.append({
                    "identity_id": identity_id,
                    "identity_dir": str(identity_dir),
                    "num_images_found": len(image_paths),
                    "num_images_success": 0,
                    "num_images_failed": 0,
                    "num_templates_added": 0,
                    "status": "skipped",
                    "reason": "incomplete_images"
                })
                print(f"[WARN] 身份 {identity_id} 图片数量不足，跳过。")
                continue

        try:
            identity_embeddings, identity_label_records, log_info = build_identity_templates(
                session=session,
                input_name=input_name,
                identity_id=identity_id,
                identity_dir=identity_dir,
                image_exts=image_exts,
                input_size=input_size,
                embedding_dim=embedding_dim,
                expected_images=expected_images
            )

            if len(identity_embeddings) == 0:
                log_records.append(log_info)

                if log_info["reason"] == "no_images":
                    num_skipped_empty += 1
                else:
                    num_failed += 1

                print(f"[WARN] 身份 {identity_id} 构建失败: {log_info['reason']}")
                continue

            start_index = len(gallery_features)
            identity_indices: List[int] = []

            for template_offset, embedding in enumerate(identity_embeddings):
                feature_index = start_index + template_offset

                label_record = identity_label_records[template_offset]
                label_record["feature_index"] = feature_index

                gallery_features.append(embedding)
                gallery_labels.append(label_record)
                identity_indices.append(feature_index)

            gallery_person_index[identity_id] = identity_indices
            existing_identity_set.add(identity_id)

            num_success_new += 1
            num_templates_added_total += len(identity_indices)

            log_info["num_templates_added"] = len(identity_indices)
            log_records.append(log_info)

            print(
                f"[INFO] 身份 {identity_id} 多模板构建成功，"
                f"有效图片数: {log_info['num_images_success']}, "
                f"模板数: {len(identity_indices)}, "
                f"发现图片数: {log_info['num_images_found']}"
            )

            # 断点续跑关键机制：
            # 每成功处理一个身份，就立即保存一次 gallery。
            # 本版本使用原子写入，避免中断时写坏 JSON。
            save_multi_template_gallery(
                features=gallery_features,
                labels=gallery_labels,
                person_index=gallery_person_index,
                features_path=features_path,
                labels_path=labels_path,
                person_index_path=person_index_path
            )

            save_log(log_records, log_path)

        except KeyboardInterrupt:
            print("\n[WARN] 检测到用户中断 KeyboardInterrupt。")
            print("[WARN] 已成功处理的身份已经通过原子写入保存，可下次使用 resume=true 续跑。")
            save_log(log_records, log_path)
            raise

        except Exception as e:
            num_failed += 1

            log_records.append({
                "identity_id": identity_id,
                "identity_dir": str(identity_dir),
                "num_images_found": len(image_paths),
                "num_images_success": "",
                "num_images_failed": "",
                "num_templates_added": 0,
                "status": "failed",
                "reason": str(e)
            })

            print(f"[ERROR] 身份 {identity_id} 处理异常: {e}")

            save_log(log_records, log_path)
            continue

    if len(gallery_features) > 0:
        save_multi_template_gallery(
            features=gallery_features,
            labels=gallery_labels,
            person_index=gallery_person_index,
            features_path=features_path,
            labels_path=labels_path,
            person_index_path=person_index_path
        )

    save_log(log_records, log_path)

    print("=" * 80)
    print("[INFO] 多模板 gallery 构建完成")
    print(f"[INFO] 新增成功身份数: {num_success_new}")
    print(f"[INFO] 新增模板总数: {num_templates_added_total}")
    print(f"[INFO] 跳过已有身份数: {num_skipped_existing}")
    print(f"[INFO] 空文件夹跳过数: {num_skipped_empty}")
    print(f"[INFO] 图片不足但已使用身份数: {num_incomplete_used}")
    print(f"[INFO] 失败身份数: {num_failed}")
    print(f"[INFO] 当前 gallery 总身份数: {len(gallery_person_index)}")
    print(f"[INFO] 当前 gallery 总模板数: {len(gallery_labels)}")

    if len(gallery_features) > 0:
        final_features = np.stack(gallery_features, axis=0)
        print(f"[INFO] 当前 gallery_features shape: {final_features.shape}")

    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build multi-template face recognition gallery from identities directory."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_gallery(args.config)
