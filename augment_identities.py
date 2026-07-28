import os
import cv2
import csv
import yaml
import random
import numpy as np
from PIL import Image, ImageEnhance


CONFIG_PATH = "config.yaml"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def join_project_path(project_root, relative_path):
    return os.path.join(project_root, relative_path).replace("\\", "/")


def read_image_chinese_path(image_path):
    """
    支持中文路径的 OpenCV 图像读取。
    """
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def save_image_chinese_path(image_path, image, quality=95):
    """
    支持中文路径的 OpenCV 图像保存。
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        params = []

    success, encoded = cv2.imencode(ext, image, params)

    if not success:
        return False

    encoded.tofile(image_path)
    return True


def cv2_to_pil(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def pil_to_cv2(image):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image_bgr


def horizontal_flip(image):
    return cv2.flip(image, 1)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return rotated


def adjust_brightness(image, factor):
    pil_image = cv2_to_pil(image)
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced = enhancer.enhance(factor)
    return pil_to_cv2(enhanced)


def adjust_contrast(image, factor):
    pil_image = cv2_to_pil(image)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(factor)
    return pil_to_cv2(enhanced)


def gaussian_blur(image, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_gaussian_noise(image, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def jpeg_compression(image, quality_min=45, quality_max=65):
    quality = random.randint(int(quality_min), int(quality_max))

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", image, encode_param)

    if not success:
        return image

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if decoded is None:
        return image

    return decoded


def get_random_occlusion_color(augment_cfg):
    colors = augment_cfg.get("occlusion_colors", [[0, 0, 0]])
    color = random.choice(colors)
    return tuple(int(v) for v in color)


def rectangle_occlusion(image, region_cfg, augment_cfg):
    h, w = image.shape[:2]
    result = image.copy()

    x1 = int(w * float(region_cfg["x1"]))
    x2 = int(w * float(region_cfg["x2"]))
    y1 = int(h * float(region_cfg["y1"]))
    y2 = int(h * float(region_cfg["y2"]))

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    color = get_random_occlusion_color(augment_cfg)

    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=-1)

    return result


def upper_face_occlusion(image, augment_cfg):
    region_cfg = augment_cfg["upper_occlusion"]
    return rectangle_occlusion(image, region_cfg, augment_cfg)


def lower_face_occlusion(image, augment_cfg):
    region_cfg = augment_cfg["lower_occlusion"]
    return rectangle_occlusion(image, region_cfg, augment_cfg)


def random_erasing(image, augment_cfg):
    h, w = image.shape[:2]
    result = image.copy()

    erase_cfg = augment_cfg["random_erasing"]

    min_w = int(w * float(erase_cfg["min_width_ratio"]))
    max_w = int(w * float(erase_cfg["max_width_ratio"]))
    min_h = int(h * float(erase_cfg["min_height_ratio"]))
    max_h = int(h * float(erase_cfg["max_height_ratio"]))

    min_w = max(1, min_w)
    max_w = max(min_w, max_w)
    min_h = max(1, min_h)
    max_h = max(min_h, max_h)

    erase_w = random.randint(min_w, max_w)
    erase_h = random.randint(min_h, max_h)

    x1 = random.randint(0, max(0, w - erase_w))
    y1 = random.randint(0, max(0, h - erase_h))
    x2 = x1 + erase_w
    y2 = y1 + erase_h

    color = get_random_occlusion_color(augment_cfg)

    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=-1)

    return result


def shift_and_scale(image, augment_cfg):
    h, w = image.shape[:2]

    shift_cfg = augment_cfg["shift_scale"]
    scale = float(shift_cfg.get("scale", 1.05))

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    max_x = max(0, new_w - w)
    max_y = max(0, new_h - h)

    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)

    cropped = resized[start_y:start_y + h, start_x:start_x + w]

    if cropped.shape[:2] != (h, w):
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return cropped


def generate_augmented_images(image, augment_cfg):
    """
    固定编号增强策略：
    000：原图，不在这里生成
    001：水平翻转
    002：左旋转
    003：右旋转
    004：偏暗
    005：偏亮
    006：对比度降低
    007：对比度增强
    008：轻微模糊
    009：轻微噪声
    010：JPEG压缩
    011：上半脸遮挡
    012：下半脸遮挡
    013：随机小块遮挡
    014：轻微平移/裁剪
    """
    augmented = {}

    augmented["001"] = horizontal_flip(image)

    augmented["002"] = rotate_image(
        image,
        float(augment_cfg.get("rotate_left_angle", -8))
    )

    augmented["003"] = rotate_image(
        image,
        float(augment_cfg.get("rotate_right_angle", 8))
    )

    augmented["004"] = adjust_brightness(
        image,
        float(augment_cfg.get("brightness_dark_factor", 0.8))
    )

    augmented["005"] = adjust_brightness(
        image,
        float(augment_cfg.get("brightness_bright_factor", 1.2))
    )

    augmented["006"] = adjust_contrast(
        image,
        float(augment_cfg.get("contrast_low_factor", 0.8))
    )

    augmented["007"] = adjust_contrast(
        image,
        float(augment_cfg.get("contrast_high_factor", 1.2))
    )

    augmented["008"] = gaussian_blur(
        image,
        int(augment_cfg.get("gaussian_blur_kernel", 3))
    )

    augmented["009"] = add_gaussian_noise(
        image,
        mean=float(augment_cfg.get("gaussian_noise_mean", 0)),
        sigma=float(augment_cfg.get("gaussian_noise_sigma", 10))
    )

    augmented["010"] = jpeg_compression(
        image,
        quality_min=int(augment_cfg.get("jpeg_quality_min", 45)),
        quality_max=int(augment_cfg.get("jpeg_quality_max", 65))
    )

    augmented["011"] = upper_face_occlusion(image, augment_cfg)
    augmented["012"] = lower_face_occlusion(image, augment_cfg)
    augmented["013"] = random_erasing(image, augment_cfg)
    augmented["014"] = shift_and_scale(image, augment_cfg)

    return augmented


def is_source_image(filename, source_index, image_exts):
    name, ext = os.path.splitext(filename)

    if ext.lower() not in image_exts:
        return False

    parts = name.split("_")

    if len(parts) < 2:
        return False

    return parts[-1] == source_index


def get_prefix_without_index(filename):
    """
    例如：
    101024_王金霞_000.jpg -> 101024_王金霞
    """
    name, _ = os.path.splitext(filename)
    parts = name.split("_")

    if len(parts) >= 2:
        return "_".join(parts[:-1])

    return name


def init_csv(csv_path, header):
    ensure_dir(os.path.dirname(csv_path))

    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_csv(csv_path, row):
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def augment_identities(config):
    project_root = config["project"]["root"]
    paths_cfg = config["paths"]
    preprocess_cfg = config["preprocess"]
    augment_cfg = config["augment"]
    output_cfg = config["output"]

    if not augment_cfg.get("enable", True):
        print("数据增强未启用：augment.enable = false")
        return

    identities_dir = join_project_path(project_root, augment_cfg.get("input_dir", paths_cfg["output_dir"]))
    log_dir = join_project_path(project_root, paths_cfg["log_dir"])

    success_csv = os.path.join(log_dir, output_cfg["augment_success_csv"])
    skipped_csv = os.path.join(log_dir, output_cfg["augment_skipped_csv"])
    failed_csv = os.path.join(log_dir, output_cfg["augment_failed_csv"])

    init_csv(
        success_csv,
        ["person_folder", "source_image", "aug_index", "aug_type", "save_path"]
    )
    init_csv(
        skipped_csv,
        ["person_folder", "source_image", "aug_index", "reason", "save_path"]
    )
    init_csv(
        failed_csv,
        ["person_folder", "source_image", "reason"]
    )

    image_exts = [ext.lower() for ext in preprocess_cfg.get("image_exts", [".jpg", ".jpeg", ".png", ".bmp"])]

    source_index = str(augment_cfg.get("source_index", "000"))
    output_indices = [str(idx) for idx in augment_cfg.get("output_indices", [])]

    skip_if_exists = bool(augment_cfg.get("skip_if_exists", True))
    overwrite = bool(augment_cfg.get("overwrite", False))
    save_ext = augment_cfg.get("save_ext", ".jpg")
    save_quality = int(augment_cfg.get("save_quality", 95))

    aug_name_map = {
        "001": "horizontal_flip",
        "002": "rotate_left",
        "003": "rotate_right",
        "004": "brightness_dark",
        "005": "brightness_bright",
        "006": "contrast_low",
        "007": "contrast_high",
        "008": "gaussian_blur",
        "009": "gaussian_noise",
        "010": "jpeg_compression",
        "011": "upper_face_occlusion",
        "012": "lower_face_occlusion",
        "013": "random_erasing",
        "014": "shift_and_scale"
    }

    if not os.path.exists(identities_dir):
        raise FileNotFoundError(f"identities 目录不存在：{identities_dir}")

    print(f"开始数据增强，输入目录：{identities_dir}")
    print("增强策略：000 保留原图，001-014 生成固定增强图")
    print(f"skip_if_exists={skip_if_exists}, overwrite={overwrite}")

    total_success = 0
    total_skipped = 0
    total_failed = 0

    for person_folder in os.listdir(identities_dir):
        person_dir = os.path.join(identities_dir, person_folder)

        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if not is_source_image(filename, source_index, image_exts):
                continue

            source_path = os.path.join(person_dir, filename)
            image = read_image_chinese_path(source_path)

            if image is None:
                reason = "image_read_failed"
                append_csv(failed_csv, [person_folder, filename, reason])
                print(f"[失败] 读取失败：{source_path}")
                total_failed += 1
                continue

            prefix = get_prefix_without_index(filename)

            try:
                augmented_images = generate_augmented_images(image, augment_cfg)
            except Exception as e:
                reason = f"augment_generate_failed: {str(e)}"
                append_csv(failed_csv, [person_folder, filename, reason])
                print(f"[失败] 增强生成失败：{source_path}，原因：{e}")
                total_failed += 1
                continue

            for aug_index in output_indices:
                if aug_index not in augmented_images:
                    reason = "aug_index_not_defined"
                    save_path = os.path.join(person_dir, f"{prefix}_{aug_index}{save_ext}")
                    append_csv(skipped_csv, [person_folder, filename, aug_index, reason, save_path])
                    total_skipped += 1
                    continue

                save_name = f"{prefix}_{aug_index}{save_ext}"
                save_path = os.path.join(person_dir, save_name)

                if os.path.exists(save_path) and skip_if_exists and not overwrite:
                    reason = "target_exists"
                    append_csv(skipped_csv, [person_folder, filename, aug_index, reason, save_path])
                    print(f"[跳过] 已存在：{save_path}")
                    total_skipped += 1
                    continue

                success = save_image_chinese_path(
                    save_path,
                    augmented_images[aug_index],
                    quality=save_quality
                )

                if success:
                    aug_type = aug_name_map.get(aug_index, "unknown")
                    append_csv(success_csv, [person_folder, filename, aug_index, aug_type, save_path])
                    print(f"[成功] {filename} -> {save_name}")
                    total_success += 1
                else:
                    reason = "image_save_failed"
                    append_csv(failed_csv, [person_folder, filename, reason])
                    print(f"[失败] 保存失败：{save_path}")
                    total_failed += 1

    print("=" * 60)
    print("数据增强完成")
    print(f"成功生成：{total_success}")
    print(f"跳过数量：{total_skipped}")
    print(f"失败数量：{total_failed}")
    print(f"成功日志：{success_csv}")
    print(f"跳过日志：{skipped_csv}")
    print(f"失败日志：{failed_csv}")
    print("=" * 60)


def main():
    config = load_config(CONFIG_PATH)
    augment_identities(config)


if __name__ == "__main__":
    main()