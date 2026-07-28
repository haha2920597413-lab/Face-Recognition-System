import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_step(step_name: str, command: list, cwd: Path, stop_on_error: bool = True) -> bool:
    """
    执行单个流水线步骤。
    """
    print("\n" + "=" * 80)
    print(f"[PIPELINE] 开始执行：{step_name}")
    print(f"[PIPELINE] 命令：{' '.join(command)}")
    print(f"[PIPELINE] 工作目录：{cwd}")
    print("=" * 80)

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            text=True,
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print("\n" + "-" * 80)
            print(f"[PIPELINE] 步骤成功：{step_name}")
            print(f"[PIPELINE] 耗时：{elapsed:.2f} 秒")
            print("-" * 80)
            return True

        print("\n" + "!" * 80)
        print(f"[PIPELINE] 步骤失败：{step_name}")
        print(f"[PIPELINE] 返回码：{result.returncode}")
        print(f"[PIPELINE] 耗时：{elapsed:.2f} 秒")
        print("!" * 80)

        if stop_on_error:
            print("[PIPELINE] 已停止后续步骤。")
            return False

        return True

    except Exception as e:
        elapsed = time.time() - start_time

        print("\n" + "!" * 80)
        print(f"[PIPELINE] 步骤异常：{step_name}")
        print(f"[PIPELINE] 错误信息：{e}")
        print(f"[PIPELINE] 耗时：{elapsed:.2f} 秒")
        print("!" * 80)

        if stop_on_error:
            print("[PIPELINE] 已停止后续步骤。")
            return False

        return True


def check_required_files(project_root: Path, script_names: list) -> bool:
    """
    检查流水线依赖脚本是否存在。
    """
    ok = True

    print("\n[PIPELINE] 检查依赖脚本...")

    for script_name in script_names:
        script_path = project_root / script_name

        if script_path.exists():
            print(f"[OK] {script_name}")
        else:
            print(f"[MISSING] {script_name} 不存在：{script_path}")
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for updating face recognition gallery."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="项目根目录。默认使用当前脚本所在目录。"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径，默认 config.yaml。"
    )

    parser.add_argument(
        "--skip-crop",
        action="store_true",
        help="跳过人脸裁剪步骤。"
    )

    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="跳过数据增强步骤。"
    )

    parser.add_argument(
        "--skip-gallery",
        action="store_true",
        help="跳过特征库构建步骤。"
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某一步失败后继续执行后续步骤。默认失败即停止。"
    )

    args = parser.parse_args()

    if args.project_root is None:
        project_root = Path(__file__).resolve().parent
    else:
        project_root = Path(args.project_root).resolve()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = project_root / config_path

    print("\n" + "=" * 80)
    print("[PIPELINE] 人脸识别特征库增量更新流水线")
    print(f"[PIPELINE] 启动时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[PIPELINE] 项目根目录：{project_root}")
    print(f"[PIPELINE] 配置文件：{config_path}")
    print(f"[PIPELINE] Python解释器：{sys.executable}")
    print("=" * 80)

    if not project_root.exists():
        raise FileNotFoundError(f"项目根目录不存在：{project_root}")

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    required_scripts = []

    if not args.skip_crop:
        required_scripts.append("face_crop_align.py")

    if not args.skip_augment:
        required_scripts.append("augment_identities.py")

    if not args.skip_gallery:
        required_scripts.append("build_gallery.py")

    if not check_required_files(project_root, required_scripts):
        print("\n[PIPELINE] 依赖脚本缺失，流水线终止。")
        sys.exit(1)

    stop_on_error = not args.continue_on_error

    steps = []

    if not args.skip_crop:
        steps.append(
            (
                "1. 人脸裁剪 + 关键点对齐",
                [
                    sys.executable,
                    "face_crop_align.py",
                    "--config",
                    str(config_path),
                ],
            )
        )

    if not args.skip_augment:
        steps.append(
            (
                "2. 数据增强",
                [
                    sys.executable,
                    "augment_identities.py",
                    "--config",
                    str(config_path),
                ],
            )
        )

    if not args.skip_gallery:
        steps.append(
            (
                "3. 多模板特征库构建 / 增量更新",
                [
                    sys.executable,
                    "build_gallery.py",
                    "--config",
                    str(config_path),
                ],
            )
        )

    pipeline_start = time.time()

    success_steps = 0
    failed_steps = 0

    for step_name, command in steps:
        ok = run_step(
            step_name=step_name,
            command=command,
            cwd=project_root,
            stop_on_error=stop_on_error,
        )

        if ok:
            success_steps += 1
        else:
            failed_steps += 1
            break

    total_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 80)
    print("[PIPELINE] 流水线执行结束")
    print(f"[PIPELINE] 成功步骤数：{success_steps}")
    print(f"[PIPELINE] 失败步骤数：{failed_steps}")
    print(f"[PIPELINE] 总耗时：{total_elapsed:.2f} 秒")
    print(f"[PIPELINE] 结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if failed_steps > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()