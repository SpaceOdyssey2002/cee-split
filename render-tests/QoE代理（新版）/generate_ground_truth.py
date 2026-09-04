import pandas as pd
import subprocess
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置路径 ---
UNITY_PROJECT_ROOT = r"D:\Unity\QoEProxy"
BASE_DIR = os.path.join(UNITY_PROJECT_ROOT, "DatasetOutputs")
OUTPUT_CSV_NAME = "final_training_dataset_vp9.csv"
VIDEO_CODEC = "libvpx-vp9"
FPS = 60.0
MAX_WORKERS = min(4, max(1, (os.cpu_count() or 4) // 2))

def get_image_dimensions(path):
    """Read the reference dimensions with ffprobe so VMAF inputs can be aligned."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        width, height = result.stdout.strip().split("x")
        return int(width), int(height)
    except (ValueError, TypeError):
        raise RuntimeError(f"Unable to read reference dimensions: {path}\n{result.stderr}")

def get_encoded_payload_bytes(encoded_video):
    """Return coded VP9 packet bytes without counting the WebM container header."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=size", "-of", "csv=p=0", encoded_video
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {encoded_video}:\n{result.stderr}")
    sizes = [int(value) for value in re.findall(r"\d+", result.stdout)]
    if not sizes:
        raise RuntimeError(f"No VP9 packets were found in {encoded_video}")
    return sum(sizes)


def get_vmaf_and_bitrate(rendered_img, gt_img, qp, folder_path):
    """
    计算 VMAF 和码率
    """
    full_rendered_path = os.path.join(UNITY_PROJECT_ROOT, rendered_img)
    full_gt_path = os.path.join(UNITY_PROJECT_ROOT, gt_img)

    temp_file = tempfile.NamedTemporaryFile(
        prefix="vp9_", suffix=".webm", dir=folder_path, delete=False
    )
    encoded_video = temp_file.name
    temp_file.close()

    try:
        # VP9 constant-quality encoding. QP is represented by libvpx-vp9 CRF
        # because the offline proxy needs a reproducible quality control variable.
        encode_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", full_rendered_path,
            "-frames:v", "1", "-c:v", VIDEO_CODEC,
            "-deadline", "good", "-cpu-used", "5", "-row-mt", "1",
            "-threads", "2",
            "-crf", str(qp), "-b:v", "0", "-pix_fmt", "yuv420p",
            encoded_video
        ]
        encoded = subprocess.run(encode_cmd, capture_output=True, text=True)
        if encoded.returncode != 0 or os.path.getsize(encoded_video) == 0:
            raise RuntimeError(
                f"VP9 encoding failed for {full_rendered_path}, QP={qp}:\n{encoded.stderr}"
            )

        payload_bytes = get_encoded_payload_bytes(encoded_video)
        bitrate = payload_bytes * 8.0 * FPS / 1000.0

        reference_width, reference_height = get_image_dimensions(full_gt_path)
        vmaf_filter = (
            f"[0:v]scale={reference_width}:{reference_height}:flags=lanczos,"
            "setsar=1,format=yuv420p[v0];"
            "[1:v]setsar=1,format=yuv420p[v1];"
            "[v0][v1]libvmaf"
        )
        vmaf_cmd = [
            "ffmpeg", "-hide_banner", "-i", encoded_video, "-i", full_gt_path,
            "-lavfi", vmaf_filter, "-frames:v", "1", "-f", "null", "-"
        ]
        result_vmaf = subprocess.run(vmaf_cmd, capture_output=True, text=True)
        if result_vmaf.returncode != 0:
            raise RuntimeError(
                f"VMAF failed for {full_rendered_path}, QP={qp}:\n{result_vmaf.stderr}"
            )

        match = re.search(r"VMAF score:\s*([0-9.]+)", result_vmaf.stderr)
        if match is None:
            raise RuntimeError(
                f"VMAF score was not present for {full_rendered_path}, QP={qp}"
            )
        return float(match.group(1)), bitrate
    finally:
        if os.path.exists(encoded_video):
            os.remove(encoded_video)

def process_all_captures():
    if not os.path.exists(BASE_DIR):
        print(f"❌ 找不到目录: {BASE_DIR}")
        return

    # 获取所有子文件夹
    subdirs = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR)
               if os.path.isdir(os.path.join(BASE_DIR, d))]

    if not subdirs:
        print(f"❌ {BASE_DIR} 中没有找到任何采集数据文件夹")
        return

    print(f"🔍 找到 {len(subdirs)} 个文件夹，准备开始处理...")

    for current_dir in subdirs:
        folder_name = os.path.basename(current_dir)
        csv_path = os.path.join(current_dir, "metadata.csv")
        output_file = os.path.join(current_dir, OUTPUT_CSV_NAME)
        partial_file = output_file + ".partial.csv"

        # 检查是否存在 metadata.csv
        if not os.path.exists(csv_path):
            print(f"⏩ 跳过 {folder_name}: 找不到 metadata.csv")
            continue

        # 可选：如果已经处理过，是否跳过？
        if os.path.exists(output_file):
            print(f"⏩ 跳过 {folder_name}: 已经存在 final_training_dataset.csv")
            continue

        print(f"\n📂 正在处理文件夹: {folder_name}", flush=True)

        metadata = pd.read_csv(csv_path)
        if os.path.exists(partial_file):
            df = pd.read_csv(partial_file)
            if len(df) != len(metadata):
                raise RuntimeError(f"断点文件与 metadata 行数不一致: {partial_file}")
            print(f"   恢复断点: {df['vmaf'].notna().sum()} / {len(df)}", flush=True)
        else:
            df = metadata.copy()
            df['vmaf'] = float('nan')
            df['bitrate'] = float('nan')
            df['codec'] = 'VP9'
            df['encoder'] = VIDEO_CODEC
            df['fps'] = FPS

        pending_indices = [
            int(index) for index in df.index
            if pd.isna(df.at[index, 'vmaf']) or pd.isna(df.at[index, 'bitrate'])
        ]

        def process_row(index):
            row = df.loc[index]
            vmaf, bitrate = get_vmaf_and_bitrate(
                row['rendered_image_path'], row['gt_image_path'],
                int(row['qp']), current_dir
            )
            return index, vmaf, bitrate

        completed = len(df) - len(pending_indices)
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_row, index) for index in pending_indices]
                for future in as_completed(futures):
                    index, vmaf, bitrate = future.result()
                    df.at[index, 'vmaf'] = vmaf
                    df.at[index, 'bitrate'] = bitrate
                    completed += 1
                    if completed % 10 == 0 or completed == len(df):
                        df.to_csv(partial_file, index=False)
                        print(
                            f"   进度: [{completed}/{len(df)}] - "
                            f"当前 VMAF: {vmaf:.2f} - workers={MAX_WORKERS}",
                            flush=True
                        )
        except Exception:
            df.to_csv(partial_file, index=False)
            raise

        df.to_csv(output_file, index=False)
        if os.path.exists(partial_file):
            os.remove(partial_file)
        print(f"✅ 处理完成: {folder_name}")

    print("\n" + "=" * 30)
    print("🎉 所有文件夹处理任务结束！")
    print("=" * 30)

if __name__ == "__main__":
    process_all_captures()
