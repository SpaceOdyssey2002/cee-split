"""Measure static codec quality and image-edge boundary fidelity.

The script treats fig_gt.png as the reference and evaluates the AV1, VP8,
and VP9 reconstructions at their native 2664 x 1080 resolution.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "figures" / "quality"
OUTPUT = ROOT / "results" / "quality_metrics" / "codec_comparison"
REFERENCE = ASSETS / "fig_gt.png"
CANDIDATES = {
    "AV1": ASSETS / "fig_av1_10k.png",
    "VP8": ASSETS / "fig_vp8_10k.png",
    "VP9": ASSETS / "fig_vp9_10k.png",
}

CANNY_LOW = 50
CANNY_HIGH = 150
BOUNDARY_TOLERANCE_PX = 2.0


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def canny_edges(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH) > 0


def boundary_metrics(reference_edges: np.ndarray, test_edges: np.ndarray) -> dict[str, float]:
    if not reference_edges.any() or not test_edges.any():
        raise ValueError("Canny produced an empty boundary map")

    distance_to_reference = distance_transform_edt(~reference_edges)
    distance_to_test = distance_transform_edt(~test_edges)

    test_to_reference = float(distance_to_reference[test_edges].mean())
    reference_to_test = float(distance_to_test[reference_edges].mean())
    symmetric_bde = 0.5 * (test_to_reference + reference_to_test)

    precision = float(
        np.mean(distance_to_reference[test_edges] <= BOUNDARY_TOLERANCE_PX)
    )
    recall = float(
        np.mean(distance_to_test[reference_edges] <= BOUNDARY_TOLERANCE_PX)
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "boundary_displacement_px": symmetric_bde,
        "test_to_reference_px": test_to_reference,
        "reference_to_test_px": reference_to_test,
        "boundary_precision_at_2px": precision,
        "boundary_recall_at_2px": recall,
        "boundary_f1_at_2px": f1,
        "boundary_miss_rate_at_2px": 1.0 - recall,
        "false_edge_rate_at_2px": 1.0 - precision,
        "test_edge_density": float(test_edges.mean()),
    }


def ffmpeg_log_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def measure_vmaf(test_path: Path, name: str) -> float:
    log_path = OUTPUT / f"vmaf_{name.lower()}.json"
    filter_graph = (
        "[0:v]setpts=PTS-STARTPTS[dist];"
        "[1:v]setpts=PTS-STARTPTS[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={ffmpeg_log_path(log_path)}:n_threads=4"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(test_path),
        "-i",
        str(REFERENCE),
        "-lavfi",
        filter_graph,
        "-frames:v",
        "1",
        "-f",
        "null",
        "NUL",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    data = json.loads(log_path.read_text(encoding="utf-8"))
    return float(data["pooled_metrics"]["vmaf"]["mean"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = load_rgb(REFERENCE)
    reference_edges = canny_edges(reference)

    rows: list[dict[str, float | int | str]] = []
    for codec, path in CANDIDATES.items():
        test = load_rgb(path)
        if test.shape != reference.shape:
            raise ValueError(f"Shape mismatch for {codec}: {test.shape} vs {reference.shape}")

        difference = test.astype(np.float64) - reference.astype(np.float64)
        mae = float(np.mean(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        psnr = float(peak_signal_noise_ratio(reference, test, data_range=255))
        ssim = float(
            structural_similarity(reference, test, channel_axis=2, data_range=255)
        )
        edge_values = boundary_metrics(reference_edges, canny_edges(test))

        rows.append(
            {
                "codec": codec,
                "width": int(reference.shape[1]),
                "height": int(reference.shape[0]),
                "vmaf_single_frame": measure_vmaf(path, codec),
                "psnr_rgb_db": psnr,
                "ssim_rgb": ssim,
                "mae_rgb_0_255": mae,
                "rmse_rgb_0_255": rmse,
                **edge_values,
            }
        )

    csv_path = OUTPUT / "codec_quality_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "reference": str(REFERENCE),
        "method": {
            "vmaf": "libvmaf default model applied to one static frame; motion feature is zero",
            "psnr_ssim_mae": "full-resolution RGB comparison",
            "boundary_detector": (
                f"grayscale Gaussian blur (5x5, sigma=1.0) plus Canny "
                f"({CANNY_LOW}, {CANNY_HIGH})"
            ),
            "boundary_displacement": (
                "mean of test-to-reference and reference-to-test Euclidean edge distances"
            ),
            "boundary_f1": (
                f"edge precision/recall with {BOUNDARY_TOLERANCE_PX:g}-pixel tolerance"
            ),
        },
        "reference_edge_density": float(reference_edges.mean()),
        "results": rows,
    }
    (OUTPUT / "codec_quality_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for row in rows:
        print(
            f"{row['codec']}: VMAF={row['vmaf_single_frame']:.4f}, "
            f"PSNR={row['psnr_rgb_db']:.4f} dB, SSIM={row['ssim_rgb']:.6f}, "
            f"BDE={row['boundary_displacement_px']:.4f} px, "
            f"BF1@2px={row['boundary_f1_at_2px']:.6f}, "
            f"miss@2px={100.0 * row['boundary_miss_rate_at_2px']:.3f}%"
        )


if __name__ == "__main__":
    main()
