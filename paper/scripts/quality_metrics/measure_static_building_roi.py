"""Measure codec quality on a conservative, static building-only ROI.

The ROI deliberately excludes the moving sky, the tree on the right, and
foreground vegetation. Coordinates follow the usual half-open convention:
x in [0, 1500), y in [590, 850).
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
OUTPUT = ROOT / "results" / "quality_metrics" / "static_building_roi"
REFERENCE = ASSETS / "fig_gt.png"
CANDIDATES = {
    "AV1": ASSETS / "fig_av1_10k.png",
    "VP8": ASSETS / "fig_vp8_10k.png",
    "VP9": ASSETS / "fig_vp9_10k.png",
}

X0, Y0, X1, Y1 = 0, 590, 1500, 850
CANNY_LOW = 50
CANNY_HIGH = 150
BOUNDARY_TOLERANCE_PX = 2.0


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def canny_edges(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH) > 0


def boundary_metrics(reference_edges: np.ndarray, test_edges: np.ndarray) -> dict[str, float]:
    distance_to_reference = distance_transform_edt(~reference_edges)
    distance_to_test = distance_transform_edt(~test_edges)
    test_to_reference = float(distance_to_reference[test_edges].mean())
    reference_to_test = float(distance_to_test[reference_edges].mean())
    precision = float(np.mean(distance_to_reference[test_edges] <= BOUNDARY_TOLERANCE_PX))
    recall = float(np.mean(distance_to_test[reference_edges] <= BOUNDARY_TOLERANCE_PX))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "boundary_displacement_px": 0.5 * (test_to_reference + reference_to_test),
        "boundary_precision_at_2px": precision,
        "boundary_recall_at_2px": recall,
        "boundary_f1_at_2px": f1,
        "boundary_miss_rate_at_2px": 1.0 - recall,
        "false_edge_rate_at_2px": 1.0 - precision,
        "test_edge_density": float(test_edges.mean()),
    }


def measure_vmaf(test_crop: Path, reference_crop: Path, codec: str) -> float:
    log_path = OUTPUT / f"vmaf_{codec.lower()}_building_roi.json"
    relative_log = log_path.relative_to(ROOT).as_posix()
    filter_graph = (
        "[0:v]setpts=PTS-STARTPTS[dist];"
        "[1:v]setpts=PTS-STARTPTS[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={relative_log}:n_threads=4"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(test_crop), "-i", str(reference_crop),
            "-lavfi", filter_graph, "-frames:v", "1", "-f", "null", "NUL",
        ],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(log_path.read_text(encoding="utf-8"))
    return float(data["pooled_metrics"]["vmaf"]["mean"])


def save_roi_location(reference_bgr: np.ndarray) -> None:
    preview = reference_bgr.copy()
    overlay = preview.copy()
    cv2.rectangle(overlay, (X0, Y0), (X1 - 1, Y1 - 1), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.18, preview, 0.82, 0.0, preview)
    cv2.rectangle(preview, (X0, Y0), (X1 - 1, Y1 - 1), (0, 255, 0), 5)
    cv2.putText(
        preview,
        f"Static building ROI: x={X0}:{X1}, y={Y0}:{Y1}",
        (25, Y0 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(OUTPUT / "building_roi_location.png"), preview)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference_bgr = load_bgr(REFERENCE)
    save_roi_location(reference_bgr)

    reference_crop_bgr = reference_bgr[Y0:Y1, X0:X1]
    reference_crop_path = OUTPUT / "building_roi_reference.png"
    cv2.imwrite(str(reference_crop_path), reference_crop_bgr)
    reference_rgb = cv2.cvtColor(reference_crop_bgr, cv2.COLOR_BGR2RGB)
    reference_edges = canny_edges(reference_rgb)

    rows: list[dict[str, float | int | str]] = []
    for codec, path in CANDIDATES.items():
        test_bgr = load_bgr(path)
        if test_bgr.shape != reference_bgr.shape:
            raise ValueError(f"Shape mismatch for {codec}: {test_bgr.shape} vs {reference_bgr.shape}")
        test_crop_bgr = test_bgr[Y0:Y1, X0:X1]
        test_crop_path = OUTPUT / f"building_roi_{codec.lower()}.png"
        cv2.imwrite(str(test_crop_path), test_crop_bgr)
        test_rgb = cv2.cvtColor(test_crop_bgr, cv2.COLOR_BGR2RGB)

        difference = test_rgb.astype(np.float64) - reference_rgb.astype(np.float64)
        row: dict[str, float | int | str] = {
            "codec": codec,
            "roi_x0": X0,
            "roi_y0": Y0,
            "roi_x1_exclusive": X1,
            "roi_y1_exclusive": Y1,
            "roi_width": X1 - X0,
            "roi_height": Y1 - Y0,
            "vmaf_single_frame": measure_vmaf(test_crop_path, reference_crop_path, codec),
            "psnr_rgb_db": float(peak_signal_noise_ratio(reference_rgb, test_rgb, data_range=255)),
            "ssim_rgb": float(structural_similarity(reference_rgb, test_rgb, channel_axis=2, data_range=255)),
            "mae_rgb_0_255": float(np.mean(np.abs(difference))),
            "rmse_rgb_0_255": float(np.sqrt(np.mean(np.square(difference)))),
            **boundary_metrics(reference_edges, canny_edges(test_rgb)),
        }
        rows.append(row)

    with (OUTPUT / "building_roi_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "reference": str(REFERENCE),
        "roi": {
            "x0": X0, "y0": Y0, "x1_exclusive": X1, "y1_exclusive": Y1,
            "description": "Left main building facade; excludes sky, right tree, and foreground vegetation.",
        },
        "limitations": [
            "The four screenshots may come from different timestamps.",
            "Single-frame VMAF has no temporal-motion contribution.",
            "Canny boundary metrics describe all visible architectural edges, not segmentation boundaries.",
        ],
        "reference_edge_density": float(reference_edges.mean()),
        "results": rows,
    }
    (OUTPUT / "building_roi_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for row in rows:
        print(
            f"{row['codec']}: VMAF={row['vmaf_single_frame']:.4f}, "
            f"PSNR={row['psnr_rgb_db']:.4f} dB, SSIM={row['ssim_rgb']:.6f}, "
            f"MAE={row['mae_rgb_0_255']:.4f}, "
            f"BDE={row['boundary_displacement_px']:.4f} px, "
            f"BF1@2px={row['boundary_f1_at_2px']:.6f}"
        )


if __name__ == "__main__":
    main()
