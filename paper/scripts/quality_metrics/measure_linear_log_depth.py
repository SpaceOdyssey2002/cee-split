"""Compare linear and logarithmic depth composition against local GT.

All phone status bars are removed. Each composed screenshot is registered to
the local viewport independently, then evaluated on the same static building
ROI that excludes sky, foliage, and the Development Console.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

import measure_local_cloud_quality as metrics


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = Path(r"C:\Users\13577\Desktop\local.jpg")
TESTS = {
    "linear_depth": Path(r"C:\Users\13577\Desktop\线性.jpg"),
    "log_depth": Path(r"C:\Users\13577\Desktop\对数.jpg"),
}
OUTPUT = ROOT / "results" / "quality_metrics" / "linear_vs_log_depth"


def load_bgr_unicode(path: Path) -> np.ndarray:
    """Read image paths containing non-ASCII characters on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def save_difference_map(reference: np.ndarray, test: np.ndarray, label: str) -> None:
    difference = cv2.absdiff(reference, test)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(np.clip(gray * 5, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(OUTPUT / f"{label}_difference_x5.png"), heat)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.OUTPUT = OUTPUT

    local_full = metrics.load_bgr(LOCAL_PATH)
    local = local_full[metrics.STATUS_BAR_HEIGHT:, :]
    x0, y0, x1, y1 = metrics.ROI_X0, metrics.ROI_Y0, metrics.ROI_X1, metrics.ROI_Y1
    local_roi = local[y0:y1, x0:x1]

    rows: list[dict[str, float | int | str]] = []
    registrations: dict[str, dict[str, float | int]] = {}
    cv2.imwrite(str(OUTPUT / "local_gt_roi.png"), local_roi)

    for label, path in TESTS.items():
        test_full = load_bgr_unicode(path)
        test = test_full[metrics.STATUS_BAR_HEIGHT:, :]
        if test.shape != local.shape:
            raise ValueError(f"Viewport shape mismatch for {label}: {test.shape} vs {local.shape}")

        affine, registration = metrics.estimate_affine(local, test)
        registrations[label] = registration
        aligned = cv2.warpAffine(
            test,
            affine,
            (local.shape[1], local.shape[0]),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned_roi = aligned[y0:y1, x0:x1]
        cv2.imwrite(str(OUTPUT / f"{label}_viewport_aligned.png"), aligned)
        save_difference_map(local_roi, aligned_roi, label)

        row = metrics.measure_pair(local_roi, aligned_roi, f"{label}_vs_local_gt")
        rgb_difference = cv2.absdiff(local_roi, aligned_roi)
        pixel_mae = rgb_difference.astype(np.float32).mean(axis=2)
        row["pixel_fraction_mae_gt_10"] = float(np.mean(pixel_mae > 10.0))
        row["pixel_fraction_mae_gt_20"] = float(np.mean(pixel_mae > 20.0))
        rows.append(row)

    with (OUTPUT / "linear_log_depth_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    linear = next(row for row in rows if row["scope"].startswith("linear_depth"))
    logarithmic = next(row for row in rows if row["scope"].startswith("log_depth"))
    numeric_keys = [
        "vmaf_single_frame", "psnr_rgb_db", "ssim_rgb", "mae_rgb_0_255",
        "rmse_rgb_0_255", "boundary_f1_at_2px", "boundary_median_px",
        "boundary_clipped_mean_at_10px", "pixel_fraction_mae_gt_10",
        "pixel_fraction_mae_gt_20",
    ]
    delta_log_minus_linear = {
        key: float(logarithmic[key]) - float(linear[key]) for key in numeric_keys
    }

    report = {
        "reference": str(LOCAL_PATH),
        "tests": {name: str(path) for name, path in TESTS.items()},
        "status_bar_pixels_removed": metrics.STATUS_BAR_HEIGHT,
        "static_building_roi": {
            "x0": x0, "y0": y0, "x1_exclusive": x1, "y1_exclusive": y1,
            "description": "Static left building; excludes sky, foliage, and Development Console.",
        },
        "registrations_test_to_local": registrations,
        "delta_log_minus_linear": delta_log_minus_linear,
        "limitations": [
            "The screenshots were captured at different timestamps.",
            "Evaluation is restricted to a static architectural ROI.",
            "Affine registration introduces resampling.",
            "Single-frame VMAF has no temporal-motion contribution.",
        ],
        "results": rows,
    }
    (OUTPUT / "linear_log_depth_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Registrations:")
    print(json.dumps(registrations, ensure_ascii=False, indent=2))
    for row in rows:
        print(
            f"{row['scope']}: VMAF={row['vmaf_single_frame']:.4f}, "
            f"PSNR={row['psnr_rgb_db']:.4f} dB, SSIM={row['ssim_rgb']:.6f}, "
            f"MAE={row['mae_rgb_0_255']:.4f}, BF1@2px={row['boundary_f1_at_2px']:.6f}, "
            f"median boundary={row['boundary_median_px']:.4f} px"
        )
    print("Delta (log - linear):")
    print(json.dumps(delta_log_minus_linear, indent=2))


if __name__ == "__main__":
    main()
