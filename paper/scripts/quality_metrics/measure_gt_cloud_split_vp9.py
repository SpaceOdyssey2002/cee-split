"""Controlled static-ROI comparison of split VP9 and pure-cloud VP9.

Both strategies are evaluated against assets/figures/quality/fig_gt.png over the same static
architectural ROI. The phone status bar is removed from the pure-cloud image,
and its display-induced horizontal stretch is corrected before measurement.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

import measure_local_cloud_quality as metrics


ROOT = Path(__file__).resolve().parents[1]
GT_PATH = ROOT / "assets" / "figures" / "quality" / "fig_gt.png"
SPLIT_VP9_PATH = ROOT / "assets" / "figures" / "quality" / "fig_vp9_10k.png"
CLOUD_VP9_PATH = Path(r"C:\Users\13577\Desktop\cloud.jpg")
OUTPUT = ROOT / "results" / "quality_metrics" / "gt_cloud_split_vp9"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.OUTPUT = OUTPUT

    gt = metrics.load_bgr(GT_PATH)
    split = metrics.load_bgr(SPLIT_VP9_PATH)
    cloud_full = metrics.load_bgr(CLOUD_VP9_PATH)
    cloud = cloud_full[metrics.STATUS_BAR_HEIGHT:, :]
    if gt.shape != split.shape or gt.shape != cloud.shape:
        raise ValueError(f"Shape mismatch: GT={gt.shape}, split={split.shape}, cloud={cloud.shape}")

    cloud_affine, cloud_registration = metrics.estimate_affine(gt, cloud)
    cloud_aligned = cv2.warpAffine(
        cloud,
        cloud_affine,
        (gt.shape[1], gt.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    cv2.imwrite(str(OUTPUT / "cloud_vp9_aligned_to_gt.png"), cloud_aligned)

    x0, y0, x1, y1 = metrics.ROI_X0, metrics.ROI_Y0, metrics.ROI_X1, metrics.ROI_Y1
    gt_roi = gt[y0:y1, x0:x1]
    split_roi = split[y0:y1, x0:x1]
    cloud_roi = cloud_aligned[y0:y1, x0:x1]

    split_row = metrics.measure_pair(gt_roi, split_roi, "split_vp9_vs_gt")
    cloud_row = metrics.measure_pair(gt_roi, cloud_roi, "cloud_vp9_vs_gt")
    rows = [cloud_row, split_row]

    keys_for_delta = [
        "vmaf_single_frame",
        "psnr_rgb_db",
        "ssim_rgb",
        "mae_rgb_0_255",
        "rmse_rgb_0_255",
        "boundary_f1_at_2px",
        "boundary_median_px",
        "boundary_clipped_mean_at_10px",
    ]
    delta = {
        key: float(split_row[key]) - float(cloud_row[key])
        for key in keys_for_delta
    }

    with (OUTPUT / "gt_cloud_split_vp9_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "reference": str(GT_PATH),
        "strategies": {
            "pure_cloud_vp9": str(CLOUD_VP9_PATH),
            "split_vp9": str(SPLIT_VP9_PATH),
        },
        "static_building_roi": {
            "x0": x0,
            "y0": y0,
            "x1_exclusive": x1,
            "y1_exclusive": y1,
            "description": "Static architectural region excluding sky, foliage, statistics overlays, and console overlays.",
        },
        "cloud_preprocessing": {
            "phone_status_bar_pixels_removed": metrics.STATUS_BAR_HEIGHT,
            "affine_registration_to_gt": cloud_registration,
        },
        "delta_split_minus_cloud": delta,
        "limitations": [
            "The screenshots were not captured from a synchronized frame identifier.",
            "The comparison is restricted to a static architectural ROI.",
            "The pure-cloud JPEG and split-rendered PNG use different capture containers.",
            "Affine correction of the cloud screenshot introduces resampling.",
            "Single-frame VMAF has no temporal-motion contribution.",
        ],
        "results": rows,
    }
    (OUTPUT / "gt_cloud_split_vp9_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Cloud-to-GT affine registration:")
    print(json.dumps(cloud_registration, indent=2))
    for row in rows:
        print(
            f"{row['scope']}: VMAF={row['vmaf_single_frame']:.4f}, "
            f"PSNR={row['psnr_rgb_db']:.4f} dB, SSIM={row['ssim_rgb']:.6f}, "
            f"MAE={row['mae_rgb_0_255']:.4f}, BF1@2px={row['boundary_f1_at_2px']:.6f}, "
            f"median boundary={row['boundary_median_px']:.4f} px"
        )
    print("Delta (split - cloud):")
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()
