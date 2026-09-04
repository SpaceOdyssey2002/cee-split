"""Compare the phone-captured pure-cloud frame against fig_vp9_10k.png.

fig_vp9_10k.png is treated as the reference. The 120-pixel phone status bar
is removed from cloud.jpg before comparison. The primary result uses a static
building ROI after affine registration.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

import measure_local_cloud_quality as metrics


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "assets" / "figures" / "quality" / "fig_vp9_10k.png"
CLOUD_PHONE_PATH = Path(r"C:\Users\13577\Desktop\cloud.jpg")
OUTPUT = ROOT / "results" / "quality_metrics" / "cloud_vs_vp9"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Reuse the tested metric functions while directing their VMAF logs and
    # image products to this comparison's own output directory.
    metrics.OUTPUT = OUTPUT

    reference = metrics.load_bgr(REFERENCE_PATH)
    cloud_full = metrics.load_bgr(CLOUD_PHONE_PATH)
    cloud = cloud_full[metrics.STATUS_BAR_HEIGHT:, :]
    if reference.shape != cloud.shape:
        raise ValueError(f"Viewport shapes differ: {reference.shape} vs {cloud.shape}")

    cv2.imwrite(str(OUTPUT / "vp9_reference.png"), reference)
    cv2.imwrite(str(OUTPUT / "cloud_no_status_bar.png"), cloud)

    affine, registration = metrics.estimate_affine(reference, cloud)
    aligned_cloud = cv2.warpAffine(
        cloud,
        affine,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    cv2.imwrite(str(OUTPUT / "cloud_affine_aligned_to_vp9.png"), aligned_cloud)

    x0, y0, x1, y1 = metrics.ROI_X0, metrics.ROI_Y0, metrics.ROI_X1, metrics.ROI_Y1
    reference_roi = reference[y0:y1, x0:x1]
    cloud_roi = cloud[y0:y1, x0:x1]
    aligned_roi = aligned_cloud[y0:y1, x0:x1]

    rows = [
        metrics.measure_pair(reference, cloud, "viewport_raw"),
        metrics.measure_pair(reference_roi, cloud_roi, "building_raw"),
        metrics.measure_pair(reference_roi, aligned_roi, "building_affine_aligned"),
    ]
    reverse_vmaf = metrics.measure_vmaf(
        OUTPUT / "building_affine_aligned_reference.png",
        OUTPUT / "building_affine_aligned_cloud.png",
        "building_affine_aligned_reverse",
    )
    for row in rows:
        row["vmaf_reverse_cloud_as_reference"] = (
            reverse_vmaf if row["scope"] == "building_affine_aligned" else ""
        )
    with (OUTPUT / "cloud_vs_vp9_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "reference": str(REFERENCE_PATH),
        "test": str(CLOUD_PHONE_PATH),
        "status_bar_crop": {"top_pixels_removed_from_cloud": metrics.STATUS_BAR_HEIGHT},
        "static_building_roi": {
            "x0": x0,
            "y0": y0,
            "x1_exclusive": x1,
            "y1_exclusive": y1,
            "description": "Static left building; excludes sky, right tree, foreground vegetation, and VP9 statistics panel.",
        },
        "registration_cloud_to_vp9": registration,
        "vmaf_direction_note": (
            "Primary VMAF treats VP9 as reference and cloud as distorted. "
            "The reverse value treats cloud as reference and VP9 as distorted; "
            "the asymmetry demonstrates why two distorted outputs should not be used as mutual VMAF references."
        ),
        "limitations": [
            "The images were captured at different timestamps.",
            "The raw viewport contains moving clouds and foliage.",
            "The raw viewport contains a VP9-only statistics panel.",
            "The cloud source is JPEG while the VP9 reference is PNG.",
            "Affine alignment introduces an additional resampling operation.",
            "VMAF is evaluated on one static frame, so its temporal feature is zero.",
        ],
        "results": rows,
    }
    (OUTPUT / "cloud_vs_vp9_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Affine registration:")
    print(json.dumps(registration, indent=2))
    for row in rows:
        print(
            f"{row['scope']}: VMAF={row['vmaf_single_frame']:.4f}, "
            f"PSNR={row['psnr_rgb_db']:.4f} dB, SSIM={row['ssim_rgb']:.6f}, "
            f"MAE={row['mae_rgb_0_255']:.4f}, BDE={row['boundary_displacement_px']:.4f} px, "
            f"BF1@2px={row['boundary_f1_at_2px']:.6f}"
        )
    print(f"building_affine_aligned reverse VMAF (cloud as reference)={reverse_vmaf:.4f}")


if __name__ == "__main__":
    main()
