"""Compare a local reference screenshot with a cloud-rendered screenshot.

The 120-pixel phone status bar is removed first. Results are reported for:
1. the entire 2664x1080 game viewport (diagnostic only because it contains
   moving clouds/foliage and the local-only Development Console),
2. a conservative static building ROI, and
3. the same ROI after affine registration of the cloud image to the local one.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = Path(r"C:\Users\13577\Desktop\local.jpg")
CLOUD_PATH = Path(r"C:\Users\13577\Desktop\cloud.jpg")
OUTPUT = ROOT / "results" / "quality_metrics" / "local_vs_cloud"

STATUS_BAR_HEIGHT = 120
# Half-open viewport coordinates. This strip contains only the left building;
# it stays above the local Development Console and left of the moving tree.
ROI_X0, ROI_Y0, ROI_X1, ROI_Y1 = 100, 590, 1500, 800
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
    if not reference_edges.any() or not test_edges.any():
        raise ValueError("Canny produced an empty edge map")
    distance_to_reference = distance_transform_edt(~reference_edges)
    distance_to_test = distance_transform_edt(~test_edges)
    test_distances = distance_to_reference[test_edges]
    reference_distances = distance_to_test[reference_edges]
    test_to_reference = float(test_distances.mean())
    reference_to_test = float(reference_distances.mean())
    precision = float(np.mean(distance_to_reference[test_edges] <= BOUNDARY_TOLERANCE_PX))
    recall = float(np.mean(distance_to_test[reference_edges] <= BOUNDARY_TOLERANCE_PX))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "boundary_displacement_px": 0.5 * (test_to_reference + reference_to_test),
        "test_to_reference_px": test_to_reference,
        "reference_to_test_px": reference_to_test,
        "boundary_median_px": 0.5 * (
            float(np.median(test_distances)) + float(np.median(reference_distances))
        ),
        "boundary_p95_px": 0.5 * (
            float(np.percentile(test_distances, 95)) + float(np.percentile(reference_distances, 95))
        ),
        "boundary_clipped_mean_at_10px": 0.5 * (
            float(np.minimum(test_distances, 10.0).mean())
            + float(np.minimum(reference_distances, 10.0).mean())
        ),
        "boundary_precision_at_2px": precision,
        "boundary_recall_at_2px": recall,
        "boundary_f1_at_2px": f1,
        "boundary_miss_rate_at_2px": 1.0 - recall,
        "false_edge_rate_at_2px": 1.0 - precision,
        "test_edge_density": float(test_edges.mean()),
        "reference_edge_density": float(reference_edges.mean()),
    }


def estimate_affine(local: np.ndarray, cloud: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    local_gray = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
    cloud_gray = cv2.cvtColor(cloud, cv2.COLOR_BGR2GRAY)

    # Architectural feature mask. It avoids most sky, the right tree, the
    # foreground vegetation, and the console-covered bottom-left region.
    mask = np.zeros(local_gray.shape, dtype=np.uint8)
    cv2.rectangle(mask, (0, 360), (1850, 790), 255, -1)
    cv2.rectangle(mask, (850, 120), (1650, 600), 255, -1)

    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    kp_local, desc_local = sift.detectAndCompute(local_gray, mask)
    kp_cloud, desc_cloud = sift.detectAndCompute(cloud_gray, mask)
    if desc_local is None or desc_cloud is None:
        raise RuntimeError("SIFT could not find descriptors")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_cloud, desc_local, k=2)
    good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
    if len(good) < 12:
        raise RuntimeError(f"Too few feature matches: {len(good)}")

    cloud_points = np.float32([kp_cloud[m.queryIdx].pt for m in good])
    local_points = np.float32([kp_local[m.trainIdx].pt for m in good])
    affine, inlier_mask = cv2.estimateAffine2D(
        cloud_points,
        local_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=20000,
        confidence=0.999,
        refineIters=50,
    )
    if affine is None or inlier_mask is None:
        raise RuntimeError("Affine registration failed")

    projected = cv2.transform(cloud_points.reshape(-1, 1, 2), affine).reshape(-1, 2)
    errors = np.linalg.norm(projected - local_points, axis=1)
    inliers = inlier_mask.ravel().astype(bool)
    linear = affine[:, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    rotation_deg = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
    stats: dict[str, float | int] = {
        "local_keypoints": len(kp_local),
        "cloud_keypoints": len(kp_cloud),
        "ratio_test_matches": len(good),
        "ransac_inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "median_inlier_reprojection_error_px": float(np.median(errors[inliers])),
        "affine_a00": float(affine[0, 0]),
        "affine_a01": float(affine[0, 1]),
        "affine_tx": float(affine[0, 2]),
        "affine_a10": float(affine[1, 0]),
        "affine_a11": float(affine[1, 1]),
        "affine_ty": float(affine[1, 2]),
        "max_axis_scale": float(singular_values.max()),
        "min_axis_scale": float(singular_values.min()),
        "rotation_deg_approx": float(rotation_deg),
    }
    return affine, stats


def measure_vmaf(test_path: Path, reference_path: Path, label: str) -> float:
    log_path = OUTPUT / f"vmaf_{label}.json"
    relative_log = log_path.relative_to(ROOT).as_posix()
    filter_graph = (
        "[0:v]setpts=PTS-STARTPTS[dist];"
        "[1:v]setpts=PTS-STARTPTS[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={relative_log}:n_threads=4"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(test_path), "-i", str(reference_path),
            "-lavfi", filter_graph, "-frames:v", "1", "-f", "null", "NUL",
        ],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(log_path.read_text(encoding="utf-8"))
    return float(data["pooled_metrics"]["vmaf"]["mean"])


def measure_pair(reference_bgr: np.ndarray, test_bgr: np.ndarray, scope: str) -> dict[str, float | int | str]:
    reference_path = OUTPUT / f"{scope}_reference.png"
    test_path = OUTPUT / f"{scope}_cloud.png"
    cv2.imwrite(str(reference_path), reference_bgr)
    cv2.imwrite(str(test_path), test_bgr)
    reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
    difference = test_rgb.astype(np.float64) - reference_rgb.astype(np.float64)
    return {
        "scope": scope,
        "width": reference_rgb.shape[1],
        "height": reference_rgb.shape[0],
        "vmaf_single_frame": measure_vmaf(test_path, reference_path, scope),
        "psnr_rgb_db": float(peak_signal_noise_ratio(reference_rgb, test_rgb, data_range=255)),
        "ssim_rgb": float(structural_similarity(reference_rgb, test_rgb, channel_axis=2, data_range=255)),
        "mae_rgb_0_255": float(np.mean(np.abs(difference))),
        "rmse_rgb_0_255": float(np.sqrt(np.mean(np.square(difference)))),
        **boundary_metrics(canny_edges(reference_rgb), canny_edges(test_rgb)),
    }


def save_location_preview(local_full: np.ndarray) -> None:
    preview = local_full.copy()
    cv2.rectangle(preview, (0, 0), (preview.shape[1] - 1, STATUS_BAR_HEIGHT - 1), (0, 0, 255), 4)
    cv2.putText(preview, "Excluded phone status bar", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4, cv2.LINE_AA)
    y0 = STATUS_BAR_HEIGHT + ROI_Y0
    y1 = STATUS_BAR_HEIGHT + ROI_Y1
    overlay = preview.copy()
    cv2.rectangle(overlay, (ROI_X0, y0), (ROI_X1 - 1, y1 - 1), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.18, preview, 0.82, 0.0, preview)
    cv2.rectangle(preview, (ROI_X0, y0), (ROI_X1 - 1, y1 - 1), (0, 255, 0), 5)
    cv2.putText(preview, "Static building ROI", (ROI_X0 + 20, y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imwrite(str(OUTPUT / "measurement_regions.png"), preview)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    local_full = load_bgr(LOCAL_PATH)
    cloud_full = load_bgr(CLOUD_PATH)
    if local_full.shape != cloud_full.shape:
        raise ValueError(f"Screenshot shapes differ: {local_full.shape} vs {cloud_full.shape}")
    if local_full.shape[:2] != (1200, 2664):
        raise ValueError(f"Unexpected screenshot size: {local_full.shape}")

    save_location_preview(local_full)
    local = local_full[STATUS_BAR_HEIGHT:, :]
    cloud = cloud_full[STATUS_BAR_HEIGHT:, :]
    cv2.imwrite(str(OUTPUT / "local_viewport_no_status_bar.png"), local)
    cv2.imwrite(str(OUTPUT / "cloud_viewport_no_status_bar.png"), cloud)

    affine, registration = estimate_affine(local, cloud)
    aligned_cloud = cv2.warpAffine(
        cloud,
        affine,
        (local.shape[1], local.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    cv2.imwrite(str(OUTPUT / "cloud_viewport_affine_aligned.png"), aligned_cloud)

    local_roi = local[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
    cloud_roi = cloud[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
    aligned_roi = aligned_cloud[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]

    rows = [
        measure_pair(local, cloud, "viewport_raw"),
        measure_pair(local_roi, cloud_roi, "building_raw"),
        measure_pair(local_roi, aligned_roi, "building_affine_aligned"),
    ]
    with (OUTPUT / "local_cloud_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "reference": str(LOCAL_PATH),
        "test": str(CLOUD_PATH),
        "status_bar_crop": {"top_pixels_removed": STATUS_BAR_HEIGHT, "viewport_size": [2664, 1080]},
        "static_building_roi_viewport_coordinates": {
            "x0": ROI_X0, "y0": ROI_Y0, "x1_exclusive": ROI_X1, "y1_exclusive": ROI_Y1,
            "description": "Static left building only; excludes sky, right tree, foreground vegetation, and local console.",
        },
        "registration_cloud_to_local": registration,
        "limitations": [
            "The full viewport result is contaminated by moving clouds and foliage.",
            "The full viewport result is also contaminated by the local-only Development Console.",
            "Affine resampling can itself slightly alter sharpness.",
            "Single-frame VMAF has no temporal-motion contribution.",
            "Canny metrics describe visible image edges, not semantic or compositing boundaries.",
        ],
        "results": rows,
    }
    (OUTPUT / "local_cloud_metrics.json").write_text(
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


if __name__ == "__main__":
    main()
