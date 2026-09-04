"""Fit the system models used in the paper.

Render CSV columns:
    node,triangles,pixels,render_ms

Stage CSV columns:
    pixels,encode_ms,decode_ms,display_pixels,composition_ms

Payload CSV columns:
    qp,pixels and one of payload_mbits, frame_size_bytes, bitrate_kbps, bitrate
An optional ``scene`` column produces one content factor kappa per scene.

Example:
    python fit_system_models.py ^
      --render-csv unity_render_calibration.csv ^
      --capacity End=2 --capacity Edge=10 --capacity Cloud=100 ^
      --stage-csv pipeline_stage_calibration.csv ^
      --payload-csv encoded_frames.csv ^
      --fps 60 --output system_model_coefficients.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_capacities(items):
    capacities = {}
    for item in items:
        node, value = item.split("=", 1)
        capacities[node.strip()] = float(value)
    return capacities


def r_squared(actual, predicted):
    residual = np.sum((actual - predicted) ** 2)
    total = np.sum((actual - np.mean(actual)) ** 2)
    return float(1.0 - residual / total) if total > 0 else 1.0


def fit_nonnegative_two_feature_model(design, target, description):
    scales = np.maximum(np.max(np.abs(design), axis=0), 1.0)
    scaled_coefficients, _, _, _ = np.linalg.lstsq(
        design / scales, target, rcond=None
    )
    coefficients = scaled_coefficients / scales
    if np.any(coefficients < 0):
        raise ValueError(
            f"The fitted {description} coefficient is negative. Collect more varied "
            "views and resolutions, or verify the timing samples."
        )
    return coefficients


def fit_render_model(path, capacities):
    data = pd.read_csv(path)
    required = {"node", "triangles", "pixels", "render_ms"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Render CSV is missing columns: {sorted(missing)}")

    nodes = sorted(data["node"].astype(str).unique())
    unknown = sorted(set(nodes) - set(capacities))
    if capacities and unknown:
        raise ValueError(
            "Missing effective throughput for nodes: "
            + ", ".join(unknown)
            + ". Add --capacity Node=GFLOPS."
        )

    data = data.copy()
    data = data[
        (data["triangles"] >= 0)
        & (data["pixels"] > 0)
        & (data["render_ms"] > 0)
    ]
    if len(data) < 3:
        raise ValueError("At least three valid render samples are required.")

    # Directly identifiable ratios from
    # T_ms = 1000 (g_v D + g_f R) / C.
    per_node = {}
    for node in nodes:
        subset = data[data["node"].astype(str) == node]
        if len(subset) < 3:
            raise ValueError(f"Node {node!r} needs at least three render samples.")
        design = subset[["triangles", "pixels"]].to_numpy(float)
        target_ms = subset["render_ms"].to_numpy(float)
        time_coefficients = fit_nonnegative_two_feature_model(
            design, target_ms, f"render-time model for node {node}"
        )
        predicted_ms = design @ time_coefficients
        per_node[node] = {
            "g_v_over_c_s_per_triangle": float(time_coefficients[0] / 1000.0),
            "g_f_over_c_s_per_pixel": float(time_coefficients[1] / 1000.0),
            "render_ms_per_triangle": float(time_coefficients[0]),
            "render_ms_per_pixel": float(time_coefficients[1]),
            "samples": int(len(subset)),
            "render_time_r2": r_squared(target_ms, predicted_ms),
        }

    result = {
        "per_node_ratios": per_node,
        "samples": int(len(data)),
    }

    # Absolute effective-FLOP coefficients require a capacity anchor for every
    # node; otherwise only g_v/C and g_f/C are identifiable.
    if capacities:
        data["capacity_gflops"] = data["node"].astype(str).map(capacities)
        if np.any(data["capacity_gflops"] <= 0):
            raise ValueError("Every effective GFLOPS value must be positive.")
        workload_gflop = (
            data["render_ms"].to_numpy(float)
            * data["capacity_gflops"].to_numpy(float)
            / 1000.0
        )
        design = data[["triangles", "pixels"]].to_numpy(float)
        coefficients = fit_nonnegative_two_feature_model(
            design, workload_gflop, "effective-FLOP model"
        )
        predicted_gflop = design @ coefficients
        predicted_ms = (
            1000.0
            * predicted_gflop
            / data["capacity_gflops"].to_numpy(float)
        )
        result["effective_flop_model"] = {
            "flop_per_triangle": float(coefficients[0] * 1e9),
            "flop_per_pixel": float(coefficients[1] * 1e9),
            "g_v_gflop_per_triangle": float(coefficients[0]),
            "g_f_gflop_per_pixel": float(coefficients[1]),
            "capacity_gflops": capacities,
            "render_time_r2": r_squared(
                data["render_ms"].to_numpy(float), predicted_ms
            ),
        }

    return result


def fit_affine_stage(pixels, times_ms, name):
    valid = np.isfinite(pixels) & np.isfinite(times_ms) & (pixels > 0) & (times_ms >= 0)
    pixels = pixels[valid]
    times_ms = times_ms[valid]
    if len(times_ms) < 3:
        raise ValueError(f"Stage {name!r} needs at least three valid samples.")

    scale = max(float(np.max(np.abs(pixels))), 1.0)
    design = np.column_stack([np.ones(len(pixels)), pixels / scale])
    coefficients, _, _, _ = np.linalg.lstsq(design, times_ms, rcond=None)
    intercept_ms = float(coefficients[0])
    slope_ms_per_pixel = float(coefficients[1] / scale)
    predicted_ms = intercept_ms + slope_ms_per_pixel * pixels
    return {
        "intercept_ms": intercept_ms,
        "slope_ms_per_pixel": slope_ms_per_pixel,
        "samples": int(len(times_ms)),
        "r2": r_squared(times_ms, predicted_ms),
    }


def fit_stage_models(path):
    data = pd.read_csv(path)
    required = {
        "pixels",
        "encode_ms",
        "decode_ms",
        "display_pixels",
        "composition_ms",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Stage CSV is missing columns: {sorted(missing)}")

    pixels = data["pixels"].to_numpy(float)
    display_pixels = data["display_pixels"].to_numpy(float)
    return {
        "encoding": fit_affine_stage(
            pixels, data["encode_ms"].to_numpy(float), "encoding"
        ),
        "decoding": fit_affine_stage(
            pixels, data["decode_ms"].to_numpy(float), "decoding"
        ),
        "composition": fit_affine_stage(
            display_pixels,
            data["composition_ms"].to_numpy(float),
            "composition",
        ),
    }


def payload_mbits(data, fps):
    if "payload_mbits" in data:
        return data["payload_mbits"].to_numpy(float)
    if "frame_size_bytes" in data:
        return data["frame_size_bytes"].to_numpy(float) * 8.0 / 1e6
    if "bitrate_kbps" in data:
        return data["bitrate_kbps"].to_numpy(float) / (1000.0 * fps)
    if "bitrate" in data:
        return data["bitrate"].to_numpy(float) / (1000.0 * fps)
    raise ValueError(
        "Payload CSV needs payload_mbits, frame_size_bytes, bitrate_kbps, or bitrate."
    )


def fit_payload_model(path, fps, qp_reference):
    data = pd.read_csv(path).copy()
    required = {"qp", "pixels"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Payload CSV is missing columns: {sorted(missing)}")

    data["payload_mbits"] = payload_mbits(data, fps)
    data = data[
        (data["qp"] >= 0)
        & (data["pixels"] > 0)
        & (data["payload_mbits"] > 0)
    ]
    if len(data) < 3:
        raise ValueError("At least three valid encoded-frame samples are required.")

    scenes = (
        sorted(data["scene"].astype(str).unique())
        if "scene" in data.columns
        else ["default"]
    )
    design_columns = [
        np.ones(len(data)),
        data["qp"].to_numpy(float) - qp_reference,
    ]
    if "scene" in data.columns:
        for scene in scenes[1:]:
            design_columns.append(
                (data["scene"].astype(str) == scene).to_numpy(float)
            )
    design = np.column_stack(design_columns)
    target = np.log(
        data["payload_mbits"].to_numpy(float) / data["pixels"].to_numpy(float)
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients

    kappas = {scenes[0]: 1.0}
    for scene, coefficient in zip(scenes[1:], coefficients[2:]):
        kappas[scene] = float(np.exp(coefficient))

    return {
        "qp_reference": float(qp_reference),
        "beta_ref_mbit_per_pixel": float(np.exp(coefficients[0])),
        "lambda": float(-coefficients[1]),
        "content_kappa": kappas,
        "samples": int(len(data)),
        "log_payload_density_r2": r_squared(target, predicted),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-csv", type=Path)
    parser.add_argument("--stage-csv", type=Path)
    parser.add_argument("--payload-csv", type=Path)
    parser.add_argument(
        "--capacity",
        action="append",
        default=[],
        metavar="NODE=GFLOPS",
        help="Effective node throughput; repeat once per node in render CSV.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--qp-reference", type=float, default=20.0)
    parser.add_argument(
        "--output", type=Path, default=Path("system_model_coefficients.json")
    )
    args = parser.parse_args()

    if (
        args.render_csv is None
        and args.stage_csv is None
        and args.payload_csv is None
    ):
        parser.error(
            "Provide --render-csv, --stage-csv, --payload-csv, or a combination."
        )

    result = {}
    if args.render_csv is not None:
        result["render"] = fit_render_model(
            args.render_csv, parse_capacities(args.capacity)
        )
    if args.stage_csv is not None:
        result["pipeline_stages"] = fit_stage_models(args.stage_csv)
    if args.payload_csv is not None:
        result["payload"] = fit_payload_model(
            args.payload_csv, args.fps, args.qp_reference
        )

    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
