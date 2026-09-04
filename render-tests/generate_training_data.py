"""Generate reproducible workload and independent Cloud/Edge network traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOAD_LEVELS = np.array([60_000, 250_000, 800_000, 1_600_000, 2_800_000], dtype=float)
LAYER_ARCHETYPES = np.array(
    [
        [0.70, 0.20, 0.10],  # Near-heavy motion/interaction.
        [0.20, 0.65, 0.15],  # Mid-heavy scene.
        [0.12, 0.25, 0.63],  # Far-heavy open world.
        [0.34, 0.33, 0.33],  # Balanced scene.
        [0.45, 0.35, 0.20],  # Large object/terrain spanning depth bands.
    ],
    dtype=float,
)


def generate_workload(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    output = np.empty((rows, 3), dtype=np.int64)
    cursor = 0
    previous_total = LOAD_LEVELS[2]
    previous_share = LAYER_ARCHETYPES[3].copy()

    while cursor < rows:
        duration = int(rng.integers(80, 1200))
        end = min(cursor + duration, rows)
        target_total = float(rng.choice(LOAD_LEVELS)) * rng.uniform(0.75, 1.25)
        target_share = LAYER_ARCHETYPES[int(rng.integers(0, len(LAYER_ARCHETYPES)))]

        for index in range(cursor, end):
            previous_total = 0.96 * previous_total + 0.04 * target_total
            total = max(3_000.0, previous_total + rng.normal(0.0, previous_total * 0.04))

            sampled_share = rng.dirichlet(target_share * 30.0 + 0.5)
            previous_share = 0.92 * previous_share + 0.08 * sampled_share
            previous_share /= previous_share.sum()
            layer_tris = np.maximum(100, np.rint(total * previous_share)).astype(np.int64)
            output[index] = layer_tris
        cursor = end

    return pd.DataFrame(output, columns=["Near_Tris", "Mid_Tris", "Far_Tris"])


def generate_network_pair(rows: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.empty(rows, dtype=np.float32)
    edge = np.empty(rows, dtype=np.float32)
    cursor = 0
    common_factor = 1.0

    # Cloud typically has more throughput; Edge has a shorter path but a lower ceiling.
    cloud_levels = np.array([120.0, 70.0, 35.0, 12.0, 4.0])
    edge_levels = np.array([90.0, 55.0, 28.0, 10.0, 3.0])

    while cursor < rows:
        duration = int(rng.integers(50, 900))
        end = min(cursor + duration, rows)
        state = int(rng.integers(0, len(cloud_levels)))
        cloud_target = cloud_levels[state] * rng.uniform(0.8, 1.2)
        edge_state = int(np.clip(state + rng.choice([-1, 0, 0, 0, 1]), 0, 4))
        edge_target = edge_levels[edge_state] * rng.uniform(0.8, 1.2)

        for index in range(cursor, end):
            common_factor = np.clip(
                0.97 * common_factor + 0.03 * rng.lognormal(mean=0.0, sigma=0.18),
                0.45,
                1.8,
            )
            cloud[index] = max(0.3, cloud_target * common_factor * rng.lognormal(0.0, 0.08))
            edge[index] = max(0.3, edge_target * common_factor * rng.lognormal(0.0, 0.10))
        cursor = end

    # Short independent dropouts are important because WebRTC estimates can disappear.
    for trace in (cloud, edge):
        for _ in range(max(1, rows // 20_000)):
            start = int(rng.integers(0, max(rows - 30, 1)))
            length = int(rng.integers(5, 30))
            trace[start:start + length] = rng.uniform(0.3, 1.5)

    return np.clip(cloud, 0.3, 200.0), np.clip(edge, 0.3, 200.0)


def write_split(output_dir: Path, name: str, rows: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    workload = generate_workload(rows, rng)
    cloud, edge = generate_network_pair(rows, rng)

    workload_path = output_dir / f"{name}_workload.csv"
    cloud_path = output_dir / f"{name}_cloud_network.txt"
    edge_path = output_dir / f"{name}_edge_network.txt"
    workload.to_csv(workload_path, index=False)
    np.savetxt(cloud_path, cloud, fmt="%.3f")
    np.savetxt(edge_path, edge, fmt="%.3f")

    return {
        "name": name,
        "rows": rows,
        "seed": seed,
        "workload": str(workload_path.resolve()),
        "cloud_network": str(cloud_path.resolve()),
        "edge_network": str(edge_path.resolve()),
        "total_tris_percentiles": {
            str(p): float(np.percentile(workload.sum(axis=1), p))
            for p in (1, 5, 50, 95, 99)
        },
        "cloud_bw_percentiles": {str(p): float(np.percentile(cloud, p)) for p in (1, 5, 50, 95, 99)},
        "edge_bw_percentiles": {str(p): float(np.percentile(edge, p)) for p in (1, 5, 50, 95, 99)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--train-rows", type=int, default=300_000)
    parser.add_argument("--eval-rows", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()
    if args.train_rows <= 0 or args.eval_rows <= 0:
        parser.error("row counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train": write_split(args.output_dir, "train", args.train_rows, args.seed),
        "eval": write_split(args.output_dir, "eval", args.eval_rows, args.seed + 1),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
