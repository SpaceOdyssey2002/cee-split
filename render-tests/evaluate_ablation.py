"""Evaluate trained PPO ablations on identical deterministic scenarios."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hybrid_render_env import HybridRenderEnv


@dataclass(frozen=True)
class Scenario:
    name: str
    bandwidth: str
    workload_scale: float = 1.0
    local_capacity: float = 1.0
    edge_capacity: float = 1.0
    cloud_capacity: float = 1.0
    cloud_telemetry_valid: float = 1.0
    edge_telemetry_valid: float = 1.0
    local_telemetry_valid: float = 1.0


SCENARIOS = {
    "nominal": Scenario("nominal", "nominal"),
    "congested": Scenario("congested", "congested"),
    "fluctuating": Scenario("fluctuating", "fluctuating"),
    "heavy_workload": Scenario("heavy_workload", "nominal", workload_scale=2.0),
    "deadline_stress": Scenario("deadline_stress", "deadline_stress", workload_scale=2.5),
    "local_throttling": Scenario("local_throttling", "nominal", local_capacity=0.5),
    "edge_overload": Scenario("edge_overload", "nominal", edge_capacity=0.5),
    "remote_telemetry_missing": Scenario(
        "remote_telemetry_missing",
        "fluctuating",
        cloud_telemetry_valid=0.0,
        edge_telemetry_valid=0.0,
    ),
}

METRICS = [
    "reward",
    "latency_ms",
    "p95_latency_ms",
    "vmaf",
    "energy_mj",
    "deadline_miss_rate",
    "switch_rate",
    "node_switch_rate",
    "resolution_switch_rate",
    "rate_switch_rate",
    "switch_magnitude",
    "local_layer_ratio",
    "edge_layer_ratio",
    "cloud_layer_ratio",
]


def make_bandwidth(kind: str, length: int, episode: int) -> np.ndarray:
    if kind == "nominal":
        return np.full(length, 60.0, dtype=np.float32)
    if kind == "congested":
        return np.full(length, 6.0, dtype=np.float32)
    if kind == "fluctuating":
        x = np.arange(length, dtype=np.float32) + episode * 37
        values = 35.0 + 28.0 * np.sin(x / 31.0) + 12.0 * np.sin(x / 7.0)
        return np.clip(values, 3.0, 80.0).astype(np.float32)
    if kind == "deadline_stress":
        return np.full(length, 2.0, dtype=np.float32)
    raise ValueError(f"Unknown bandwidth profile: {kind}")


def discover_models(
    model_root: Path,
    variants: list[str],
    required_timesteps: int = 3_000_000,
) -> list[dict]:
    found = []
    for variant in variants:
        variant_dir = model_root / variant
        if not variant_dir.exists():
            print(f"[missing] variant directory: {variant_dir}")
            continue
        for run_dir in sorted(variant_dir.glob("seed_*")):
            match = re.fullmatch(r"seed_(-?\d+)", run_dir.name)
            if not match:
                continue
            model_path = run_dir / "best_model.zip"
            norm_path = run_dir / "vec_normalize.pkl"
            completion_path = run_dir / "TRAINING_COMPLETE"
            evaluations_path = run_dir / "evaluations.npz"
            completed_steps = 0
            if evaluations_path.exists():
                with np.load(evaluations_path) as evaluations:
                    completed_steps = int(np.max(evaluations["timesteps"]))
            if (
                model_path.exists()
                and norm_path.exists()
                and completion_path.exists()
                and completed_steps >= required_timesteps
            ):
                found.append(
                    {
                        "variant": variant,
                        "seed": int(match.group(1)),
                        "model_path": model_path,
                        "norm_path": norm_path,
                    }
                )
            else:
                print(
                    f"[incomplete] {run_dir}: model={model_path.exists()}, "
                    f"normalizer={norm_path.exists()}, marker={completion_path.exists()}, "
                    f"evaluated_steps={completed_steps:,}/{required_timesteps:,}"
                )
    return found


def build_eval_env(
    args: argparse.Namespace,
    variant: str,
    scenario: Scenario,
    episode: int,
) -> VecNormalize:
    raw_env = HybridRenderEnv(
        str(args.trace_file),
        str(args.network_trace),
        is_eval=True,
        ablation_mode=variant,
        edge_network_trace_path=str(args.edge_network_trace),
        payload_model=args.payload_model,
        content_profile=args.content_profile,
    )
    raw_env.eval_c_local = scenario.local_capacity
    raw_env.eval_c_edge = scenario.edge_capacity
    raw_env.eval_c_cloud = scenario.cloud_capacity
    raw_env.eval_cloud_telemetry_valid = scenario.cloud_telemetry_valid
    raw_env.eval_edge_telemetry_valid = scenario.edge_telemetry_valid
    raw_env.eval_local_telemetry_valid = scenario.local_telemetry_valid

    workload = raw_env.workload_data.to_numpy(copy=True)
    workload = np.roll(workload, -(episode * args.steps), axis=0)
    raw_env.workload_data = pd.DataFrame(
        workload * scenario.workload_scale,
        columns=["Near_Tris", "Mid_Tris", "Far_Tris"],
    )
    raw_env.total_rows = len(raw_env.workload_data)
    cloud_bandwidth = make_bandwidth(scenario.bandwidth, args.steps + 10, episode)
    raw_env.cloud_trace_loader.bandwidths = cloud_bandwidth
    raw_env.edge_trace_loader.bandwidths = np.clip(
        np.roll(cloud_bandwidth, 17) * 1.15, 0.1, 1000.0
    )

    return DummyVecEnv([lambda: raw_env])


def evaluate_episode(
    args: argparse.Namespace,
    run: dict,
    scenario: Scenario,
    episode: int,
) -> dict:
    base_venv = build_eval_env(args, run["variant"], scenario, episode)
    env = VecNormalize.load(str(run["norm_path"]), base_venv)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(run["model_path"]), env=env, device=args.device)

    latencies = []
    vmafs = []
    energies = []
    rewards = []
    switch_count = 0
    transition_count = 0
    node_switch_sum = 0.0
    resolution_switch_sum = 0.0
    rate_switch_sum = 0.0
    switch_magnitude_sum = 0.0
    node_counts = np.zeros(3, dtype=np.int64)
    previous_action = None
    obs = env.reset()

    try:
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done, infos = env.step(action)
            info = infos[0]
            applied_action = np.asarray(info["applied_action"], dtype=np.int64)

            latencies.append(float(info["latency"]))
            vmafs.append(float(info["vmaf"]))
            energies.append(float(info["energy_mJ"]))
            rewards.append(float(info["raw_reward"]))
            for node in applied_action[[0, 3, 6]]:
                node_counts[int(node)] += 1

            if previous_action is not None:
                changed = applied_action != previous_action
                switch_count += int(np.any(changed))
                node_switch_sum += float(np.mean(changed[[0, 3, 6]]))
                resolution_switch_sum += float(np.mean(changed[[1, 4, 7]]))
                rate_switch_sum += float(np.mean(changed[[2, 5, 8]]))
                action_ranges = np.tile(np.asarray([2.0, 4.0, 4.0]), 3)
                switch_magnitude_sum += float(
                    np.mean(np.abs(applied_action - previous_action) / action_ranges)
                )
                transition_count += 1
            previous_action = applied_action

            if bool(done[0]):
                break
    finally:
        env.close()

    latencies_array = np.asarray(latencies)
    total_layers = max(int(node_counts.sum()), 1)
    return {
        "variant": run["variant"],
        "seed": run["seed"],
        "scenario": scenario.name,
        "episode": episode,
        "steps": len(latencies),
        "reward": float(np.mean(rewards)),
        "latency_ms": float(np.mean(latencies_array)),
        "p95_latency_ms": float(np.percentile(latencies_array, 95)),
        "vmaf": float(np.mean(vmafs)),
        "energy_mj": float(np.mean(energies)),
        "deadline_miss_rate": float(np.mean(latencies_array > args.deadline_ms)),
        "switch_rate": switch_count / max(transition_count, 1),
        "node_switch_rate": node_switch_sum / max(transition_count, 1),
        "resolution_switch_rate": resolution_switch_sum / max(transition_count, 1),
        "rate_switch_rate": rate_switch_sum / max(transition_count, 1),
        "switch_magnitude": switch_magnitude_sum / max(transition_count, 1),
        "local_layer_ratio": node_counts[0] / total_layers,
        "edge_layer_ratio": node_counts[1] / total_layers,
        "cloud_layer_ratio": node_counts[2] / total_layers,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    # Episodes are first averaged within each trained seed; confidence intervals
    # are then computed across independently trained seeds.
    by_seed = raw.groupby(["variant", "scenario", "seed"], as_index=False)[METRICS].mean()
    rows = []
    for (variant, scenario), group in by_seed.groupby(["variant", "scenario"]):
        row = {"variant": variant, "scenario": scenario, "n_seeds": len(group)}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            if len(values) > 1:
                row[f"{metric}_ci95"] = float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))
            else:
                row[f"{metric}_ci95"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "variant"]).reset_index(drop=True)


def relative_to_full(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"] == "full"].set_index("scenario")
    rows = []
    for _, row in summary.iterrows():
        scenario = row["scenario"]
        if scenario not in baseline.index:
            continue
        result = {"variant": row["variant"], "scenario": scenario}
        for metric in METRICS:
            current = float(row[f"{metric}_mean"])
            full = float(baseline.loc[scenario, f"{metric}_mean"])
            result[f"{metric}_delta"] = current - full
            result[f"{metric}_delta_pct"] = 100.0 * (current - full) / max(abs(full), 1e-9)
        rows.append(result)
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    scenarios = list(dict.fromkeys(summary["scenario"]))
    variants = list(dict.fromkeys(summary["variant"]))
    x = np.arange(len(scenarios), dtype=float)
    width = 0.82 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 1.8), 5.2))
    for index, variant in enumerate(variants):
        subset = summary[summary["variant"] == variant].set_index("scenario")
        means = [subset.loc[s, f"{metric}_mean"] if s in subset.index else np.nan for s in scenarios]
        errors = [subset.loc[s, f"{metric}_ci95"] if s in subset.index else 0.0 for s in scenarios]
        position = x - 0.41 + width / 2 + index * width
        ax.bar(position, means, width, yerr=errors, capsize=2, label=variant)

    ax.set_xticks(x, scenarios, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(4, len(variants)), fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_latex_table(frame: pd.DataFrame, output: Path) -> None:
    """Write a compact LaTeX table without pandas' optional Jinja dependency."""

    def format_cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("_", r"\_")

    alignment = "ll" + "r" * max(len(frame.columns) - 2, 0)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(format_cell(column) for column in frame.columns) + r" \\")
    lines.append("\\midrule")
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(format_cell(value) for value in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("ablation_models"))
    parser.add_argument("--trace-file", type=Path, default=Path("train_trace.csv"))
    parser.add_argument("--network-trace", type=Path, default=Path("train_network.txt"))
    parser.add_argument("--edge-network-trace", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_results"))
    parser.add_argument("--variants", nargs="+", default=list(HybridRenderEnv.SUPPORTED_ABLATIONS))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--required-timesteps",
        type=int,
        default=3_000_000,
        help="Ignore seeds whose evaluation log has not reached this training budget.",
    )
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--payload-model",
        choices=HybridRenderEnv.SUPPORTED_PAYLOAD_MODELS,
        default="lookup",
    )
    parser.add_argument(
        "--content-profile",
        choices=HybridRenderEnv.SUPPORTED_CONTENT_PROFILES,
        default="mean",
    )
    args = parser.parse_args()
    if args.edge_network_trace is None:
        args.edge_network_trace = args.network_trace

    unknown_variants = sorted(set(args.variants) - set(HybridRenderEnv.SUPPORTED_ABLATIONS))
    unknown_scenarios = sorted(set(args.scenarios) - set(SCENARIOS))
    if unknown_variants:
        parser.error(f"Unknown variants: {unknown_variants}")
    if unknown_scenarios:
        parser.error(f"Unknown scenarios: {unknown_scenarios}")
    if not all(path.exists() for path in (
        args.trace_file, args.network_trace, args.edge_network_trace
    )):
        parser.error("Trace files do not exist")
    args.model_root = args.model_root.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_models(args.model_root, args.variants, args.required_timesteps)
    if not runs:
        raise FileNotFoundError(f"No trained ablation models found under {args.model_root}")

    records = []
    total = len(runs) * len(args.scenarios) * args.episodes
    completed = 0
    for run in runs:
        for scenario_name in args.scenarios:
            scenario = SCENARIOS[scenario_name]
            for episode in range(args.episodes):
                records.append(evaluate_episode(args, run, scenario, episode))
                completed += 1
                print(
                    f"[{completed}/{total}] {run['variant']} seed={run['seed']} "
                    f"scenario={scenario_name} episode={episode}"
                )

    raw = pd.DataFrame(records)
    summary = summarize(raw)
    relative = relative_to_full(summary)
    raw.to_csv(args.output_dir / "ablation_raw.csv", index=False)
    summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    relative.to_csv(args.output_dir / "ablation_relative_to_full.csv", index=False)

    table_columns = [
        "variant",
        "scenario",
        "n_seeds",
        "latency_ms_mean",
        "vmaf_mean",
        "deadline_miss_rate_mean",
        "switch_rate_mean",
        "reward_mean",
    ]
    write_latex_table(summary[table_columns], args.output_dir / "ablation_summary.tex")

    plot_metric(summary, "latency_ms", "Mean latency (ms)", args.output_dir / "latency.png")
    plot_metric(summary, "vmaf", "Mean VMAF", args.output_dir / "vmaf.png")
    plot_metric(
        summary,
        "deadline_miss_rate",
        "Deadline miss rate",
        args.output_dir / "deadline_miss_rate.png",
    )
    plot_metric(summary, "switch_rate", "Action switch rate", args.output_dir / "switch_rate.png")

    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (args.output_dir / "evaluation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
