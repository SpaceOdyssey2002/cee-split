"""Evaluate strong model-based baselines without overwriting paper results.

The script adds three baselines to the existing five-scenario evaluation:

* Best Static: one globally fixed configuration selected only from the
  independent training split and then reused unchanged in every test scenario.
* Scenario-Optimal Static Oracle: one fixed configuration selected after
  seeing the complete evaluated scenario window.  It is an optimistic oracle,
  rather than a deployable baseline.
* One-step Greedy: an exhaustive one-step oracle over all unique physical
  execution configurations, including the switching penalty from the
  previously selected action.

Actions that differ only in unused local rates or in sub-maximal rate requests
on a shared remote node are physically equivalent in the current simulator.
Rendering resolutions remain independent for every layer.  The candidate set
contains one representative for each unique execution configuration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evalute import Baselines, UnifiedEvaluator, select_best_completed_run


STRATEGIES = (
    "DRL_Agent",
    "End_Only",
    "Cloud_Only",
    "Heuristic",
    "Best_Static",
    "Scenario_Optimal_Static_Oracle",
    "One_Step_Greedy",
)


def canonical_physical_actions() -> np.ndarray:
    """Return one policy action for every unique physical configuration."""
    actions = []
    for layer_nodes in itertools.product(range(3), repeat=3):
        active_remote_nodes = tuple(sorted(set(layer_nodes) & {1, 2}))
        for layer_resolutions in itertools.product(range(5), repeat=3):
            for selected_rates in itertools.product(
                range(5), repeat=len(active_remote_nodes)
            ):
                node_rate = dict(zip(active_remote_nodes, selected_rates))
                action = []
                for layer, node in enumerate(layer_nodes):
                    rate = 4 if node == 0 else node_rate[node]
                    action.extend((node, layer_resolutions[layer], rate))
                actions.append(action)
    return np.asarray(actions, dtype=np.int8)


class VectorizedImmediateModel:
    """Vectorized copy of HybridRenderEnv's immediate outcome equations."""

    def __init__(self, env, actions: np.ndarray):
        self.env = env
        self.actions = np.asarray(actions, dtype=np.int64)
        self.nodes = self.actions[:, (0, 3, 6)]
        self.resolutions = self.actions[:, (1, 4, 7)]
        self.rates = self.actions[:, (2, 5, 8)]
        self.rows = np.arange(len(self.actions))
        self.pixel_lookup = np.asarray([env.RES_MAP[i] for i in range(5)], dtype=float)
        self.rate_lookup = np.asarray(
            [env.TARGET_BITRATE_KBPS[i] for i in range(5)], dtype=float
        )
        self.qp_lookup = np.asarray([env.QP_MAP[i] for i in range(5)], dtype=float)
        self.render_d = np.asarray(
            [env.RENDER_COEFFICIENTS[node][0] for node in range(3)], dtype=float
        )
        self.render_r = np.asarray(
            [env.RENDER_COEFFICIENTS[node][1] for node in range(3)], dtype=float
        )

    def evaluate(self, previous_action: np.ndarray, include_switch: bool = True) -> dict[str, np.ndarray]:
        env = self.env
        row = env.workload_data.iloc[env.data_ptr % env.total_rows]
        triangles = np.asarray(
            [row["Near_Tris"], row["Mid_Tris"], row["Far_Tris"]], dtype=float
        )
        active_layers = triangles >= 10.0
        if not np.any(active_layers):
            raise RuntimeError("Strong baselines require at least one active rendering layer")

        count = len(self.actions)
        node_triangles = np.zeros((count, 3), dtype=float)
        node_render_pixels = np.zeros((count, 3), dtype=float)
        node_resolution = np.zeros((count, 3), dtype=np.int64)
        node_rate = np.zeros((count, 3), dtype=np.int64)
        node_active = np.zeros((count, 3), dtype=bool)

        for node in range(3):
            assigned = (self.nodes == node) & active_layers[None, :]
            node_active[:, node] = np.any(assigned, axis=1)
            node_triangles[:, node] = assigned @ triangles
            node_render_pixels[:, node] = (
                assigned * self.pixel_lookup[self.resolutions]
            ).sum(axis=1)
            node_resolution[:, node] = np.max(
                np.where(assigned, self.resolutions, -1), axis=1
            ).clip(min=0)
            node_rate[:, node] = np.max(
                np.where(assigned, self.rates, -1), axis=1
            ).clip(min=0)

        node_pixels = self.pixel_lookup[node_resolution]
        payload = np.zeros((count, 3), dtype=float)
        for node in (1, 2):
            if env.payload_model == "lookup":
                payload[:, node] = self.rate_lookup[node_rate[:, node]] / 60_000.0
            else:
                payload[:, node] = (
                    env.content_kappa
                    * node_pixels[:, node]
                    * env.PAYLOAD_BETA_REF
                    * np.exp(
                        -env.PAYLOAD_LAMBDA
                        * (self.qp_lookup[node_rate[:, node]] - env.PAYLOAD_QP_REF)
                    )
                )
            payload[:, node] *= node_active[:, node]

        weights = np.asarray(env.VMAF_WEIGHTS, dtype=float)
        active_weight = float(np.sum(weights[active_layers]))
        weighted_vmaf = np.zeros(count, dtype=float)
        for layer in np.flatnonzero(active_layers):
            layer_node = self.nodes[:, layer]
            effective_resolution = self.resolutions[:, layer]
            effective_rate = node_rate[self.rows, layer_node]
            local_score = env.local_vmaf_table[effective_resolution, effective_rate]
            remote_score = env.remote_vmaf_table[effective_resolution, effective_rate]
            layer_score = np.where(layer_node == 0, local_score, remote_score)
            weighted_vmaf += weights[layer] * layer_score
        vmaf = weighted_vmaf / active_weight

        capacity = np.asarray(
            [env.local_capacity_factor, env.edge_capacity_factor, env.cloud_capacity_factor],
            dtype=float,
        )
        rendering = (
            self.render_d[None, :] * node_triangles
            + self.render_r[None, :] * node_render_pixels
        ) / np.maximum(capacity[None, :], 1e-6)
        rendering *= node_active

        encoding = np.zeros((count, 3), dtype=float)
        decoding = np.zeros((count, 3), dtype=float)
        transmission = np.zeros((count, 3), dtype=float)
        finish = np.zeros((count, 3), dtype=float)
        finish[:, 0] = rendering[:, 0]
        for node, telemetry_name in ((1, "edge"), (2, "cloud")):
            active = node_active[:, node]
            encoding[:, node] = (0.5 + node_pixels[:, node] / 2_000_000.0) * active
            decoding[:, node] = (1.0 + node_pixels[:, node] / 2_000_000.0) * active
            bandwidth = max(float(env.node_telemetry[telemetry_name]["bw"]), 0.1)
            transmission[:, node] = np.minimum(
                payload[:, node] / bandwidth * 1000.0, 2000.0
            )
            propagation = float(env.node_telemetry[telemetry_name]["rtt"]) / 2.0
            finish[:, node] = (
                rendering[:, node]
                + encoding[:, node]
                + propagation
                + transmission[:, node]
                + decoding[:, node]
            ) * active

        latency = np.max(finish, axis=1) + env.COMPOSITION_TIME_MS
        energy = 5.0 * rendering[:, 0] + 1.5 * (
            transmission[:, 1] + transmission[:, 2]
        ) + latency

        reward_quality = 0.15 * (vmaf - 50.0)
        reward_latency = -0.03 * latency
        deadline_excess = np.maximum(latency - 100.0, 0.0)
        if env.ablation_mode not in ("no_barrier", "no_barrier_no_switch"):
            deadline_penalty = np.where(
                deadline_excess > 0.0,
                2.0 + 0.08 * deadline_excess + 0.002 * deadline_excess**2,
                0.0,
            )
            reward_latency -= deadline_penalty
        reward_energy = -0.004 * energy
        if include_switch:
            previous = np.asarray(previous_action, dtype=np.int64)
            switch = 0.2 * np.any(self.actions != previous[None, :], axis=1)
        else:
            switch = np.zeros(count, dtype=float)
        reward = np.clip(
            reward_quality + reward_latency + reward_energy - switch, -30.0, 15.0
        )
        return {
            "reward": reward,
            "vmaf": vmaf,
            "latency": latency,
            "energy_mJ": energy,
        }


def scenario_bandwidths(evaluator: UnifiedEvaluator, steps: int) -> dict[str, np.ndarray]:
    sequences = {}
    rng = np.random.default_rng(20260615)
    for name, bandwidth, _, _ in evaluator.scenarios:
        if bandwidth == "dynamic":
            sequence = np.clip(
                [22.0 + 24.0 * np.sin(i / 5.0) + rng.normal(0, 3.0) for i in range(steps + 10)],
                0.3,
                200.0,
            )
        else:
            sequence = np.full(steps + 10, float(bandwidth), dtype=float)
        sequences[name] = np.asarray(sequence, dtype=np.float32)
    return sequences


def advance_with_action(evaluator: UnifiedEvaluator, action: np.ndarray):
    observation, _, _, infos = evaluator.env.step([np.asarray(action, dtype=np.int64)])
    return observation, infos[0]


def choose_scenario_optimal_static(
    evaluator: UnifiedEvaluator,
    model: VectorizedImmediateModel,
    steps: int,
    start_index: int,
) -> tuple[np.ndarray, float]:
    evaluator.reset_env(start_index)
    reward_sum = np.zeros(len(model.actions), dtype=float)
    initial_action = evaluator.raw_env.last_act_vec.copy()
    placeholder = Baselines.end_only()
    for step in range(steps):
        outcome = model.evaluate(
            initial_action if step == 0 else model.actions[0],
            include_switch=(step == 0),
        )
        reward_sum += outcome["reward"]
        advance_with_action(evaluator, placeholder)
    best_index = int(np.argmax(reward_sum))
    return model.actions[best_index].copy(), float(reward_sum[best_index] / steps)


def _trace_rtt(node: int, bandwidth: float) -> float:
    """Match HybridRenderEnv's deterministic bandwidth-to-RTT calibration."""

    if node == 1:
        return 8.0 if bandwidth >= 80.0 else (18.0 if bandwidth >= 20.0 else 45.0)
    return 18.0 if bandwidth >= 80.0 else (35.0 if bandwidth >= 20.0 else 70.0)


def choose_train_selected_best_static(
    model: VectorizedImmediateModel,
    workload_path: str,
    cloud_trace_path: str,
    edge_trace_path: str,
    samples: int,
) -> tuple[np.ndarray, float, dict]:
    """Select one fixed action without consulting any evaluation scenario.

    Equally spaced samples cover the independent training split without making
    baseline selection depend on a single contiguous workload/network regime.
    Capacity factors are held at the calibrated value c=1, matching the five
    macro-evaluation scenarios.
    """

    workload = pd.read_csv(workload_path).iloc[:, :3].dropna().reset_index(drop=True)
    workload.columns = ["Near_Tris", "Mid_Tris", "Far_Tris"]
    cloud = np.atleast_1d(np.loadtxt(cloud_trace_path)).astype(float)
    edge = np.atleast_1d(np.loadtxt(edge_trace_path)).astype(float)
    available = min(len(workload), len(cloud), len(edge))
    if available <= 0:
        raise ValueError("Training split is empty; cannot select Best Static")
    sample_count = min(max(int(samples), 1), available)
    sample_indices = np.linspace(0, available - 1, sample_count, dtype=int)

    env = model.env
    saved_workload = env.workload_data
    saved_total_rows = env.total_rows
    saved_data_ptr = env.data_ptr
    saved_telemetry = {
        name: values.copy() for name, values in env.node_telemetry.items()
    }
    saved_capacities = (
        env.local_capacity_factor,
        env.edge_capacity_factor,
        env.cloud_capacity_factor,
    )
    reward_sum = np.zeros(len(model.actions), dtype=float)
    try:
        env.workload_data = workload
        env.total_rows = len(workload)
        env.local_capacity_factor = 1.0
        env.edge_capacity_factor = 1.0
        env.cloud_capacity_factor = 1.0
        for index in sample_indices:
            env.data_ptr = int(index)
            edge_bw = float(edge[index])
            cloud_bw = float(cloud[index])
            env.node_telemetry["edge"]["bw"] = edge_bw
            env.node_telemetry["edge"]["rtt"] = _trace_rtt(1, edge_bw)
            env.node_telemetry["cloud"]["bw"] = cloud_bw
            env.node_telemetry["cloud"]["rtt"] = _trace_rtt(2, cloud_bw)
            reward_sum += model.evaluate(model.actions[0], include_switch=False)["reward"]
    finally:
        env.workload_data = saved_workload
        env.total_rows = saved_total_rows
        env.data_ptr = saved_data_ptr
        env.node_telemetry = saved_telemetry
        (
            env.local_capacity_factor,
            env.edge_capacity_factor,
            env.cloud_capacity_factor,
        ) = saved_capacities

    best_index = int(np.argmax(reward_sum))
    metadata = {
        "selection_split": "train",
        "sample_count": int(sample_count),
        "available_training_rows": int(available),
        "selection_rule": "equally spaced samples over the independent training split",
        "capacity_factors": [1.0, 1.0, 1.0],
    }
    return (
        model.actions[best_index].copy(),
        float(reward_sum[best_index] / sample_count),
        metadata,
    )


def run_strategy(
    evaluator: UnifiedEvaluator,
    model: VectorizedImmediateModel,
    strategy: str,
    steps: int,
    start_index: int,
    static_action: np.ndarray | None = None,
) -> list[dict]:
    observation = evaluator.reset_env(start_index)
    records = []
    for step in range(steps):
        started = time.perf_counter()
        if strategy == "DRL_Agent":
            predicted, _ = evaluator.model.predict(observation, deterministic=True)
            action = np.asarray(predicted[0], dtype=np.int64)
        elif strategy == "End_Only":
            action = Baselines.end_only()
        elif strategy == "Cloud_Only":
            action = Baselines.cloud_only()
        elif strategy == "Heuristic":
            action = Baselines.heuristic(evaluator.current_bottleneck_bandwidth())
        elif strategy in ("Best_Static", "Scenario_Optimal_Static_Oracle"):
            action = np.asarray(static_action, dtype=np.int64)
        elif strategy == "One_Step_Greedy":
            outcome = model.evaluate(evaluator.raw_env.last_act_vec, include_switch=True)
            action = model.actions[int(np.argmax(outcome["reward"]))]
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        decision_ms = (time.perf_counter() - started) * 1000.0
        observation, info = advance_with_action(evaluator, action)
        records.append(
            {
                "Step": step,
                "Strategy": strategy,
                "VMAF": float(info["vmaf"]),
                "Latency": float(info["latency"]),
                "Energy_mJ": float(info["energy_mJ"]),
                "Reward": float(info["raw_reward"]),
                "Deadline_Met": int(float(info["latency"]) <= 100.0),
                "Decision_ms": decision_ms,
                "Action": json.dumps(info["applied_action"]),
            }
        )
    return records


def validate_vectorized_model(evaluator: UnifiedEvaluator, model: VectorizedImmediateModel):
    evaluator.set_scenario(np.full(20, 37.0, dtype=float), 1.2)
    evaluator.reset_env(0)
    sample_indices = np.linspace(0, len(model.actions) - 1, 9, dtype=int)
    preview = model.evaluate(evaluator.raw_env.last_act_vec, include_switch=True)
    for index in sample_indices:
        action = model.actions[index]
        evaluator.reset_env(0)
        expected = {
            key: float(preview[key][index])
            for key in ("reward", "vmaf", "latency", "energy_mJ")
        }
        _, info = advance_with_action(evaluator, action)
        actual = {
            "reward": float(info["raw_reward"]),
            "vmaf": float(info["vmaf"]),
            "latency": float(info["latency"]),
            "energy_mJ": float(info["energy_mJ"]),
        }
        for key in expected:
            if not np.isclose(expected[key], actual[key], rtol=1e-6, atol=1e-6):
                raise AssertionError(
                    f"Vectorized model mismatch for {key}: "
                    f"expected={expected[key]}, actual={actual[key]}, action={action.tolist()}"
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="ablation_models_decoupled_vp9")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vec-normalize", default=None)
    parser.add_argument("--workload", default="datasets/eval_workload.csv")
    parser.add_argument("--cloud-trace", default="datasets/eval_cloud_network.txt")
    parser.add_argument("--edge-trace", default="datasets/eval_edge_network.txt")
    parser.add_argument("--train-workload", default="datasets/train_workload.csv")
    parser.add_argument("--train-cloud-trace", default="datasets/train_cloud_network.txt")
    parser.add_argument("--train-edge-trace", default="datasets/train_edge_network.txt")
    parser.add_argument(
        "--static-train-samples",
        type=int,
        default=5000,
        help="Equally spaced training-split samples used to select fair Best Static",
    )
    parser.add_argument("--output-dir", default="strong_baseline_results")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--ablation-mode", default="full")
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.model_path) != bool(args.vec_normalize):
        raise ValueError("--model-path and --vec-normalize must be supplied together")
    if args.model_path:
        model_path, norm_path = args.model_path, args.vec_normalize
    else:
        score, model_path, norm_path = select_best_completed_run(args.model_root)
        print(f"Selected full model with validation score {score:.3f}: {model_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = UnifiedEvaluator(
        model_path,
        norm_path,
        workload_path=args.workload,
        cloud_trace_path=args.cloud_trace,
        edge_trace_path=args.edge_trace,
        log_dir=str(output_dir),
        ablation_mode=args.ablation_mode,
    )
    actions = canonical_physical_actions()
    print(f"Enumerating {len(actions):,} unique physical configurations")
    immediate_model = VectorizedImmediateModel(evaluator.raw_env, actions)
    validate_vectorized_model(evaluator, immediate_model)
    print("Vectorized immediate model matches HybridRenderEnv.step")

    print("Selecting fair Best Static only from the independent training split...")
    fair_static_action, fair_static_reward, fair_static_metadata = (
        choose_train_selected_best_static(
            immediate_model,
            args.train_workload,
            args.train_cloud_trace,
            args.train_edge_trace,
            args.static_train_samples,
        )
    )
    print(
        f"Best Static training reward={fair_static_reward:.3f}, "
        f"action={fair_static_action.tolist()}"
    )

    bandwidths = scenario_bandwidths(evaluator, args.steps)
    records = []
    selected_static = {
        "Best_Static": {
            "action": fair_static_action.tolist(),
            "training_selection_reward": fair_static_reward,
            **fair_static_metadata,
        },
        "Scenario_Optimal_Static_Oracle": {},
    }
    for scenario_name, _, load_multiplier, start_index in evaluator.scenarios:
        print(f"Evaluating {scenario_name}...")
        evaluator.set_scenario(bandwidths[scenario_name], load_multiplier)
        oracle_static_action, oracle_static_reward = choose_scenario_optimal_static(
            evaluator, immediate_model, args.steps, start_index
        )
        selected_static["Scenario_Optimal_Static_Oracle"][scenario_name] = {
            "action": oracle_static_action.tolist(),
            "test_window_selection_reward": oracle_static_reward,
        }
        for strategy in STRATEGIES:
            static_action = None
            if strategy == "Best_Static":
                static_action = fair_static_action
            elif strategy == "Scenario_Optimal_Static_Oracle":
                static_action = oracle_static_action
            scenario_records = run_strategy(
                evaluator,
                immediate_model,
                strategy,
                args.steps,
                start_index,
                static_action=static_action,
            )
            for record in scenario_records:
                record["Scenario"] = scenario_name
            records.extend(scenario_records)

    raw = pd.DataFrame(records)
    summary = (
        raw.groupby(["Scenario", "Strategy"], as_index=False)
        .agg(
            VMAF=("VMAF", "mean"),
            Latency=("Latency", "mean"),
            Energy_mJ=("Energy_mJ", "mean"),
            Reward=("Reward", "mean"),
            Deadline_Rate=("Deadline_Met", "mean"),
            Decision_ms=("Decision_ms", "mean"),
        )
        .sort_values(["Scenario", "Reward"], ascending=[True, False])
    )
    raw.to_csv(output_dir / "strong_baselines_raw.csv", index=False)
    summary.to_csv(output_dir / "strong_baselines_summary.csv", index=False)
    with open(output_dir / "best_static_actions.json", "w", encoding="utf-8") as handle:
        json.dump(selected_static, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "README.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "Best Static is selected once from equally spaced samples in the independent "
            "training split and reused unchanged in all held-out test scenarios. "
            "Scenario-Optimal Static Oracle sees each complete test window and is therefore "
            "an optimistic static upper bound, not a deployable baseline. One-step Greedy "
            "exhaustively selects the highest "
            "immediate composite reward from all unique physical configurations at every step. "
            "Neither baseline decision time is included in rendering E2E latency.\n"
        )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
