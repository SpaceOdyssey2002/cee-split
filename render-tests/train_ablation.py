"""Train a fair suite of PPO ablation models.

Every variant uses the same traces, PPO hyperparameters, vectorized training,
normalization, evaluation cadence, and random seeds. Only the requested module
is removed by HybridRenderEnv.ablation_mode.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from hybrid_render_env import OBSERVATION_DIM, OBSERVATION_NAMES, HybridRenderEnv
from scheduling_config import ACTION_NVECS, RESOLUTION_SIZES, TARGET_BITRATES_KBPS, VP9_QP_BY_RATE


DEFAULT_VARIANTS = list(HybridRenderEnv.SUPPORTED_ABLATIONS)


def configure_worker(torch_threads: int) -> None:
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def make_env(
    trace_file: str,
    network_trace: str,
    edge_network_trace: str,
    ablation_mode: str,
    seed: int,
    monitor_path: str,
    is_eval: bool = False,
    payload_model: str = "lookup",
    content_profile: str = "mean",
):
    def _init():
        env = HybridRenderEnv(
            trace_file,
            network_trace,
            is_eval=is_eval,
            ablation_mode=ablation_mode,
            edge_network_trace_path=edge_network_trace,
            payload_model=payload_model,
            content_profile=content_profile,
        )
        env.reset(seed=seed)
        return Monitor(env, monitor_path)

    return _init


class SaveBestVecNormalize(BaseCallback):
    """Save normalization statistics whenever EvalCallback finds a new best."""

    def __init__(self, save_path: Path):
        super().__init__(verbose=0)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(self.save_path))
        return True


def validate_args(args: argparse.Namespace) -> None:
    required_paths = (
        args.trace_file,
        args.network_trace,
        args.edge_network_trace,
        args.eval_trace_file,
        args.eval_network_trace,
        args.eval_edge_network_trace,
    )
    missing = [p for p in required_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing trace files: {', '.join(map(str, missing))}")

    unknown = sorted(set(args.variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid values: {DEFAULT_VARIANTS}")

    rollout_size = args.n_steps * args.n_envs
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must divide n_steps * n_envs "
            f"({args.n_steps} * {args.n_envs} = {rollout_size})."
        )


def train_one(args: argparse.Namespace, variant: str, seed: int) -> None:
    run_dir = args.output_dir / variant / f"seed_{seed}"
    best_model_path = run_dir / "best_model.zip"
    completion_marker = run_dir / "TRAINING_COMPLETE"
    if completion_marker.exists() and best_model_path.exists() and not args.overwrite:
        print(f"[skip] {variant} seed={seed}: training is complete")
        return
    resume_incomplete = (
        args.resume_incomplete
        and best_model_path.exists()
        and (run_dir / "vec_normalize.pkl").exists()
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    train_monitor_dir = run_dir / "train_monitor"
    eval_monitor_dir = run_dir / "eval_monitor"
    train_monitor_dir.mkdir(exist_ok=True)
    eval_monitor_dir.mkdir(exist_ok=True)

    env_fns = [
        make_env(
            str(args.trace_file),
            str(args.network_trace),
            str(args.edge_network_trace),
            variant,
            seed + rank,
            str(train_monitor_dir / f"env_{rank}"),
            payload_model=args.payload_model,
            content_profile=args.content_profile,
        )
        for rank in range(args.n_envs)
    ]
    if args.vec_backend == "dummy" or args.n_envs == 1:
        train_venv = DummyVecEnv(env_fns)
    else:
        train_venv = SubprocVecEnv(env_fns)
    if resume_incomplete:
        train_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), train_venv)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_venv = DummyVecEnv(
        [
            make_env(
                str(args.eval_trace_file),
                str(args.eval_network_trace),
                str(args.eval_edge_network_trace),
                variant,
                seed + 100_000,
                str(eval_monitor_dir / "eval"),
                is_eval=True,
                payload_model=args.payload_model,
                content_profile=args.content_profile,
            )
        ]
    )
    eval_env = VecNormalize(
        eval_venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    best_norm_callback = SaveBestVecNormalize(run_dir / "vec_normalize.pkl")
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=best_norm_callback,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    config = {
        "environment_revision": "36d_per_layer_render_shared_canvas_vp9_v2",
        "action_layout": "[node, resolution, rate] x 3 layers",
        "action_nvecs": ACTION_NVECS.tolist(),
        "resolution_sizes": [list(size) for size in RESOLUTION_SIZES],
        "target_bitrates_kbps": TARGET_BITRATES_KBPS.tolist(),
        "vp9_proxy_qp_by_rate": VP9_QP_BY_RATE.tolist(),
        "vmaf_proxy": "QoE代理（新版）/vp9_proxy/qoe_proxy_model.onnx",
        "render_model": "per_node_dual_coefficient",
        "payload_model": args.payload_model,
        "content_profile": args.content_profile,
        "payload_coefficients": {
            "beta_ref_mbit_per_pixel": train_venv.get_attr("PAYLOAD_BETA_REF", indices=[0])[0],
            "lambda": train_venv.get_attr("PAYLOAD_LAMBDA", indices=[0])[0],
            "qp_reference": train_venv.get_attr("PAYLOAD_QP_REF", indices=[0])[0],
            "content_kappa": train_venv.get_attr("content_kappa", indices=[0])[0],
        },
        "composition_time_ms": train_venv.get_attr("COMPOSITION_TIME_MS", indices=[0])[0],
        "render_coefficients_ms": {
            str(node): {
                "triangle": coefficients[0],
                "pixel": coefficients[1],
            }
            for node, coefficients in train_venv.get_attr(
                "RENDER_COEFFICIENTS", indices=[0]
            )[0].items()
        },
        "render_calibration_reference": {
            "triangles": 558700,
            "pixels": 1920 * 1080,
            "end_ms": 60.0,
            "edge_ms": 4.0,
            "cloud_ms": 2.5,
        },
        "variant": variant,
        "observation_dim": OBSERVATION_DIM,
        "observation_names": OBSERVATION_NAMES,
        "seed": seed,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "vec_backend": args.vec_backend,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "ent_coef": args.ent_coef,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "trace_file": str(args.trace_file.resolve()),
        "network_trace": str(args.network_trace.resolve()),
        "edge_network_trace": str(args.edge_network_trace.resolve()),
        "eval_trace_file": str(args.eval_trace_file.resolve()),
        "eval_network_trace": str(args.eval_network_trace.resolve()),
        "eval_edge_network_trace": str(args.eval_edge_network_trace.resolve()),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[train] variant={variant} seed={seed} timesteps={args.timesteps:,}")
    if resume_incomplete:
        model = PPO.load(str(best_model_path), env=train_env, device=args.device)
        remaining_timesteps = max(args.timesteps - int(model.num_timesteps), 0)
        print(
            f"[resume] {variant} seed={seed}: checkpoint={model.num_timesteps:,}, "
            f"remaining={remaining_timesteps:,}"
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            seed=seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        remaining_timesteps = args.timesteps

    try:
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=eval_callback,
                reset_num_timesteps=not resume_incomplete,
            )
        model.save(str(run_dir / "final_model"))
        train_env.save(str(run_dir / "final_vec_normalize.pkl"))

        # Very short smoke runs may finish before the first scheduled evaluation.
        if not best_model_path.exists():
            model.save(str(run_dir / "best_model"))
            train_env.save(str(run_dir / "vec_normalize.pkl"))

        config["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        completion_marker.write_text(config["finished_at"] + "\n", encoding="ascii")
    finally:
        eval_env.close()
        train_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", type=Path, default=Path("train_trace.csv"))
    parser.add_argument("--network-trace", type=Path, default=Path("train_network.txt"))
    parser.add_argument("--edge-network-trace", type=Path, default=None)
    parser.add_argument("--eval-trace-file", type=Path, default=None)
    parser.add_argument("--eval-network-trace", type=Path, default=None)
    parser.add_argument("--eval-edge-network-trace", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_models"))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-backend", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
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
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    if args.edge_network_trace is None:
        args.edge_network_trace = args.network_trace
    if args.eval_trace_file is None:
        args.eval_trace_file = args.trace_file
    if args.eval_network_trace is None:
        args.eval_network_trace = args.network_trace
    if args.eval_edge_network_trace is None:
        args.eval_edge_network_trace = args.edge_network_trace
    args.output_dir = args.output_dir.resolve()
    validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    runs = [(variant, seed) for variant in args.variants for seed in args.seeds]
    total = len(runs)
    print(f"Training {total} runs: {len(args.variants)} variants x {len(args.seeds)} seeds")
    if args.jobs == 1:
        configure_worker(args.torch_threads)
        for variant, seed in runs:
            train_one(args, variant, seed)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=configure_worker,
            initargs=(args.torch_threads,),
        ) as executor:
            futures = {
                executor.submit(train_one, args, variant, seed): (variant, seed)
                for variant, seed in runs
            }
            for future in concurrent.futures.as_completed(futures):
                variant, seed = futures[future]
                future.result()
                print(f"[complete] variant={variant} seed={seed}")

    from plot_training_curves import generate_training_curves

    generate_training_curves(
        args.output_dir,
        args.output_dir / "training_curves",
        variants=args.variants,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
