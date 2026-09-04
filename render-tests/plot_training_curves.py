"""Plot PPO training and validation convergence curves across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#7f8c8d", "#34495e"]


def discover_variants(model_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(path.name for path in model_root.iterdir() if path.is_dir())


def load_eval_seed(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "evaluations.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        timesteps = data["timesteps"].astype(np.int64)
        episode_rewards = data["results"].astype(float)
    return pd.DataFrame(
        {
            "timesteps": timesteps,
            "reward": np.mean(episode_rewards, axis=1),
            "episode_std": np.std(episode_rewards, axis=1),
        }
    )


def load_train_seed(run_dir: Path, smooth_episodes: int) -> pd.DataFrame | None:
    files = sorted((run_dir / "train_monitor").glob("*.monitor.csv"))
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path, skiprows=1)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if {"r", "l", "t"}.issubset(frame.columns):
            frames.append(frame[["r", "l", "t"]])
    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    merged["timesteps"] = merged["l"].cumsum()
    merged["reward"] = merged["r"].rolling(
        window=smooth_episodes, min_periods=max(2, smooth_episodes // 5)
    ).mean()
    return merged[["timesteps", "reward"]].dropna()


def aggregate_seed_curves(seed_curves: dict[int, pd.DataFrame], points: int = 300) -> pd.DataFrame | None:
    if not seed_curves:
        return None
    starts = [frame["timesteps"].min() for frame in seed_curves.values()]
    ends = [frame["timesteps"].max() for frame in seed_curves.values()]
    start = max(starts)
    end = min(ends)
    if end <= start:
        return None

    exact_steps = sorted(set.intersection(*[
        set(frame["timesteps"].astype(int)) for frame in seed_curves.values()
    ]))
    if len(exact_steps) >= 3:
        grid = np.asarray(exact_steps, dtype=float)
    else:
        grid = np.linspace(start, end, min(points, max(20, int(end - start) + 1)))

    interpolated = []
    for frame in seed_curves.values():
        clean = frame.groupby("timesteps", as_index=False)["reward"].mean().sort_values("timesteps")
        interpolated.append(np.interp(grid, clean["timesteps"], clean["reward"]))
    values = np.vstack(interpolated)
    mean = values.mean(axis=0)
    if len(values) > 1:
        ci95 = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    else:
        ci95 = np.zeros_like(mean)
    return pd.DataFrame(
        {"timesteps": grid, "mean": mean, "ci95": ci95, "n_seeds": len(values)}
    )


def collect_curves(
    model_root: Path,
    variants: list[str],
    loader,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    aggregated = {}
    raw_rows = []
    for variant in variants:
        seed_curves = {}
        for run_dir in sorted((model_root / variant).glob("seed_*")):
            try:
                seed = int(run_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            frame = loader(run_dir)
            if frame is None or frame.empty:
                continue
            seed_curves[seed] = frame
            copy = frame.copy()
            copy.insert(0, "seed", seed)
            copy.insert(0, "variant", variant)
            raw_rows.append(copy)
        curve = aggregate_seed_curves(seed_curves)
        if curve is not None:
            aggregated[variant] = curve

    raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    return aggregated, raw


def plot_curves(
    curves: dict[str, pd.DataFrame],
    ylabel: str,
    title: str,
    output_stem: Path,
) -> None:
    if not curves:
        print(f"Skipping {title}: no matching data")
        return

    fig, ax = plt.subplots(figsize=(9, 6.2))
    for index, (variant, frame) in enumerate(curves.items()):
        color = COLORS[index % len(COLORS)]
        x = frame["timesteps"].to_numpy() / 1_000_000.0
        mean = frame["mean"].to_numpy()
        ci95 = frame["ci95"].to_numpy()
        seeds = int(frame["n_seeds"].iloc[0])
        ax.plot(x, mean, color=color, linewidth=2.6, label=f"{variant} (n={seeds})")
        ax.fill_between(x, mean - ci95, mean + ci95, color=color, alpha=0.18)

    ax.set_xlabel("Environment Steps (Million)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, ncol=2 if len(curves) > 4 else 1)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_aggregate(curves: dict[str, pd.DataFrame], output: Path) -> None:
    rows = []
    for variant, frame in curves.items():
        copy = frame.copy()
        copy.insert(0, "variant", variant)
        rows.append(copy)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(output, index=False)


def generate_training_curves(
    model_root: Path,
    output_dir: Path,
    variants: list[str] | None = None,
    smooth_episodes: int = 50,
) -> None:
    model_root = Path(model_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = discover_variants(model_root, variants)

    eval_curves, eval_raw = collect_curves(model_root, variants, load_eval_seed)
    train_curves, train_raw = collect_curves(
        model_root,
        variants,
        lambda run_dir: load_train_seed(run_dir, smooth_episodes),
    )

    plot_curves(
        eval_curves,
        "Mean Evaluation Episode Reward",
        "Validation Convergence (Mean and 95% CI)",
        output_dir / "convergence_evaluation",
    )
    plot_curves(
        train_curves,
        "Smoothed Training Episode Reward",
        "Training Convergence (Mean and 95% CI)",
        output_dir / "convergence_training",
    )
    save_aggregate(eval_curves, output_dir / "convergence_evaluation_summary.csv")
    save_aggregate(train_curves, output_dir / "convergence_training_summary.csv")
    if not eval_raw.empty:
        eval_raw.to_csv(output_dir / "convergence_evaluation_raw.csv", index=False)
    if not train_raw.empty:
        train_raw.to_csv(output_dir / "convergence_training_raw.csv", index=False)

    metadata = {
        "model_root": str(model_root.resolve()),
        "variants": variants,
        "smooth_episodes": smooth_episodes,
        "evaluation_variants_plotted": list(eval_curves),
        "training_variants_plotted": list(train_curves),
    }
    (output_dir / "convergence_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Convergence curves written to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("ablation_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_curves"))
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--smooth-episodes", type=int, default=50)
    args = parser.parse_args()
    if not args.model_root.exists():
        parser.error(f"Model root does not exist: {args.model_root}")
    if args.smooth_episodes < 1:
        parser.error("--smooth-episodes must be positive")
    generate_training_curves(
        args.model_root,
        args.output_dir,
        variants=args.variants,
        smooth_episodes=args.smooth_episodes,
    )


if __name__ == "__main__":
    main()
