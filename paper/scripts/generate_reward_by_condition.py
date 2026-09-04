"""Generate the five-condition composite-reward comparison for Section 6.2."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "assets" / "data" / "reward_by_condition.csv"
OUTPUT_STEM = ROOT / "assets" / "figures" / "simulation" / "fig_reward_by_condition"

CONDITIONS = [
    "Light Task",
    "Excellent Network",
    "Fluctuating Network",
    "Congested Network",
    "Heavy Workload",
]
TICK_LABELS = [
    "Light\nTask",
    "Excellent\nNetwork",
    "Fluctuating\nNetwork",
    "Congested\nNetwork",
    "Heavy\nWorkload",
]
STYLES = {
    "End Only": dict(color="#7f8c8d", marker="o", linestyle=":"),
    "Cloud Only": dict(color="#3498db", marker="v", linestyle="--"),
    "Heuristic": dict(color="#8e44ad", marker="D", linestyle="--"),
    "Best Static": dict(color="#b7950b", marker="P", linestyle="-."),
    "Scenario Static Oracle": dict(color="#a65e2e", marker="X", linestyle=":"),
    "One-step Greedy Oracle": dict(color="#229954", marker="s", linestyle="-"),
    "LBAS-DRL": dict(color="#d62728", marker="*", linestyle="-"),
}


def main() -> None:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.15, 4.35))
    x = range(len(CONDITIONS))
    for strategy, style in STYLES.items():
        values = (
            data[data["Strategy"] == strategy]
            .set_index("Condition")
            .reindex(CONDITIONS)["Reward"]
            .to_numpy()
        )
        is_ours = strategy == "LBAS-DRL"
        ax.plot(
            x,
            values,
            label=strategy,
            linewidth=2.5 if is_ours else 1.55,
            markersize=9 if is_ours else 5.5,
            markeredgewidth=0.8,
            **style,
        )

    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.7)
    ax.set_xticks(list(x), TICK_LABELS)
    ax.set_ylabel("Composite reward (QoE utility)")
    ax.set_ylim(-11.0, 6.55)
    ax.set_yticks([-10, -8, -6, -4, -2, 0, 2, 4, 6])
    ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.18, top=0.79)

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
