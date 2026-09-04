from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "figures" / "design" / "fig_drl_mdp_decoupled.eps"


# Match the Arial typography used by the original MDP artwork and the paper's
# other schematic figures instead of Matplotlib's default DejaVu Sans.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})


def rounded_box(ax, xy, width, height, face, edge, title, lines):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height - 0.045, title,
            ha="center", va="top", fontsize=10.2, fontweight="bold")
    ax.text(x + 0.025, y + height - 0.105, "\n".join(lines),
            ha="left", va="top", fontsize=7.7, linespacing=1.28)


def arrow(ax, start, end, color="#333333", connection="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.7, color=color, connectionstyle=connection,
    ))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, (0.04, 0.30), 0.28, 0.43, "#E7F2E2", "#5A9E52",
                "State Observation  $s_t$", [
                    "1. Scene complexity",
                    "   Near / Mid / Far geometry",
                    "2. Network telemetry",
                    "   bandwidth, RTT, transmit time, jitter",
                    "3. Rendering pipeline telemetry",
                    "   render, encode, decode, local GPU time",
                    "4. Historical feedback",
                    "   prior latency, expected VMAF, and action",
                    "5. Validity indicators",
                    "   network and pipeline sample validity",
                ])

    rounded_box(ax, (0.365, 0.30), 0.25, 0.43, "#F9E6E5", "#C9554D",
                "Reward Shaping  $R_t$", [
                    "1. QoE utility",
                    "   configuration-level expected quality",
                    "2. Latency penalty",
                    "   linear delay cost",
                    "3. Soft deadline barrier",
                    "   nonlinear excess-latency penalty",
                    "4. Switching penalty",
                    "   changes in any action component",
                ])

    rounded_box(ax, (0.66, 0.30), 0.30, 0.43, "#E5EFFA", "#4C82B8",
                "Decision Action  $a_t$", [
                    "1. Per-layer node selection",
                    "   End / Edge / Cloud",
                    "2. Per-layer rendering-resolution selection",
                    "   360p / 720p / 1080p / 1440p / 2160p",
                    "3. Per-layer target-rate request",
                    "   1 / 2.5 / 5 / 8 / 10 Mbit/s",
                    "4. Node-level stream aggregation",
                    "   maximum canvas and target rate per node",
                ])

    agent = Ellipse((0.50, 0.88), 0.20, 0.14,
                    facecolor="#FCE5A7", edgecolor="#C39120", linewidth=1.7)
    ax.add_patch(agent)
    ax.text(0.50, 0.895, "Agent", ha="center", va="center", fontsize=10.5)
    ax.text(0.50, 0.855, r"PPO Scheduler  $\pi_\theta$",
            ha="center", va="center", fontsize=9.2, fontweight="bold")

    env = FancyBboxPatch((0.30, 0.07), 0.40, 0.115,
                         boxstyle="round,pad=0.015,rounding_size=0.025",
                         facecolor="#E7DDF2", edgecolor="#8063A5", linewidth=1.6)
    ax.add_patch(env)
    ax.text(0.50, 0.127, "Environment", ha="center", va="center", fontsize=10.5)
    ax.text(0.50, 0.092, "Cloud-Edge-End Rendering System",
            ha="center", va="center", fontsize=8.5, fontweight="bold")

    arrow(ax, (0.18, 0.73), (0.42, 0.84), color="#333333", connection="arc3,rad=-0.12")
    arrow(ax, (0.49, 0.73), (0.49, 0.81), color="#C9554D")
    arrow(ax, (0.58, 0.86), (0.81, 0.73), color="#1769AA", connection="arc3,rad=-0.10")
    arrow(ax, (0.81, 0.30), (0.69, 0.13), color="#1769AA", connection="arc3,rad=0.14")
    arrow(ax, (0.30, 0.13), (0.18, 0.30), color="#333333", connection="arc3,rad=0.14")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUTPUT, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
