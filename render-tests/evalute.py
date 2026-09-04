import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hybrid_render_env import OBSERVATION_DIM, HybridRenderEnv
from scheduling_config import ACTION_NVECS

# 全局调大字号，以适应单图独立输出后缩小排版
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'font.serif':['Times New Roman', 'DejaVu Serif', 'Arial'],
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.title_fontsize': 16,
    'axes.linewidth': 2.0,
    'axes.unicode_minus': False
})

COLORS = {
    "End_Only": "#7f8c8d", "Cloud_Only": "#3498db", "Heuristic": "#9b59b6", "DRL_Agent": "#e74c3c"
}
LINESTYLE = {"End_Only": "--", "Cloud_Only": "--", "Heuristic": ":", "DRL_Agent": "-"}
MARKER = {"End_Only": "o", "Cloud_Only": "v", "Heuristic": "D", "DRL_Agent": "*"}
STRATEGY_LABEL = {
    "End_Only": "End Only", "Cloud_Only": "Cloud Only",
    "Heuristic": "Heuristic Rule", "DRL_Agent": "LBAS-DRL (Ours)"
}

class Baselines:
    @staticmethod
    def end_only(): return np.array([0, 1, 4, 0, 1, 4, 0, 1, 4])

    @staticmethod
    def cloud_only(): return np.array([2, 4, 4, 2, 4, 4, 2, 4, 4])

    @staticmethod
    def heuristic(current_bw):
        if current_bw >= 80.0:
            return np.array([2, 4, 4, 2, 4, 4, 2, 4, 4])
        if current_bw >= 20.0:
            return np.array([0, 2, 4, 1, 3, 3, 2, 3, 3])
        return np.array([0, 1, 4, 0, 1, 4, 0, 2, 4])

class UnifiedEvaluator:
    def __init__(
        self,
        model_path,
        vec_norm_path,
        workload_path="datasets/eval_workload.csv",
        cloud_trace_path="datasets/eval_cloud_network.txt",
        edge_trace_path="datasets/eval_edge_network.txt",
        log_dir="top_conf_eval_results_merged",
        ablation_results_dir="ablation_results",
        payload_model="lookup",
        content_profile="mean",
        ablation_mode="full",
    ):
        self.log_dir = log_dir
        self.ablation_results_dir = ablation_results_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.workload_path = workload_path
        self.cloud_trace_path = cloud_trace_path
        self.edge_trace_path = edge_trace_path
        self.raw_env = HybridRenderEnv(
            workload_path,
            cloud_trace_path,
            is_eval=True,
            edge_network_trace_path=edge_trace_path,
            payload_model=payload_model,
            content_profile=content_profile,
            ablation_mode=ablation_mode,
        )
        self.venv = DummyVecEnv([lambda: self.raw_env])

        if not os.path.exists(vec_norm_path):
            raise FileNotFoundError(f"Missing VecNormalize statistics: {vec_norm_path}")
        self.env = VecNormalize.load(vec_norm_path, self.venv)
        self.env.training = False
        self.env.norm_reward = False
        if self.env.obs_rms.mean.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"Normalizer expects {self.env.obs_rms.mean.shape}, current environment uses "
                f"({OBSERVATION_DIM},). Do not use the old 10-D normalizer."
            )

        self.model = PPO.load(model_path, env=self.env, device='cpu')
        if self.model.observation_space.shape != (OBSERVATION_DIM,):
            raise ValueError(f"Model is not compatible with the {OBSERVATION_DIM}-D environment")
        if not np.array_equal(np.asarray(self.model.action_space.nvec), ACTION_NVECS):
            raise ValueError("Model does not use the decoupled node/resolution/rate action space")
        self.base_workload = self.raw_env.workload_data.copy()
        self.rng = np.random.default_rng(20260615)
        self.scenarios =[
            ("Light_Task", 100.0, 0.3, 0),
            ("Excellent_Net", 150.0, 0.8, 1),
            ("Fluctuating", 'dynamic', 1.0, 2),
            ("Congested_Net", 3.0, 1.5, 3),
            ("Heavy_Workload", 80.0, 2.5, 4),
        ]

    def set_scenario(self, bandwidth, load_multiplier=1.0):
        cloud = np.asarray(bandwidth, dtype=np.float32)
        edge = np.clip(np.roll(cloud, 7) * 1.15, 0.1, 1000.0)
        self.raw_env.cloud_trace_loader.bandwidths = cloud
        self.raw_env.edge_trace_loader.bandwidths = edge
        self.raw_env.workload_data = self.base_workload * load_multiplier
        self.raw_env.total_rows = len(self.raw_env.workload_data)

    def current_bottleneck_bandwidth(self):
        return min(
            self.raw_env.node_telemetry["cloud"]["bw"],
            self.raw_env.node_telemetry["edge"]["bw"],
        )

    def reset_env(self, start_index=0):
        self.raw_env.eval_reset_index = start_index
        return self.env.reset()

    def run_macro_evaluation(self, num_steps=150):
        print("Running Macro Evaluation...")
        all_data = []
        strategies =["DRL_Agent", "End_Only", "Cloud_Only", "Heuristic"]

        for sce_name, bw_config, load_mul, start_index in self.scenarios:
            if bw_config == 'dynamic':
                bw_seq = np.clip(
                    [22.0 + 24.0 * np.sin(i / 5.0) + self.rng.normal(0, 3)
                     for i in range(num_steps + 10)],
                    0.3,
                    200.0,
                )
            else:
                bw_seq = [bw_config] * (num_steps + 10)

            self.set_scenario(bw_seq, load_mul)

            for strategy in strategies:
                obs = self.reset_env(start_index)
                for i in range(num_steps):
                    current_bw = self.current_bottleneck_bandwidth()

                    if strategy == "DRL_Agent":
                        action, _ = self.model.predict(obs, deterministic=True)
                        action_to_step = action
                    elif strategy == "End_Only":
                        action_to_step = [Baselines.end_only()]
                    elif strategy == "Cloud_Only":
                        action_to_step =[Baselines.cloud_only()]
                    elif strategy == "Heuristic":
                        action_to_step = [Baselines.heuristic(current_bw)]

                    obs, _, _, infos = self.env.step(action_to_step)
                    info = infos[0]
                    all_data.append({
                        "Scenario": sce_name, "Strategy": strategy,
                        "VMAF": info.get('vmaf', 0), "Latency": info.get('latency', 0),
                        "Energy_mJ": info.get('energy_mJ', 0),
                        "Deadline_Met": 1 if info.get('latency', 0) <= 100 else 0
                    })
        self.raw_env.workload_data = self.base_workload.copy()
        self.raw_env.total_rows = len(self.raw_env.workload_data)

        df = pd.DataFrame(all_data)
        self.plot_comprehensive_evaluation(df)
        self.plot_latency_cdf(df)
        self.plot_pareto_frontier(df)
        self.plot_energy_analysis(df)
        return df

    def plot_comprehensive_evaluation(self, df):
        print("Plotting Comprehensive Evaluation (Split into 3)...")
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

        # === 1. 兼容检测 Energy_mJ 或 Energy 列 ===
        energy_col = None
        if "Energy_mJ" in df.columns:
            energy_col = "Energy_mJ"
        elif "Energy" in df.columns:
            energy_col = "Energy"

        agg_cols = ["Latency", "VMAF"]
        if energy_col:
            agg_cols.append(energy_col)

        # 聚合数据
        agg = df.groupby(["Scenario", "Strategy"]).agg({m: "mean" for m in agg_cols}).reset_index()

        scenarios = ["Light_Task", "Excellent_Net", "Fluctuating", "Congested_Net", "Heavy_Workload"]
        x = np.arange(len(scenarios))
        xtick_labels = [s.replace("_", "\n") for s in scenarios]

        # === 1a: Latency ===
        plt.figure(figsize=(8, 6.5))
        for s in STRATEGY_LABEL.keys():
            sub = agg[agg["Strategy"] == s].set_index("Scenario").reindex(scenarios)
            plt.plot(x, sub["Latency"], label=STRATEGY_LABEL[s], color=COLORS[s], linestyle=LINESTYLE[s],
                     marker=MARKER[s], linewidth=4.0 if s == "DRL_Agent" else 3.0,
                     markersize=14 if s == "DRL_Agent" else 10)
        plt.axhline(100, linestyle="-", color="black", linewidth=2.5, alpha=0.6, label="Deadline (100ms)")
        plt.yscale('log')
        plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.yticks([20, 50, 100, 200, 500])
        plt.ylabel("Latency (ms) [Log Scale]", fontweight='bold')
        plt.xlabel("Testing Scenarios", fontweight='bold')
        plt.xlabel("Testing Scenarios", fontweight='bold')
        plt.xticks(x, xtick_labels, fontweight='bold')
        plt.legend(loc="upper left", framealpha=0.9, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig1a_Latency.pdf"), bbox_inches="tight")
        plt.close()

        # === 1b: Quality ===
        plt.figure(figsize=(8, 6.5))
        for s in STRATEGY_LABEL.keys():
            sub = agg[agg["Strategy"] == s].set_index("Scenario").reindex(scenarios)
            plt.plot(x, sub["VMAF"], label=STRATEGY_LABEL[s], color=COLORS[s], linestyle=LINESTYLE[s],
                     marker=MARKER[s], linewidth=4.0 if s == "DRL_Agent" else 3.0,
                     markersize=14 if s == "DRL_Agent" else 10)
        plt.ylabel("Perceptual Quality (VMAF)", fontweight='bold')
        plt.xlabel("Testing Scenarios", fontweight='bold')
        # Keep the low-quality End Only baseline visible instead of clipping
        # its approximately 50.8 VMAF line below the plotting area.
        plt.ylim(45, 100.5)
        plt.xticks(x, xtick_labels, fontweight='bold')
        plt.legend(loc="lower right", framealpha=0.9, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig1b_Quality.pdf"), bbox_inches="tight")
        plt.close()

        # === 1c: Robustness ===
        plt.figure(figsize=(8, 6.5))
        df_plot = df.copy()
        df_plot['Strategy_Label'] = df_plot['Strategy'].map(STRATEGY_LABEL)
        order = [STRATEGY_LABEL[s] for s in STRATEGY_LABEL.keys()]
        sns.boxplot(data=df_plot, x="Strategy_Label", y="VMAF", order=order,
                    palette=[COLORS[s] for s in STRATEGY_LABEL.keys()], showfliers=False, width=0.6)
        plt.ylabel("VMAF Distribution", fontweight='bold')
        plt.xlabel("Scheduling Strategies", fontweight='bold')
        plt.xticks(range(len(order)), ["End", "Cloud", "Heuristic", "Ours"], fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig1c_Robustness.pdf"), bbox_inches="tight")
        plt.close()

        # === 2: 直接生成并保存符合论文格式的 LaTeX 表格 ===
        print("Generating Paper-Style LaTeX Table...")

        scenarios_mapping = {
            "Excellent_Net": "Excellent Network",
            "Light_Task": "Light Task",
            "Fluctuating": "Fluctuating Network",
            "Heavy_Workload": "Heavy Workload",
            "Congested_Net": "Congested Network"
        }
        # Keep the table order identical to the introduction and figure axes.
        scenario_order = ["Light_Task", "Excellent_Net", "Fluctuating", "Congested_Net", "Heavy_Workload"]
        strategy_keys = list(STRATEGY_LABEL.keys())
        num_strategies = len(strategy_keys)

        latex_lines = []
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"	\centering")
        latex_lines.append(r"	\caption{Quantitative Performance Summary Across Diverse Scenarios}")
        latex_lines.append(r"	\label{tab:quantitative}")
        latex_lines.append(r"	\resizebox{\linewidth}{!}{")
        latex_lines.append(r"		\begin{tabular}{llcccc}")
        latex_lines.append(r"			\toprule")
        latex_lines.append(r"			\textbf{Test Scenario} & \textbf{Scheduling Strategy} &")
        latex_lines.append(r"			\textbf{Avg VMAF} & \textbf{Avg Latency (ms)} &")
        latex_lines.append(r"			\textbf{Energy (mJ)} \\")
        latex_lines.append(r"			\midrule")

        for idx, scenario_key in enumerate(scenario_order):
            scenario_name = scenarios_mapping[scenario_key]
            sub_df = agg[agg["Scenario"] == scenario_key]

            for s_idx, strat_key in enumerate(strategy_keys):
                strat_label = STRATEGY_LABEL.get(strat_key, strat_key)
                row_data = sub_df[sub_df["Strategy"] == strat_key]

                if not row_data.empty:
                    vmaf = f"{row_data['VMAF'].values[0]:.1f}"
                    latency = f"{row_data['Latency'].values[0]:.1f}"
                    energy = f"{row_data[energy_col].values[0]:.1f}" if energy_col else "0.0"
                else:
                    vmaf, latency, energy = "N/A", "N/A", "N/A"

                # 第一行使用 \multirow
                if s_idx == 0:
                    latex_lines.append(
                        f"			\\multirow{{{num_strategies}}}{{*}}{{\\textbf{{{scenario_name}}}}}")

                latex_lines.append(f"			& {strat_label:<14} & {vmaf:>4} & {latency:>5}  & {energy:>5} \\\\")

            # 在不同场景间插入 \midrule，最后一个场景后不加
            if idx < len(scenario_order) - 1:
                latex_lines.append(r"			\midrule")

        latex_lines.append(r"			\bottomrule")
        latex_lines.append(r"		\end{tabular}")
        latex_lines.append(r"	}")
        latex_lines.append(r"\end{table}")

        latex_content = "\n".join(latex_lines)
        tex_path = os.path.join(self.log_dir, "tab_quantitative.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        print(f"LaTeX table code saved to {tex_path}")
        print("\n--- Generated LaTeX Table Code ---")
        print(latex_content)
        print("----------------------------------\n")

    def plot_latency_cdf(self, df):
        print("Plotting Latency CDF...")
        plt.figure(figsize=(8, 6.5))
        for s in STRATEGY_LABEL.keys():
            sns.ecdfplot(data=df[df['Strategy'] == s]['Latency'], label=STRATEGY_LABEL[s], color=COLORS[s],
                         linestyle=LINESTYLE[s], linewidth=4.5 if s == "DRL_Agent" else 3.0)
        plt.axvline(100, color='black', linestyle='--', linewidth=3.0, alpha=0.7, label='Deadline (100ms)')
        plt.xlim(0, 300)
        plt.ylabel("Cumulative Probability (CDF)", fontweight='bold')
        plt.xlabel("End-to-End Latency (ms)", fontweight='bold')
        plt.legend(loc="lower right", frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig2_Latency_CDF.pdf"), bbox_inches='tight')
        plt.close()

    def plot_pareto_frontier(self, df):
        print("Plotting Pareto Frontier...")

        agg = df.groupby('Strategy').agg({
            'Latency': 'mean',
            'VMAF': 'mean'
        }).reset_index()

        plt.figure(figsize=(8, 6.5))

        # 所有标签统一放到 marker 左下方
        # dx < 0: 向左偏移
        # dy < 0: 向下偏移
        # ha='right': 文字主体向左展开
        # va='top': 文字主体向下展开
        label_pos = {
            "DRL_Agent": (-10, -8, "right", "top"),
            "Cloud_Only": (10, -8, "left", "top"),
            "Heuristic": (10, -8, "left", "top"),
            "End_Only": (10, -8, "left", "top"),
        }

        for _, row in agg.iterrows():
            strat = row['Strategy']

            # 缩小图中 marker，避免图例和图中符号过大
            scatter_size = 360 if strat == "DRL_Agent" else 260

            plt.scatter(
                row['Latency'],
                row['VMAF'],
                s=scatter_size,
                c=COLORS[strat],
                marker=MARKER[strat],
                label=STRATEGY_LABEL[strat],
                zorder=5,
                edgecolors='none'
            )

            dx, dy, ha, va = label_pos[strat]

            plt.annotate(
                STRATEGY_LABEL[strat],
                xy=(row['Latency'], row['VMAF']),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=14,
                fontweight='bold' if strat == "DRL_Agent" else 'normal',
                ha=ha,
                va=va,
                zorder=6
            )

        plt.axvline(
            100,
            color='black',
            linestyle='--',
            alpha=0.55,
            linewidth=2.5,
            label='Deadline (100ms)'
        )

        plt.axvspan(
            0,
            100,
            color='green',
            alpha=0.08,
            label='Safe Zone'
        )

        plt.ylabel(
            "Average VMAF \u2192 Higher is better",
            fontweight='bold'
        )

        plt.xlabel(
            "Average Latency (ms) \u2192 Lower is better",
            fontweight='bold'
        )

        # Include the End Only operating point (approximately 50.8 VMAF).
        plt.xlim(-5, 105)
        plt.ylim(45, 98)

        plt.legend(
            loc="lower right",
            framealpha=0.9,
            edgecolor='black',
            fontsize=11,
            markerscale=0.65,
            handlelength=1.4,
            handletextpad=0.5,
            labelspacing=0.35,
            borderpad=0.45,
            scatterpoints=1,
            shadow=False
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(self.log_dir, "Fig3_Pareto_Frontier.pdf"),
            bbox_inches='tight'
        )

        plt.close()

    def run_dynamic_response(self):
        print("Plotting Dynamic Response...")
        bw_trace = [150.0] * 30 +[3.0] * 40 + [100.0] * 50
        fixed_workload = pd.DataFrame(np.ones((200, 3)) * 180000, columns=['Near_Tris', 'Mid_Tris', 'Far_Tris'])
        self.raw_env.workload_data = fixed_workload
        self.raw_env.total_rows = len(fixed_workload)
        self.raw_env.cloud_trace_loader.bandwidths = np.asarray(bw_trace, dtype=np.float32)
        self.raw_env.edge_trace_loader.bandwidths = np.asarray(bw_trace, dtype=np.float32) * 1.15
        results = {"DRL_Agent": {"lat": [], "vmaf":[]}, "Heuristic": {"lat": [], "vmaf": []}}

        for strat in["DRL_Agent", "Heuristic"]:
            obs = self.reset_env(0)
            for i in range(120):
                if strat == "DRL_Agent":
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, _, infos = self.env.step(action)
                else:
                    _, _, _, infos = self.env.step([Baselines.heuristic(bw_trace[i])])
                results[strat]["lat"].append(infos[0]['latency'])
                results[strat]["vmaf"].append(infos[0]['vmaf'])
        self.raw_env.workload_data = self.base_workload.copy()
        self.raw_env.total_rows = len(self.raw_env.workload_data)

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(bw_trace, color='black', linewidth=3.5, label='Available Bandwidth')
        axes[0].axvspan(30, 70, color='red', alpha=0.1, label='Congestion')
        axes[0].set_ylabel('Bandwidth\n(Mbps)', fontweight='bold')
        axes[0].legend(loc='upper right')

        axes[1].plot(results["DRL_Agent"]["lat"], color=COLORS["DRL_Agent"], linewidth=4.0, label='Ours')
        axes[1].plot(results["Heuristic"]["lat"], color=COLORS["Heuristic"], linestyle='--', linewidth=3.0, label='Heuristic')
        axes[1].axhline(100, color='black', linestyle='--', alpha=0.8, linewidth=3.0, label='Deadline')
        axes[1].axvspan(30, 70, color='red', alpha=0.1)
        axes[1].set_ylabel('Latency\n(ms)', fontweight='bold')
        axes[1].set_ylim(bottom=0)
        axes[1].legend(loc='upper right')

        axes[2].plot(results["DRL_Agent"]["vmaf"], color=COLORS["DRL_Agent"], linewidth=4.0)
        axes[2].plot(results["Heuristic"]["vmaf"], color=COLORS["Heuristic"], linestyle='--', linewidth=3.0)
        axes[2].axvspan(30, 70, color='red', alpha=0.1)
        axes[2].set_ylabel('VMAF Score\n(Quality)', fontweight='bold')
        axes[2].set_xlabel("Frame Sequence", fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig4_Dynamic_Response.pdf"), bbox_inches='tight')
        plt.close()

    def plot_strategy_distribution(self):
        print("Plotting Strategy Distribution...")
        dist_scenarios = {
            "Light_Task": (100.0, 0.3), "Congested_Net": (3.0, 1.0), "Fluctuating": ('dynamic', 1.0),
            "Heavy_Workload": (80.0, 2.5), "Ultra_Workload": (300.0, 8.0)
        }
        display_names =["Light Task", "Congested\nNet", "Fluctuating\nNet", "Heavy\nLoad", "Ultra\nLoad"]
        results = {"End": [], "Edge": [], "Cloud":[]}
        for sce, (bw, load_mul) in dist_scenarios.items():
            if bw == 'dynamic':
                bw_seq = np.clip(
                    [22.0 + 24.0 * np.sin(i / 5.0) + self.rng.normal(0, 3)
                     for i in range(1000)],
                    0.3,
                    200.0,
                )
            else: bw_seq = [bw] * 1000
            self.set_scenario(bw_seq, load_mul)
            obs = self.reset_env(0)
            counts = {0: 0, 1: 0, 2: 0}
            for _ in range(250):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, _, _ = self.env.step(action)
                for n in [action[0][0], action[0][3], action[0][6]]: counts[n] += 1
            total = sum(counts.values())
            results["End"].append((counts[0] / total) * 100); results["Edge"].append((counts[1] / total) * 100); results["Cloud"].append((counts[2] / total) * 100)

        self.raw_env.workload_data = self.base_workload.copy()
        self.raw_env.total_rows = len(self.raw_env.workload_data)

        plt.figure(figsize=(9, 6.5))
        x = np.arange(len(display_names))
        b1 = plt.bar(x, results["End"], 0.55, color="#5DADE2", label="End", edgecolor='black', linewidth=2.0)
        b2 = plt.bar(x, results["Edge"], 0.55, bottom=results["End"], color="#58D68D", label="Edge", edgecolor='black', linewidth=2.0)
        b3 = plt.bar(x, results["Cloud"], 0.55, bottom=np.add(results["End"], results["Edge"]), color="#F1948A", label="Cloud", edgecolor='black', linewidth=2.0)

        for bars in[b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                if h > 4.0:
                    plt.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + h / 2),
                                 ha='center', va='center', color='black', fontweight='bold', fontsize=18)
        plt.ylabel("Task Distribution Ratio (%)", fontweight='bold')
        plt.xlabel("Operating Scenarios", fontweight='bold')
        plt.xticks(x, display_names, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig5_Strategy_Distribution.pdf"), bbox_inches='tight')
        plt.close()

    def run_overload_experiment(self, num_frames=100):
        print("Comparing fixed and dynamic Near/Mid/Far partitioning...")
        stale_imbalance_figure = os.path.join(self.log_dir, "Fig6c_Partition_Imbalance.pdf")
        if os.path.exists(stale_imbalance_figure):
            os.remove(stale_imbalance_figure)
        frames = np.arange(num_frames)
        rng = np.random.default_rng(20260616)
        object_count = 240
        object_positions = rng.uniform(0.0, 160.0, object_count)
        object_tris = np.clip(rng.lognormal(np.log(5000.0), 0.9, object_count), 300, 80_000)
        # A few large scene components make fixed distance bands visibly skewed.
        object_tris[rng.choice(object_count, 5, replace=False)] *= 6.0

        fixed_workload = np.zeros((num_frames, 3), dtype=float)
        dynamic_workload = np.zeros((num_frames, 3), dtype=float)
        fixed_near_distance = 25.0
        fixed_mid_distance = 65.0

        for frame in range(num_frames):
            camera_position = 55.0 + 42.0 * np.sin(frame / 17.0)
            distances = np.abs(object_positions - camera_position)

            fixed_layers = np.where(
                distances < fixed_near_distance,
                0,
                np.where(distances < fixed_mid_distance, 1, 2),
            )
            for layer in range(3):
                fixed_workload[frame, layer] = object_tris[fixed_layers == layer].sum()

            order = np.argsort(distances)
            cumulative = np.cumsum(object_tris[order])
            total = cumulative[-1]
            near_index = int(np.searchsorted(cumulative, total * 0.33))
            mid_index = int(np.searchsorted(cumulative, total * 0.66))
            dynamic_layers = np.full(object_count, 2, dtype=int)
            dynamic_layers[order[:near_index + 1]] = 0
            dynamic_layers[order[near_index + 1:mid_index + 1]] = 1
            for layer in range(3):
                dynamic_workload[frame, layer] = object_tris[dynamic_layers == layer].sum()

        # Isolate the partitioning effect with a fixed one-layer-per-node
        # assignment. All three nodes have the same geometry throughput, so a
        # latency difference comes only from partition imbalance. The frame
        # completes when the slowest node finishes.
        node_throughput_tris_per_ms = 12_000.0
        common_pipeline_latency_ms = 12.0
        fixed_latency = (
            common_pipeline_latency_ms
            + fixed_workload.max(axis=1) / node_throughput_tris_per_ms
        )
        dynamic_latency = (
            common_pipeline_latency_ms
            + dynamic_workload.max(axis=1) / node_throughput_tris_per_ms
        )
        fixed_imbalance = (
            fixed_workload.max(axis=1) - fixed_workload.min(axis=1)
        ) / np.maximum(fixed_workload.sum(axis=1), 1.0)
        dynamic_imbalance = (
            dynamic_workload.max(axis=1) - dynamic_workload.min(axis=1)
        ) / np.maximum(dynamic_workload.sum(axis=1), 1.0)
        pd.DataFrame({
            "frame": frames,
            "fixed_near_tris": fixed_workload[:, 0],
            "fixed_mid_tris": fixed_workload[:, 1],
            "fixed_far_tris": fixed_workload[:, 2],
            "dynamic_near_tris": dynamic_workload[:, 0],
            "dynamic_mid_tris": dynamic_workload[:, 1],
            "dynamic_far_tris": dynamic_workload[:, 2],
            "fixed_latency_ms": fixed_latency,
            "dynamic_latency_ms": dynamic_latency,
            "fixed_imbalance": fixed_imbalance,
            "dynamic_imbalance": dynamic_imbalance,
        }).to_csv(os.path.join(self.log_dir, "Fig6_partition_raw.csv"), index=False)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

        # === 6a: fixed distance thresholds ===
        plt.figure(figsize=(8, 6.5))
        plt.stackplot(frames, (fixed_workload / 1e6).T,
                      labels=['Near', 'Mid', 'Far'],
                      colors=['#e74c3c', '#f1c40f', '#3498db'], alpha=0.82)
        plt.xlabel("Frame Sequence", fontweight='bold')
        plt.ylabel("Layer Workload (Million Tris)", fontweight='bold')
        plt.title("Fixed Distance Partition", fontweight='bold')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig6a_Fixed_Depth.pdf"), bbox_inches="tight")
        plt.close()

        # === 6b: AutoAdaptiveTriCounter cumulative-triangle split ===
        plt.figure(figsize=(8, 6.5))
        plt.stackplot(frames, (dynamic_workload / 1e6).T,
                      labels=['Near', 'Mid', 'Far'],
                      colors=['#e74c3c', '#f1c40f', '#3498db'], alpha=0.82)
        plt.xlabel("Frame Sequence", fontweight='bold')
        plt.ylabel("Layer Workload (Million Tris)", fontweight='bold')
        plt.title("Dynamic Triangle-Balanced Partition", fontweight='bold')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig6b_LBAS.pdf"), bbox_inches="tight")
        plt.close()

        # === 6c: straggler latency under static layer-to-node assignment ===
        plt.figure(figsize=(8, 6.5))
        plt.plot(frames, fixed_latency, label='Fixed Partition', color='#7f8c8d', linestyle='--', linewidth=3.5)
        plt.plot(frames, dynamic_latency, label='Dynamic LBAS', color='#e74c3c', linestyle='-', linewidth=4.5)
        plt.axhline(100.0, color='black', linestyle=':', linewidth=2.8, label='Deadline (100 ms)')
        plt.xlabel("Frame Sequence", fontweight='bold')
        plt.ylabel("Frame Completion Latency (ms)", fontweight='bold')
        plt.ylim(0, max(140.0, fixed_latency.max() * 1.08))
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig6c_Overload_Latency.pdf"), bbox_inches="tight")
        plt.close()

    def run_layer_analysis_experiment(self, num_steps=100):
        print("Plotting Layer Analysis Experiment (Split into 2)...")
        bw_seq =[abs(60.0 + 20.0 * np.sin(i / 5.0)) for i in range(num_steps + 10)]
        self.set_scenario(bw_seq, 2.5)
        obs = self.reset_env(0)
        data =[]
        for i in range(num_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, infos = self.env.step(action)
            data.append({"Frame": i, "Near_VMAF": infos[0].get("near_vmaf", 80), "Mid_VMAF": infos[0].get("mid_vmaf", 80), "Far_VMAF": infos[0].get("far_vmaf", 80),
                         "Near_Node": infos[0].get("near_node", 0), "Mid_Node": infos[0].get("mid_node", 0), "Far_Node": infos[0].get("far_node", 0)})
        self.raw_env.workload_data = self.base_workload.copy()
        self.raw_env.total_rows = len(self.raw_env.workload_data)
        df = pd.DataFrame(data)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

        # === 7a: Layer VMAF ===
        plt.figure(figsize=(8, 6.5))
        means = [df["Near_VMAF"].mean(), df["Mid_VMAF"].mean(), df["Far_VMAF"].mean()]
        bars = plt.bar(["Near", "Mid", "Far"], means, color=["#e74c3c", "#f1c40f", "#3498db"], width=0.5, edgecolor='black', linewidth=2.0)
        plt.xlabel("Rendering Layer", fontweight='bold')
        plt.ylabel("Average VMAF Score", fontweight='bold')
        plt.ylim(0, 115)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig7a_Layer_VMAF.pdf"), bbox_inches="tight")
        plt.close()

        # === 7b: Layer Nodes ===
        plt.figure(figsize=(8, 6.5))
        plt.scatter(df["Frame"], df["Near_Node"], label="Near Layer", color="#e74c3c", s=150, alpha=0.7)
        plt.scatter(df["Frame"], df["Mid_Node"] + 0.1, label="Mid Layer", color="#f1c40f", s=150, alpha=0.7, marker="s")
        plt.scatter(df["Frame"], df["Far_Node"] + 0.2, label="Far Layer", color="#3498db", s=150, alpha=0.7, marker="^")
        plt.yticks([0, 1, 2], ["End (0)", "Edge (1)", "Cloud (2)"], fontweight='bold')
        plt.xlabel("Frame Sequence", fontweight='bold')
        plt.ylabel("Assigned Computing Node", fontweight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc="center right", frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig7b_Layer_Node.pdf"), bbox_inches="tight")
        plt.close()

    def plot_energy_analysis(self, df):
        print("Plotting Energy Analysis...")
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
        df_plot = df.copy()
        df_plot['Strategy_Label'] = df_plot['Strategy'].map(STRATEGY_LABEL)
        order = ["End Only", "Cloud Only", "Heuristic Rule", "LBAS-DRL (Ours)"]

        plt.figure(figsize=(9, 6.5))
        sns.barplot(data=df_plot, x="Scenario", y="Energy_mJ", hue="Strategy_Label", hue_order=order,
                    palette=["#7f8c8d", "#3498db", "#9b59b6", "#e74c3c"], edgecolor='black', capsize=0.05,
                    err_kws={'linewidth': 2.5})
        plt.ylabel("Terminal Energy (mJ/Frame)", fontweight='bold')
        plt.xlabel("Testing Scenarios", fontweight='bold')
        plt.legend(title="Strategy", loc='upper left', frameon=True, shadow=True)
        plt.xticks(range(len(df_plot['Scenario'].unique())), [s.replace("_", "\n") for s in df_plot['Scenario'].unique()], fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "Fig8_Energy_Analysis.pdf"), bbox_inches="tight")
        plt.close()

    def run_overhead_analysis(self):
        print("Running Overhead Analysis...")
        obs = self.reset_env(0)
        for _ in range(100):
            self.model.predict(obs, deterministic=True)
        times =[]
        for _ in range(5000):
            t0 = time.perf_counter()
            self.model.predict(obs, deterministic=True)
            times.append((time.perf_counter() - t0) * 1000)

        with open(os.path.join(self.log_dir, "Report_Overhead.txt"), "w") as f:
            f.write(
                f"Warm-up Decisions: 100\nTimed Decisions: 5000\n"
                f"DRL Inference Avg: {np.mean(times):.3f} ms\n"
                f"Max: {np.max(times):.3f} ms\n"
                f"99th: {np.percentile(times, 99):.3f} ms\n"
            )

    def experiment_ablation_and_domain(self):
        """Plot only measured ablation outputs produced by evaluate_ablation.py."""
        raw_path = os.path.join(self.ablation_results_dir, "ablation_raw.csv")
        if not os.path.exists(raw_path):
            for filename in (
                "Fig9a_Ablation_CDF.pdf",
                "Fig9b_Ablation_Switch.pdf",
                "Fig10_Domain_Randomization.pdf",
            ):
                stale_path = os.path.join(self.log_dir, filename)
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            print(f"Skipping Fig9/Fig10: missing {raw_path}. Run evaluate_ablation.py first.")
            return

        raw = pd.read_csv(raw_path)

        latency_data = raw[
            raw["variant"].isin(["full", "no_barrier"])
            & (raw["scenario"] == "deadline_stress")
        ]
        if set(latency_data["variant"]) == {"full", "no_barrier"}:
            plt.figure(figsize=(8, 6.5))
            sns.ecdfplot(data=latency_data[latency_data["variant"] == "no_barrier"]["latency_ms"],
                         color="#95a5a6", linewidth=4.0, linestyle="--", label="No Deadline Barrier")
            sns.ecdfplot(data=latency_data[latency_data["variant"] == "full"]["latency_ms"],
                         color="#e74c3c", linewidth=4.0, label="Full LBAS-DRL")
            plt.axvline(100, color='black', linestyle=':', linewidth=3.0)
            plt.xlabel("Episode Mean Latency (ms)", fontweight="bold")
            plt.ylabel("Cumulative Probability", fontweight="bold")
            plt.legend(frameon=True, shadow=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, "Fig9a_Ablation_CDF.pdf"), bbox_inches="tight")
            plt.close()
        else:
            print("Skipping Fig9a: full and no_barrier results are both required.")

        switch_data = raw[
            raw["variant"].isin(["full", "no_switch"])
            & raw["scenario"].isin(["fluctuating", "heavy_workload"])
        ]
        if set(switch_data["variant"]) == {"full", "no_switch"}:
            switch_data = switch_data.copy()
            switch_data["Variant"] = switch_data["variant"].map(
                {"full": "Full LBAS-DRL", "no_switch": "No Switch Penalty"}
            )
            switch_data["Scenario"] = switch_data["scenario"].map(
                {
                    "fluctuating": "Fluctuating Link",
                    "heavy_workload": "Heavy Workload",
                }
            )
            plt.figure(figsize=(8, 6.5))
            sns.barplot(data=switch_data, x="Scenario", y="switch_rate", hue="Variant",
                        palette=["#e74c3c", "#95a5a6"], errorbar="ci", capsize=0.08)
            plt.xlabel("Evaluation Scenario", fontweight="bold")
            plt.ylabel("Action Switch Rate", fontweight="bold")
            plt.legend(frameon=True, shadow=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, "Fig9b_Ablation_Switch.pdf"), bbox_inches="tight")
            plt.close()
        else:
            print("Skipping Fig9b: full and no_switch results are both required.")

        domain = raw[
            raw["variant"].isin(["full", "no_domain_randomization"])
            & raw["scenario"].isin(["nominal", "local_throttling", "edge_overload"])
        ].copy()
        if set(domain["variant"]) == {"full", "no_domain_randomization"}:
            domain["Variant"] = domain["variant"].map(
                {"full": "Domain Randomization", "no_domain_randomization": "No Domain Randomization"}
            )
            plt.figure(figsize=(9, 6.5))
            sns.barplot(data=domain, x="scenario", y="reward", hue="Variant",
                        palette=["#e74c3c", "#95a5a6"], errorbar="ci", capsize=0.08)
            plt.ylabel("Mean QoE Reward", fontweight="bold")
            plt.xlabel("System Degradation State", fontweight="bold")
            plt.xticks(rotation=15, ha="right")
            plt.legend(frameon=True, shadow=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, "Fig10_Domain_Randomization.pdf"), bbox_inches="tight")
            plt.close()
        else:
            print("Skipping Fig10: full and no_domain_randomization results are both required.")

def select_best_completed_run(model_root, variant="full"):
    candidates = []
    variant_dir = os.path.join(model_root, variant)
    if not os.path.isdir(variant_dir):
        raise FileNotFoundError(f"Missing trained variant directory: {variant_dir}")

    for seed_name in sorted(os.listdir(variant_dir)):
        run_dir = os.path.join(variant_dir, seed_name)
        model_path = os.path.join(run_dir, "best_model.zip")
        norm_path = os.path.join(run_dir, "vec_normalize.pkl")
        complete_path = os.path.join(run_dir, "TRAINING_COMPLETE")
        eval_path = os.path.join(run_dir, "evaluations.npz")
        if not all(os.path.exists(path) for path in (model_path, norm_path, complete_path)):
            continue
        score = -np.inf
        if os.path.exists(eval_path):
            with np.load(eval_path) as data:
                score = float(np.max(np.mean(data["results"], axis=1)))
        candidates.append((score, model_path, norm_path))

    if not candidates:
        raise RuntimeError(
            f"No completed {variant} runs found. Wait for TRAINING_COMPLETE before drawing final figures."
        )
    return max(candidates, key=lambda item: item[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures from the 36-D trained policy")
    parser.add_argument("--model-root", default="ablation_models")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vec-normalize", default=None)
    parser.add_argument("--workload", default="datasets/eval_workload.csv")
    parser.add_argument("--cloud-trace", default="datasets/eval_cloud_network.txt")
    parser.add_argument("--edge-trace", default="datasets/eval_edge_network.txt")
    parser.add_argument("--output-dir", default="top_conf_eval_results_merged")
    parser.add_argument("--ablation-results", default="ablation_results")
    parser.add_argument("--steps", type=int, default=1000)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if bool(args.model_path) != bool(args.vec_normalize):
        raise ValueError("--model-path and --vec-normalize must be supplied together")

    if args.model_path:
        model_path, vec_norm_path = args.model_path, args.vec_normalize
    else:
        score, model_path, vec_norm_path = select_best_completed_run(args.model_root)
        print(f"Selected completed full model with validation score {score:.2f}: {model_path}")

    evaluator = UnifiedEvaluator(
        model_path,
        vec_norm_path,
        workload_path=args.workload,
        cloud_trace_path=args.cloud_trace,
        edge_trace_path=args.edge_trace,
        log_dir=args.output_dir,
        ablation_results_dir=args.ablation_results,
        payload_model=args.payload_model,
        content_profile=args.content_profile,
    )
    results_df = evaluator.run_macro_evaluation(num_steps=args.steps)
    results_df.to_csv(os.path.join(args.output_dir, "macro_evaluation_raw.csv"), index=False)
    evaluator.run_dynamic_response()
    evaluator.plot_strategy_distribution()
    evaluator.run_overload_experiment(num_frames=100)
    evaluator.run_layer_analysis_experiment(num_steps=100)
    evaluator.run_overhead_analysis()
    evaluator.experiment_ablation_and_domain()
