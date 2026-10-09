# Python 说明

所有命令在本目录执行。首次运行 Unity 系统，先看 [快速开始](../快速开始.md)。

| 文件 | 用途 |
|---|---|
| `manual_server.py` | 手动输入分层决策 |
| `rl_server.py` | 用已有 PPO 模型自动决策 |
| `run_unity_server.py` | 启动自动决策服务器，等同于运行 `rl_server.py` |
| `scheduling_config.py` | 节点、分辨率和码率定义 |
| `shared_print.py` | 状态和决策输出 |
| `hybrid_render_env.py` | 强化学习仿真环境 |
| `train_ablation.py` | 训练 PPO 和消融模型 |
| `evaluate_strong_baselines.py` | 对比 PPO 与基线策略 |
| `evaluate_ablation.py` | 消融评估 |
| `evalute.py` | 公共评估逻辑，其他脚本会导入它 |
| `fit_system_models.py` | 拟合渲染和编码参数 |
| `generate_training_data.py` | 生成训练数据 |
| `plot_training_curves.py` | 绘制训练曲线 |

## 手动调控

使用仓库根目录创建的虚拟环境：

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe manual_server.py --bandwidth-mode auto
```

服务器监听 TCP 8080。输入 `mix` 可运行混合分配，输入 `local` 可运行全部本地渲染。
每层按 `[节点 分辨率 码率]` 输入，共 9 个数字，例如 `2 4 4 1 2 2 0 1 1`。

## 自动调控、训练和评估

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
..\.venv\Scripts\python.exe rl_server.py --bandwidth-mode auto
```

手动和自动服务器使用同一个端口，运行其中一个即可。训练和评估参数查看：

```powershell
..\.venv\Scripts\python.exe train_ablation.py --help
..\.venv\Scripts\python.exe evaluate_strong_baselines.py --help
..\.venv\Scripts\python.exe evaluate_ablation.py --help
```

## 数据和模型

| 目录 | 内容 |
|---|---|
| `datasets/`、`网络数据集/` | 工作负载和网络轨迹 |
| `QoE代理（新版）/vp9_proxy/` | 服务器使用的 VMAF 模型和 scaler |
| `calibration/` | 实测校准数据 |
| `ablation_models_decoupled_vp9/full/seed_1/` | 自动服务器当前使用的模型 |
| `ablation_models_per_layer_canvas_v2/full/` | 论文使用的三随机种子模型 |
| `strong_baseline_results_per_layer_canvas_v2_1000/` | PPO 与基线对比结果 |
| `variant_five_condition_per_layer_canvas_v2/` | 五种条件下的 PPO 汇总 |
| `top_conf_eval_results_decoupled_vp9/` | 分层策略对比结果 |

保留目录结构和原有脚本文件名，避免破坏导入及模型路径。同一节点上的多个层保留各自渲染分辨率，合并到最大的画布后，共享最高目标码率的视频流。
