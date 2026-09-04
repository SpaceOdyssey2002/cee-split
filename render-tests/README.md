# CEE-Split simulation and Unity control

The Python modules intentionally remain in this directory because the training,
evaluation, and Unity server entry points import their sibling modules directly.
Generated models and results are kept in separate named directories.

## Main entry points

- `hybrid_render_env.py`: trace-driven CEE rendering environment.
- `scheduling_config.py`: shared node, resolution, and target-rate definitions.
- `train_ablation.py`: PPO training entry point.
- `evalute.py`: common evaluator and baseline implementations.
- `evaluate_strong_baselines.py`: Best Static, scenario-static oracle, greedy
  oracle, and three-seed PPO comparison.
- `evaluate_ablation.py`: optional component evaluation.
- `rl_server.py`: Unity PPO inference server.
- `run_unity_server.py`: stable Unity server launcher.
- `manual_server.py`: manual-action Unity launcher for prototype debugging.
- `fit_system_models.py`: rendering and codec coefficient fitting.
- `generate_training_data.py`: workload and enhanced network-trace generation.
- `plot_training_curves.py`: convergence-curve generation.

## Canonical data and models

- `datasets/`: train/evaluation workload and network traces.
- `网络数据集/`: source mobile-network traces.
- `QoE代理（新版）/vp9_proxy/`: VP9 configuration-level VMAF proxy.
- `calibration/`: measured rendering and codec calibration data.
- `ablation_models_per_layer_canvas_v2/full/`: current three-seed paper policy.
- `ablation_models_decoupled_vp9/full/seed_1/`: policy currently loaded by
  `rl_server.py` for the physical prototype.

## Canonical evaluation outputs

- `strong_baseline_results_per_layer_canvas_v2_1000/`: five-condition strong
  baseline comparison used by the paper.
- `variant_five_condition_per_layer_canvas_v2/`: three-seed PPO condition
  summaries used by the paper.
- `top_conf_eval_results_decoupled_vp9/`: LBAS/fixed-partition component plots.

## Action convention

Each of the Near, Mid, and Far layers selects
`[render node, render resolution, target rate]`, giving a nine-component
`MultiDiscrete([3,5,5, 3,5,5, 3,5,5])` action. Layers retain independent
rendering resolutions. Co-located layers are scaled to the largest selected
node canvas and share the maximum requested WebRTC target rate.

## Common commands

```powershell
python run_unity_server.py
python manual_server.py
python train_ablation.py --help
python evaluate_strong_baselines.py --help
```
