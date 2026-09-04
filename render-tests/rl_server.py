import socket
import json
import os
import argparse
import numpy as np
import joblib
import onnxruntime as ort
import time
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hybrid_render_env import (
    OBSERVATION_DIM,
    OBSERVATION_NAMES,
    HybridRenderEnv,
    build_live_observation,
)
from shared_print import (print_unity_state, print_layer_decisions,
                          print_summary, calc_latency)
from scheduling_config import (
    ACTION_DIM,
    ACTION_NVECS,
    ACTION_STRIDE,
    RATE_OFFSET,
    build_unity_response,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ablation_models_decoupled_vp9", "full", "seed_1", "best_model.zip")
VECNORM_PATH = os.path.join(BASE_DIR, "ablation_models_decoupled_vp9", "full", "seed_1", "vec_normalize.pkl")
PROXY_DIR = os.path.join(BASE_DIR, "QoE代理（新版）", "vp9_proxy")
ONNX_PATH = os.path.join(PROXY_DIR, "qoe_proxy_model.onnx")
SCALER_X_PATH = os.path.join(PROXY_DIR, "scaler_x.pkl")
SERVER_PORT = 8080

# 推理执行的时间间隔（秒）
INFERENCE_INTERVAL = float(os.getenv("RL_INFERENCE_INTERVAL", "2.0"))
# 设置 RL_DEBUG_OBS=1 可在每次推理时打印 36 维观测向量
DEBUG_OBS = os.getenv("RL_DEBUG_OBS", "0") == "1"

# ==========================================
# 🔄 节点映射字典
# 环境定义：0=Local, 1=Edge, 2=Cloud
# Unity定义：0=Cloud, 1=Edge, 2=Local
# ==========================================
ENV_TO_UNITY_NODE = {0: 2, 1: 1, 2: 0}
    

def parse_args():
    parser = argparse.ArgumentParser(description="Unity hybrid rendering RL server")
    parser.add_argument(
        "--bandwidth-mode",
        choices=("auto", "fixed", "scaled"),
        default=os.getenv("RL_BANDWIDTH_MODE", "auto"),
        help="auto=WebRTC estimate, fixed=manual Mbps, scaled=WebRTC estimate times a factor",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=float(os.getenv("RL_FIXED_BANDWIDTH_MBPS", "20.0")),
        help="Bandwidth in Mbps when bandwidth mode is fixed",
    )
    parser.add_argument(
        "--bandwidth-scale",
        type=float,
        default=float(os.getenv("RL_BANDWIDTH_SCALE", "1.0")),
        help="Multiplier when bandwidth mode is scaled",
    )
    parser.add_argument(
        "--bandwidth-window",
        type=int,
        default=int(os.getenv("RL_BANDWIDTH_WINDOW", "5")),
        help="Median filter window for valid WebRTC bandwidth samples",
    )
    parser.add_argument(
        "--bandwidth-stale-seconds",
        type=float,
        default=float(os.getenv("RL_BANDWIDTH_STALE_SECONDS", "5.0")),
        help="How long the last valid WebRTC bandwidth estimate may be reused",
    )
    args = parser.parse_args()
    if args.bandwidth <= 0:
        parser.error("--bandwidth must be greater than zero")
    if args.bandwidth_scale <= 0:
        parser.error("--bandwidth-scale must be greater than zero")
    if args.bandwidth_window <= 0:
        parser.error("--bandwidth-window must be greater than zero")
    if args.bandwidth_stale_seconds <= 0:
        parser.error("--bandwidth-stale-seconds must be greater than zero")
    return args


def get_bandwidth_sample(stats, key):
    """Return a capacity estimate only when Unity explicitly marks it valid."""
    valid_key = f"{key}_valid"
    if valid_key in stats:
        marker = stats.get(valid_key)
        if isinstance(marker, str):
            explicitly_valid = marker.strip().lower() in ("1", "true", "yes")
        else:
            explicitly_valid = bool(marker)
        if not explicitly_valid:
            return None

    try:
        value = float(stats.get(key, 0.0))
        if np.isfinite(value) and 0.1 <= value <= 1000.0:
            return value
    except (TypeError, ValueError):
        pass
    return None


def resolve_observed_bandwidths(
        stats, bandwidth_history, bandwidth_last_valid_at, args, now):
    samples = {
        0: get_bandwidth_sample(stats, "bw0"),
        1: get_bandwidth_sample(stats, "bw1"),
    }
    raw = {node: sample if sample is not None else 0.0
           for node, sample in samples.items()}
    base = {}
    status = {}

    for node, sample in samples.items():
        if sample is not None:
            bandwidth_history[node].append(sample)
            bandwidth_last_valid_at[node] = now

        last_valid_at = bandwidth_last_valid_at[node]
        estimate_is_recent = (
            last_valid_at is not None and
            now - last_valid_at <= args.bandwidth_stale_seconds and
            len(bandwidth_history[node]) > 0
        )
        if estimate_is_recent:
            base[node] = float(np.median(bandwidth_history[node]))
            status[node] = "live" if sample is not None else "held"
        else:
            # A transparent startup/stale fallback is safer than treating media
            # traffic demand (bytesReceived) as physical link capacity.
            base[node] = args.bandwidth
            status[node] = "fallback"

    if args.bandwidth_mode == "fixed":
        observed = {0: args.bandwidth, 1: args.bandwidth}
        status = {0: "fixed", 1: "fixed"}
    elif args.bandwidth_mode == "scaled":
        observed = {node: value * args.bandwidth_scale for node, value in base.items()}
        status = {node: f"{value}/scaled" for node, value in status.items()}
    else:
        observed = base.copy()

    observed = {
        node: float(np.clip(value, 0.1, 1000.0))
        for node, value in observed.items()
    }
    return raw, observed, status


def build_policy_stats(stats, observed_bandwidths):
    """Apply manual calibration independently to the Cloud and Edge links."""
    policy_stats = dict(stats)
    policy_stats["_cloud_valid"] = any(
        float(stats.get(key, 0.0) or 0.0) > 0.0
        for key in ("bw0", "r0", "t0", "c0r", "d0")
    )
    policy_stats["_edge_valid"] = any(
        float(stats.get(key, 0.0) or 0.0) > 0.0
        for key in ("bw1", "r1", "t1", "c1r", "d1")
    )
    for node in (0, 1):
        bw_key = f"bw{node}"
        delay_key = f"t{node}"
        try:
            raw_bw = float(stats.get(bw_key, 0.0))
            raw_delay = float(stats.get(delay_key, 0.0))
        except (TypeError, ValueError):
            raw_bw = 0.0
            raw_delay = 0.0

        observed_bandwidth = observed_bandwidths[node]
        policy_stats[bw_key] = observed_bandwidth
        if raw_bw > 0.1 and raw_delay >= 0.0:
            policy_stats[delay_key] = raw_delay * raw_bw / observed_bandwidth

    return policy_stats


def create_dummy_env():
    return HybridRenderEnv("train_trace.csv", "train_network.txt", is_eval=True)


def _print_obs_vector(obs):
    """Print the current 36-dimensional observation vector compactly."""
    print(f"\n  [观测向量  {OBSERVATION_DIM}维]")
    items = list(zip(OBSERVATION_NAMES, obs))
    half = (len(items) + 1) // 2
    for i in range(half):
        left = f"    {items[i][0]:<22} {items[i][1]:6.3f}"
        right = (f"    {items[i + half][0]:<22} {items[i + half][1]:6.3f}"
                 if i + half < len(items) else "")
        print(left + right)
    print()


def main():
    args = parse_args()

    print(f"加载 RL 模型 ({OBSERVATION_DIM}维观测空间)...")
    raw_env = DummyVecEnv([create_dummy_env])
    try:
        env = VecNormalize.load(VECNORM_PATH, raw_env)
        env.training = False
        env.norm_reward = False
        if env.obs_rms.mean.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"normalizer has {env.obs_rms.mean.shape[0]} inputs, "
                f"but the current state requires {OBSERVATION_DIM}; retrain the model"
            )
    except Exception as e:
        print(f"❌ 无法加载 {VECNORM_PATH}: {e}")
        return

    try:
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
        if model.observation_space.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"model expects {model.observation_space.shape}, "
                f"but the current state requires ({OBSERVATION_DIM},)"
            )
        if not np.array_equal(np.asarray(model.action_space.nvec), ACTION_NVECS):
            raise ValueError(
                f"model action space {model.action_space.nvec} is incompatible with "
                f"the decoupled action space {ACTION_NVECS}"
            )
    except Exception as e:
        print(f"❌ 无法加载兼容的 RL 模型 {MODEL_PATH}: {e}")
        return

    policy_env = raw_env.envs[0]
    remote_vmaf_table = policy_env.remote_vmaf_table
    local_vmaf_table = policy_env.local_vmaf_table

    try:
        ort_session = ort.InferenceSession(ONNX_PATH)
        scaler_x = joblib.load(SCALER_X_PATH)
    except Exception as e:
        print(f"❌ 加载 ONNX/Scaler 失败: {e}")
        return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', SERVER_PORT))
    server_socket.listen(5)

    bw_desc = {
        "auto":   "WebRTC 自动估计",
        "fixed":  f"固定 {args.bandwidth:.1f} Mbps",
        "scaled": f"WebRTC × {args.bandwidth_scale:.2f}",
    }
    print(f"\n{'='*70}")
    print(f"  🚀 RL 推理服务器  |  Port {SERVER_PORT}  |  OBS {OBSERVATION_DIM}维")
    print(f"  推理间隔: {INFERENCE_INTERVAL}s  |  带宽模式: {bw_desc[args.bandwidth_mode]}")
    if DEBUG_OBS:
        print(f"  [调试] RL_DEBUG_OBS=1  每次推理将打印观测向量")
    print(f"{'='*70}\n")

    while True:
        conn, addr = server_socket.accept()
        print(f"\n📡 [NEW CONNECTION] {addr}")
        buffer = ""
        frame_count = 0
        cache_frames = 0
        last_infer_time = 0.0
        cached_response = None
        last_env_action = np.zeros(ACTION_DIM, dtype=np.int64)
        last_unity_action = None
        last_layer_vmafs = np.full(3, 80.0, dtype=np.float32)
        last_layer_latencies = np.full(3, 30.0, dtype=np.float32)
        bandwidth_history = {
            0: deque(maxlen=args.bandwidth_window),
            1: deque(maxlen=args.bandwidth_window),
        }
        bandwidth_last_valid_at = {0: None, 1: None}

        try:
            while True:
                data = conn.recv(8192)
                if not data: break

                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip(): continue

                    frame_count += 1
                    try:
                        stats = json.loads(line)

                        # 1. 只接收明确有效的 BWE，并用中位数抑制瞬时尖峰。
                        sample_time = time.monotonic()
                        raw_bandwidths, observed_bandwidths, bandwidth_status = (
                            resolve_observed_bandwidths(
                                stats,
                                bandwidth_history,
                                bandwidth_last_valid_at,
                                args,
                                sample_time,
                            )
                        )
                        policy_stats = build_policy_stats(stats, observed_bandwidths)

                        # 2. 根据上一帧动作更新观测到的各层延迟
                        if last_unity_action is not None:
                            observed_latencies = []
                            for i in range(3):
                                previous_node = int(last_unity_action[i * ACTION_STRIDE])
                                node_valid = (
                                    policy_stats.get("_cloud_valid", False)
                                    if previous_node == 0 else
                                    policy_stats.get("_edge_valid", False)
                                    if previous_node == 1 else
                                    bool(policy_stats.get("pr_valid", 0))
                                )
                                observed_latencies.append(
                                    calc_latency(previous_node, policy_stats)[0]
                                    if node_valid else 0.0
                                )
                            last_layer_latencies = np.asarray(
                                observed_latencies, dtype=np.float32
                            )

                        current_time = time.time()
                        time_since_infer = current_time - last_infer_time

                        # 3. 判断是否触发新推理
                        if time_since_infer >= INFERENCE_INTERVAL or cached_response is None:
                            # ── 推理块标题 ──────────────────────────────────────
                            print(f"\n{'─'*70}")
                            print(
                                f"⚡ 推理  Frame={frame_count}  {time.strftime('%H:%M:%S')}"
                                f"  缓存帧={cache_frames}  间隔={time_since_infer:.1f}s"
                            )
                            bw_line = (
                                f"   带宽({args.bandwidth_mode}):  "
                                f"Cloud={observed_bandwidths[0]:.2f}"
                                f"[{bandwidth_status[0]}]  "
                                f"Edge={observed_bandwidths[1]:.2f}"
                                f"[{bandwidth_status[1]}] Mbps"
                            )
                            if args.bandwidth_mode == "scaled":
                                bw_line += (
                                    f"  (raw Cloud={raw_bandwidths[0]:.1f}"
                                    f"  Edge={raw_bandwidths[1]:.1f})"
                                )
                            print(bw_line)
                            print(
                                "   实际接收吞吐:  "
                                f"Cloud={float(stats.get('rx0', 0.0) or 0.0):.2f}  "
                                f"Edge={float(stats.get('rx1', 0.0) or 0.0):.2f} Mbps"
                            )
                            print(f"{'─'*70}")
                            cache_frames = 0

                            print_unity_state(stats)

                            obs = build_live_observation(
                                policy_stats,
                                last_layer_latencies,
                                last_layer_vmafs,
                                last_env_action,
                            )
                            if DEBUG_OBS:
                                _print_obs_vector(obs)

                            # 4. RL 推理
                            obs_norm = env.normalize_obs(np.expand_dims(obs, axis=0))
                            action, _ = model.predict(obs_norm, deterministic=True)
                            raw_action = action[0].copy()  # Env 视角 (0=Local,1=Edge,2=Cloud)

                            # Local rendering has no transmitted stream in the
                            # simulator. Match the training environment by
                            # canonicalizing its unused network-rate head.
                            for offset in range(0, ACTION_DIM, ACTION_STRIDE):
                                if int(raw_action[offset]) == 0:
                                    raw_action[offset + RATE_OFFSET] = 4

                            # Env 节点 → Unity 节点
                            unity_action = raw_action.copy()
                            for offset in range(0, ACTION_DIM, ACTION_STRIDE):
                                unity_action[offset] = ENV_TO_UNITY_NODE[int(raw_action[offset])]

                            # 5. 预测各层 VMAF 和延迟
                            layer_vmafs, active_latencies = [], []
                            print_layer_decisions(unity_action, policy_stats,
                                                  layer_vmafs, active_latencies,
                                                  ort_session, scaler_x,
                                                  remote_vmaf_table,
                                                  local_vmaf_table)

                            if len(layer_vmafs) == 3:
                                last_layer_vmafs = np.asarray(layer_vmafs, dtype=np.float32)
                            last_env_action = raw_action.copy()
                            last_unity_action = unity_action.copy()
                            weighted_vmaf = float(np.average(
                                last_layer_vmafs, weights=np.array([0.50, 0.35, 0.15])
                            ))
                            max_latency = float(max(active_latencies)) if active_latencies else 0.0

                            # 6. 构造回传 JSON (使用 Unity 节点编号)
                            response_dict = build_unity_response(unity_action)

                            print_summary(
                                layer_vmafs,
                                active_latencies,
                                min(observed_bandwidths.values()),
                                response_dict,
                            )
                            print(
                                f"  → 加权VMAF(50/35/15%): {weighted_vmaf:.1f}"
                                f"  |  最大延迟: {max_latency:.1f} ms"
                            )

                            cached_response = response_dict
                            last_infer_time = current_time
                        else:
                            cache_frames += 1

                        # 7. 统一返回决策（新推理或缓存）
                        conn.sendall((json.dumps(cached_response) + "\n").encode('utf-8'))

                    except Exception as e:
                        print(f"⚠️ 处理出错: {e}")
                        import traceback
                        traceback.print_exc()

        except ConnectionResetError:
            print("📴 客户端已断开")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
