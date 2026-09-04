import socket
import json
import os
import argparse
import time
from collections import deque

import numpy as np
import joblib
import onnxruntime as ort

from shared_print import (
    print_unity_state,
    print_layer_decisions,
    print_summary,
)
from scheduling_config import (
    ACTION_STRIDE,
    RESOLUTION_SCALES,
    VP9_QP_BY_RATE,
    build_unity_response,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROXY_DIR = os.path.join(BASE_DIR, "QoE代理（新版）", "vp9_proxy")
ONNX_PATH = os.path.join(PROXY_DIR, "qoe_proxy_model.onnx")
SCALER_X_PATH = os.path.join(PROXY_DIR, "scaler_x.pkl")
SERVER_PORT   = 8080

NODE_MAP = {0: "Cloud", 1: "Edge", 2: "Local"}

# 手动决策刷新间隔，默认和 RL server 保持一
DECISION_INTERVAL = float(os.getenv("RL_INFERENCE_INTERVAL", "2.0"))


def parse_args():
    parser = argparse.ArgumentParser(description="Unity hybrid rendering manual decision server")
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
    args = parser.parse_args()

    if args.bandwidth <= 0:
        parser.error("--bandwidth must be greater than zero")
    if args.bandwidth_scale <= 0:
        parser.error("--bandwidth-scale must be greater than zero")

    return args


def get_valid_bandwidth(stats, key, fallback):
    try:
        value = float(stats.get(key, 0.0))
        if np.isfinite(value) and 0.1 <= value <= 1000.0:
            return value
    except (TypeError, ValueError):
        pass
    return fallback


def resolve_observed_bandwidths(stats, fallback_bandwidths, args):
    """
    Unity/WebRTC 原始定义：
        bw0 = Cloud link bandwidth
        bw1 = Edge link bandwidth

    bandwidth-mode:
        auto   : 直接使用 WebRTC 估计值
        fixed  : 手动指定 Cloud/Edge 带宽
        scaled : WebRTC 估计值乘缩放因子
    """
    raw = {
        0: get_valid_bandwidth(stats, "bw0", fallback_bandwidths[0]),
        1: get_valid_bandwidth(stats, "bw1", fallback_bandwidths[1]),
    }

    if args.bandwidth_mode == "fixed":
        observed = {
            0: args.bandwidth,
            1: args.bandwidth,
        }
    elif args.bandwidth_mode == "scaled":
        observed = {
            node: value * args.bandwidth_scale
            for node, value in raw.items()
        }
    else:
        observed = raw.copy()

    observed = {
        node: float(np.clip(value, 0.1, 1000.0))
        for node, value in observed.items()
    }

    return raw, observed


def build_policy_stats(stats, observed_bandwidths):
    """
    构造用于 VMAF/Latency 估计的 stats。

    核心作用：
    1. 保留 Unity 原始 stats；
    2. 替换 bw0/bw1 为手动校准后的带宽；
    3. 根据带宽变化缩放传输延迟 t0/t1；
    4. 标记 Cloud/Edge 链路是否有效。
    """
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

        # 如果原始带宽有效，则按带宽比例修正传输延迟
        # 带宽越大，传输延迟越小
        if raw_bw > 0.1 and raw_delay >= 0.0:
            policy_stats[delay_key] = raw_delay * raw_bw / observed_bandwidth

    return policy_stats


def get_manual_action():
    while True:
        print("\n" + "=" * 70)
        print("📝 请输入决策指令")
        print("=" * 70)
        print("格式: [NearNode NearRes NearRate] [MidNode MidRes MidRate] [FarNode FarRes FarRate]")
        print("  Node: 0=Cloud  1=Edge  2=Local")
        print("  Res: 0-4, Rate: 0-4")
        print("\n快捷: all0 / cloud / edge / local / mix")

        shortcuts = {
            "all0":  "0 0 0 0 0 0 0 0 0",
            "cloud": "0 4 4 0 4 4 0 4 4",
            "edge":  "1 3 3 1 3 3 1 3 3",
            "local": "2 2 2 2 2 2 2 2 2",
            "mix":   "2 4 4 1 2 2 0 1 1",
        }

        user_input = input("\n>> ").strip().lower()

        if user_input in shortcuts:
            user_input = shortcuts[user_input]

        parts = user_input.split()

        if len(parts) != 9:
            print("❌ 需要 9 个数字")
            continue

        try:
            vals = [int(p) for p in parts]
            valid = True

            for i in range(3):
                offset = i * ACTION_STRIDE
                node, res_idx, rate_idx = vals[offset:offset + ACTION_STRIDE]

                if node not in [0, 1, 2]:
                    print("❌ Node 必须是 0/1/2")
                    valid = False
                    break

                if res_idx not in range(5) or rate_idx not in range(5):
                    print("❌ Res 和 Rate 必须是 0-4")
                    valid = False
                    break

            if valid:
                return vals

        except ValueError:
            print("❌ 请输入数字")


def build_response_dict(action):
    return build_unity_response(action)


def build_monotonic_vmaf_tables(ort_session, scaler_x):
    """Reproduce the quality tables used by HybridRenderEnv."""
    remote = np.empty((5, 5), dtype=np.float32)
    local = np.empty((5, 5), dtype=np.float32)
    input_name = ort_session.get_inputs()[0].name
    for res_idx, scale in enumerate(RESOLUTION_SCALES):
        for rate_idx, qp in enumerate(VP9_QP_BY_RATE):
            remote_input = scaler_x.transform(
                np.array([[scale, qp]], dtype=np.float32)
            ).astype(np.float32)
            local_input = scaler_x.transform(
                np.array([[scale, 20]], dtype=np.float32)
            ).astype(np.float32)
            remote[res_idx, rate_idx] = ort_session.run(
                None, {input_name: remote_input}
            )[0][0][0]
            local[res_idx, rate_idx] = ort_session.run(
                None, {input_name: local_input}
            )[0][0][0]

    remote = np.maximum.accumulate(np.maximum.accumulate(remote, axis=0), axis=1)
    local = np.maximum.accumulate(np.maximum.accumulate(local, axis=0), axis=1)
    return np.clip(remote, 0.0, 100.0), np.clip(local + 2.0, 0.0, 100.0)


def main():
    args = parse_args()

    print("🔧 加载 ONNX 模型...")
    try:
        ort_session = ort.InferenceSession(ONNX_PATH)
        scaler_x = joblib.load(SCALER_X_PATH)
        remote_vmaf_table, local_vmaf_table = build_monotonic_vmaf_tables(
            ort_session, scaler_x
        )
        print("✅ ONNX 模型加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", SERVER_PORT))
    server_socket.listen(5)

    bw_desc = {
        "auto":   "WebRTC 自动估计",
        "fixed":  f"固定 {args.bandwidth:.1f} Mbps",
        "scaled": f"WebRTC × {args.bandwidth_scale:.2f}",
    }

    print(f"\n{'=' * 70}")
    print(f"  🚀 手动决策服务器  |  Port {SERVER_PORT}")
    print(f"  决策刷新间隔: {DECISION_INTERVAL}s  |  带宽模式: {bw_desc[args.bandwidth_mode]}")
    print(f"{'=' * 70}\n")

    frame_count = 0

    while True:
        conn, addr = server_socket.accept()
        print(f"\n📡 [NEW CONNECTION] {addr}")

        buffer = ""
        cache_frames = 0
        last_decision_time = 0.0
        cached_response = None
        cached_action = None

        bandwidth_history = {
            0: deque([20.0] * 5, maxlen=5),  # Cloud
            1: deque([20.0] * 5, maxlen=5),  # Edge
        }

        try:
            while True:
                data = conn.recv(8192)
                if not data:
                    break

                buffer += data.decode("utf-8")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue

                    frame_count += 1

                    try:
                        stats = json.loads(line)

                        # 1. 每帧更新带宽，保持平滑 fallback
                        fallback_bandwidths = {
                            node: history[-1]
                            for node, history in bandwidth_history.items()
                        }

                        raw_bandwidths, observed_bandwidths = resolve_observed_bandwidths(
                            stats,
                            fallback_bandwidths,
                            args,
                        )

                        policy_stats = build_policy_stats(stats, observed_bandwidths)

                        for node in (0, 1):
                            bandwidth_history[node].append(observed_bandwidths[node])

                        current_time = time.time()
                        time_since_decision = current_time - last_decision_time

                        # 2. 到达间隔才重新人工输入决策，否则复用缓存
                        if time_since_decision >= DECISION_INTERVAL or cached_response is None:
                            print(f"\n{'─' * 70}")
                            print(
                                f"📦 帧 #{frame_count}  [手动决策模式]  "
                                f"{time.strftime('%H:%M:%S')}  "
                                f"缓存帧={cache_frames}  "
                                f"间隔={time_since_decision:.1f}s"
                            )

                            bw_line = (
                                f"   带宽({args.bandwidth_mode}):  "
                                f"Cloud={observed_bandwidths[0]:.1f}  "
                                f"Edge={observed_bandwidths[1]:.1f} Mbps"
                            )

                            if args.bandwidth_mode == "scaled":
                                bw_line += (
                                    f"  (raw Cloud={raw_bandwidths[0]:.1f}"
                                    f"  Edge={raw_bandwidths[1]:.1f})"
                                )

                            print(bw_line)
                            print(f"{'─' * 70}")

                            cache_frames = 0

                            # 3. 打印 Unity 原始状态
                            print_unity_state(stats)

                            # 4. 人工输入 Unity 视角 action
                            # Unity 节点定义：0=Cloud, 1=Edge, 2=Local
                            action = get_manual_action()

                            # 5. 根据校准后的 policy_stats 估计各层 VMAF 和延迟
                            layer_vmafs, active_latencies = [], []

                            print_layer_decisions(
                                action,
                                policy_stats,
                                layer_vmafs,
                                active_latencies,
                                ort_session,
                                scaler_x,
                                remote_vmaf_table,
                                local_vmaf_table,
                            )

                            state_bw = min(observed_bandwidths.values())

                            # 6. 构造回传 JSON
                            response_dict = build_response_dict(action)

                            # 7. 汇总打印
                            print_summary(
                                layer_vmafs,
                                active_latencies,
                                state_bw,
                                response_dict,
                            )

                            if len(layer_vmafs) == 3:
                                weighted_vmaf = float(
                                    np.average(
                                        np.asarray(layer_vmafs, dtype=np.float32),
                                        weights=np.array([0.50, 0.35, 0.15]),
                                    )
                                )
                            else:
                                weighted_vmaf = 0.0

                            max_latency = (
                                float(max(active_latencies))
                                if active_latencies else 0.0
                            )

                            print(
                                f"  → 加权VMAF(50/35/15%): {weighted_vmaf:.1f}"
                                f"  |  最大延迟: {max_latency:.1f} ms"
                            )

                            cached_action = action
                            cached_response = response_dict
                            last_decision_time = current_time

                            print("\n✅ 新决策已生成并缓存")
                        else:
                            cache_frames += 1

                        # 8. 返回新决策或缓存决策
                        conn.sendall(
                            (json.dumps(cached_response) + "\n").encode("utf-8")
                        )

                    except json.JSONDecodeError as e:
                        print(f"❌ JSON 解析错误: {e}")
                    except Exception as e:
                        print(f"❌ 处理错误: {e}")
                        import traceback
                        traceback.print_exc()

        except ConnectionResetError:
            print("📴 客户端已断开")
        finally:
            conn.close()
            print("🔌 等待新连接...\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
