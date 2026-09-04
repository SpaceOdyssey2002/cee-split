import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import joblib
import onnxruntime as ort
from collections import deque
from pathlib import Path

from scheduling_config import (
    ACTION_DIM,
    ACTION_NVECS,
    ACTION_STRIDE,
    NUM_LAYERS,
    RESOLUTION_OFFSET,
    RATE_OFFSET,
    RESOLUTION_PIXELS,
    RESOLUTION_SCALES,
    TARGET_BITRATES_KBPS,
    VP9_QP_BY_RATE,
    decode_layer_action,
    layer_offset,
)


OBSERVATION_DIM = 36
OBSERVATION_NAMES = (
    "near_log_tris", "mid_log_tris", "far_log_tris",
    "cloud_bw", "cloud_rtt", "cloud_tx", "cloud_jitter", "cloud_valid",
    "edge_bw", "edge_rtt", "edge_tx", "edge_jitter", "edge_valid",
    "cloud_render", "cloud_encode", "cloud_decode",
    "edge_render", "edge_encode", "edge_decode", "local_render", "local_valid",
    "near_last_latency", "mid_last_latency", "far_last_latency",
    "near_last_vmaf", "mid_last_vmaf", "far_last_vmaf",
    "near_last_node", "near_last_resolution", "near_last_rate",
    "mid_last_node", "mid_last_resolution", "mid_last_rate",
    "far_last_node", "far_last_resolution", "far_last_rate",
)


def _nonnegative_float(value, default=0.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(parsed, 0.0) if np.isfinite(parsed) else float(default)


def build_live_observation(stats, layer_latencies, layer_vmafs, last_env_action):
    """Build the policy state from Unity telemetry using training-time ordering."""

    layer_latencies = np.asarray(layer_latencies, dtype=np.float32)
    layer_vmafs = np.asarray(layer_vmafs, dtype=np.float32)
    last_env_action = np.asarray(last_env_action, dtype=np.float32)
    if layer_latencies.shape != (3,) or layer_vmafs.shape != (3,):
        raise ValueError("layer_latencies and layer_vmafs must each contain three values")
    if last_env_action.shape != (ACTION_DIM,):
        raise ValueError(f"last_env_action must contain {ACTION_DIM} values")

    local_render = _nonnegative_float(stats.get("pr_current", 0.0))
    if local_render <= 0.0:
        local_render = _nonnegative_float(stats.get("pr", 0.0))

    cloud_valid = float(bool(stats.get("_cloud_valid", any(
        _nonnegative_float(stats.get(key, 0.0)) > 0.0
        for key in ("bw0", "r0", "t0", "c0r", "d0")
    ))))
    edge_valid = float(bool(stats.get("_edge_valid", any(
        _nonnegative_float(stats.get(key, 0.0)) > 0.0
        for key in ("bw1", "r1", "t1", "c1r", "d1")
    ))))
    local_valid = float(bool(stats.get("pr_valid", local_render > 0.0)))

    values = [
        np.log10(_nonnegative_float(stats.get("n", 0.0)) + 1.0),
        np.log10(_nonnegative_float(stats.get("m", 0.0)) + 1.0),
        np.log10(_nonnegative_float(stats.get("f", 0.0)) + 1.0),
        _nonnegative_float(stats.get("bw0", 0.0)) / 100.0 * cloud_valid,
        _nonnegative_float(stats.get("r0", 0.0)) / 200.0 * cloud_valid,
        _nonnegative_float(stats.get("t0", 0.0)) / 200.0 * cloud_valid,
        _nonnegative_float(stats.get("j0", 0.0)) / 100.0 * cloud_valid,
        cloud_valid,
        _nonnegative_float(stats.get("bw1", 0.0)) / 100.0 * edge_valid,
        _nonnegative_float(stats.get("r1", 0.0)) / 200.0 * edge_valid,
        _nonnegative_float(stats.get("t1", 0.0)) / 200.0 * edge_valid,
        _nonnegative_float(stats.get("j1", 0.0)) / 100.0 * edge_valid,
        edge_valid,
        _nonnegative_float(stats.get("c0r", 0.0)) / 200.0 * cloud_valid,
        _nonnegative_float(stats.get("c0e", 0.0)) / 50.0 * cloud_valid,
        _nonnegative_float(stats.get("d0", 0.0)) / 50.0 * cloud_valid,
        _nonnegative_float(stats.get("c1r", 0.0)) / 200.0 * edge_valid,
        _nonnegative_float(stats.get("c1e", 0.0)) / 50.0 * edge_valid,
        _nonnegative_float(stats.get("d1", 0.0)) / 50.0 * edge_valid,
        local_render / 200.0 * local_valid,
        local_valid,
        *(np.maximum(layer_latencies, 0.0) / 200.0),
        *(np.clip(layer_vmafs, 0.0, 100.0) / 100.0),
        last_env_action[0] / 2.0,
        last_env_action[1] / 4.0,
        last_env_action[2] / 4.0,
        last_env_action[3] / 2.0,
        last_env_action[4] / 4.0,
        last_env_action[5] / 4.0,
        last_env_action[6] / 2.0,
        last_env_action[7] / 4.0,
        last_env_action[8] / 4.0,
    ]
    obs = np.asarray(values, dtype=np.float32)
    if obs.shape != (OBSERVATION_DIM,):
        raise RuntimeError(f"Live observation shape mismatch: {obs.shape}")
    return obs


def generate_smart_data():
    if os.path.exists("train_trace.csv"): os.remove("train_trace.csv")
    print("🔄 生成智能训练数据（已移除空闲状态）...")

    total_steps = 200000
    data = []
    bw_list = []

    states = [100, 50, 20, 5]
    current_state_idx = 0
    steps_in_state = 0
    state_duration = 500

    for i in range(total_steps):
        # 1. 网络生成
        if steps_in_state > state_duration:
            if np.random.rand() > 0.5: current_state_idx = np.random.randint(0, 4)
            steps_in_state = 0
            state_duration = np.random.randint(200, 1000)
        real_bw = max(1.0, states[current_state_idx] + np.random.normal(0, states[current_state_idx] * 0.1))
        bw_list.append(real_bw)
        steps_in_state += 1

        # 2. 负载生成 (纯动态连续负载，无空闲帧)
        phase = np.sin(i / 1000.0)
        factor = (phase + 1) / 2.0
        base_tris = 15000 + factor * 2_500_000
        current_tris = max(1000, base_tris + np.random.normal(0, 50000))

        n = int(current_tris * 0.33)
        m = int(current_tris * 0.33)
        f = int(current_tris * 0.34)
        data.append([n, m, f, 0, 0, 0])

    np.savetxt("train_network.txt", bw_list, fmt='%.2f')
    pd.DataFrame(data, columns=['Near_Tris', 'Mid_Tris', 'Far_Tris', 'x', 'x', 'x']).to_csv("train_trace.csv",
                                                                                            index=False)
    print("✅ 数据生成完毕！")


class TraceLoader:
    def __init__(self, trace_path):
        self.bandwidths = [20.0] * 1000
        if os.path.exists(trace_path):
            try:
                self.bandwidths = np.loadtxt(trace_path)
            except:
                pass
        self.ptr = 0

    def get_next_bandwidth(self):
        bw = self.bandwidths[self.ptr]
        self.ptr = (self.ptr + 1) % len(self.bandwidths)
        return float(bw)

    def reset(self):
        self.ptr = 0


class HybridRenderEnv(gym.Env):
    SUPPORTED_PAYLOAD_MODELS = ('lookup', 'exponential')
    SUPPORTED_CONTENT_PROFILES = ('mean', 'garden', 'oasis', 'terminal')
    SUPPORTED_ABLATIONS = (
        'full',
        'no_barrier',
        'no_switch',
        'no_barrier_no_switch',
        'no_qoe',
        'no_metrics',
        'no_domain_randomization',
        'fixed_quality',
        'no_layer',
    )

    def __init__(
        self,
        trace_file,
        network_trace_path,
        is_eval=False,
        ablation_mode='full',
        edge_network_trace_path=None,
        payload_model='lookup',
        content_profile='mean',
    ):
        super().__init__()

        if ablation_mode not in self.SUPPORTED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation_mode={ablation_mode!r}. "
                f"Expected one of {self.SUPPORTED_ABLATIONS}."
            )
        if payload_model not in self.SUPPORTED_PAYLOAD_MODELS:
            raise ValueError(
                f"Unknown payload_model={payload_model!r}. "
                f"Expected one of {self.SUPPORTED_PAYLOAD_MODELS}."
            )
        if content_profile not in self.SUPPORTED_CONTENT_PROFILES:
            raise ValueError(
                f"Unknown content_profile={content_profile!r}. "
                f"Expected one of {self.SUPPORTED_CONTENT_PROFILES}."
            )

        # Rendering resolution and network quality are independent actions.
        self.RES_MAP = {index: int(value) for index, value in enumerate(RESOLUTION_PIXELS)}
        self.RES_SCALE_MAP = {
            index: float(value) for index, value in enumerate(RESOLUTION_SCALES)
        }
        self.QP_MAP = {index: int(value) for index, value in enumerate(VP9_QP_BY_RATE)}
        self.TARGET_BITRATE_KBPS = {
            index: float(value) for index, value in enumerate(TARGET_BITRATES_KBPS)
        }
        # Paper payload model:
        #   S = kappa * pixels * beta_ref * exp[-lambda * (QP - QP_ref)]
        # S is measured in Mbit/frame. The rate-controlled WebRTC lookup is
        # selected through payload_model='lookup'.
        self.PAYLOAD_BETA_REF = 2.3045533696200824e-6
        self.PAYLOAD_LAMBDA = 0.051338898676894715
        self.PAYLOAD_QP_REF = 20.0
        self.CONTENT_KAPPAS = {
            'garden': 1.0000,
            'oasis': 1.6807805756327179,
            'terminal': 0.7644505223335383,
        }
        self.CONTENT_KAPPAS['mean'] = float(np.mean(list(self.CONTENT_KAPPAS.values())))
        self.payload_model = payload_model
        self.content_profile = content_profile
        self.content_kappa = self.CONTENT_KAPPAS[content_profile]
        self.COMPOSITION_TIME_MS = 0.1

        # Calibrated rendering-time model:
        #   T_render(node) = (a_D(node) * triangles
        #                     + a_R(node) * sum(layer_pixels))
        #                    / capacity_factor(node)
        # Coefficients are in ms/triangle and ms/pixel. They share the same
        # triangle-to-pixel cost ratio because all three nodes use the same
        # rendering pipeline. Node-specific scaling captures compute capacity.
        # The coefficients reproduce the measured 1080p rendering times at
        # 558.7k triangles: End=60 ms, Edge=4 ms, and Cloud=2.5 ms.
        self.RENDER_COEFFICIENTS = {
            0: (2.6856645499891052e-5, 2.1699070292829317e-5),  # End
            1: (1.7904430333260702e-6, 1.4466046861886210e-6),  # Edge
            2: (1.1190268958287936e-6, 9.0412792886788810e-7),  # Cloud
        }

        # 🚀 修改点 6：引入 VMAF 景深权重 (近景，中景，远景)
        self.VMAF_WEIGHTS = [0.50, 0.35, 0.15]

        # 加载渲染 Trace
        if not os.path.exists(trace_file):
            self.workload_data = pd.DataFrame(
                np.random.randint(10000, 1000000, size=(1000, 3)),
                columns=['Near_Tris', 'Mid_Tris', 'Far_Tris']
            )
        else:
            self.workload_data = pd.read_csv(trace_file).dropna().reset_index(drop=True)
            if len(self.workload_data.columns) > 3:
                self.workload_data = self.workload_data.iloc[:, 0:3]
            self.workload_data.columns = ['Near_Tris', 'Mid_Tris', 'Far_Tris']

        self.total_rows = len(self.workload_data)
        self.cloud_trace_loader = TraceLoader(network_trace_path)
        self.edge_trace_loader = TraceLoader(edge_network_trace_path or network_trace_path)
        # Backward-compatible alias used by older evaluation utilities.
        self.trace_loader = self.cloud_trace_loader
        self.bw_history = {
            1: deque(maxlen=5),  # Edge in environment node numbering.
            2: deque(maxlen=5),  # Cloud in environment node numbering.
        }

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(ACTION_NVECS)

        self.is_eval = is_eval
        self.max_steps = 1000 if is_eval else 500
        self.data_ptr = 0
        self.elapsed_steps = 0

        self.last_latency = 30.0
        self.last_vmaf = 80.0
        self.last_act_vec = np.zeros(ACTION_DIM, dtype=int)
        self.last_layer_latencies = np.full(3, 30.0, dtype=np.float32)
        self.last_layer_vmafs = np.full(3, 80.0, dtype=np.float32)
        self.node_telemetry = self._default_node_telemetry()
        self.current_obs = None

        self.ablation_mode = ablation_mode
        self.eval_c_local = 1.0
        self.eval_c_edge = 1.0
        self.eval_c_cloud = 1.0
        self.eval_cloud_telemetry_valid = 1.0
        self.eval_edge_telemetry_valid = 1.0
        self.eval_local_telemetry_valid = 1.0
        self.eval_start_fractions = (0.0, 0.2, 0.4, 0.6, 0.8)
        self.eval_reset_index = 0

        proxy_dir = Path(__file__).resolve().parent / "QoE代理（新版）" / "vp9_proxy"
        self.proxy_model_path = str(proxy_dir / "qoe_proxy_model.onnx")
        self.scaler_path = str(proxy_dir / "scaler_x.pkl")
        proxy_files_exist = os.path.exists(self.proxy_model_path) and os.path.exists(self.scaler_path)
        self.use_proxy = proxy_files_exist and self.ablation_mode != 'no_qoe'

        if self.use_proxy:
            self.ort_session = ort.InferenceSession(self.proxy_model_path)
            scaler_x = joblib.load(self.scaler_path)
            self.scaler_mean = scaler_x.mean_.astype(np.float32)
            self.scaler_scale = scaler_x.scale_.astype(np.float32)
            self.remote_vmaf_table = self._build_monotonic_vmaf_table(local=False)
            self.local_vmaf_table = self._build_monotonic_vmaf_table(local=True)
            print("Loaded QoE proxy model and input scaler.")
        else:
            reason = "disabled by no_qoe" if self.ablation_mode == 'no_qoe' else "model files missing"
            print(f"QoE proxy unavailable ({reason}); using heuristic quality model.")

    @staticmethod
    def _default_node_telemetry():
        return {
            "cloud": {"bw": 20.0, "rtt": 40.0, "tx": 0.0, "jitter": 5.0,
                      "render": 0.0, "encode": 0.0, "decode": 0.0},
            "edge": {"bw": 20.0, "rtt": 25.0, "tx": 0.0, "jitter": 3.0,
                     "render": 0.0, "encode": 0.0, "decode": 0.0},
            "local": {"render": 0.0},
        }

    def _run_proxy_vmaf(self, res_idx, qp_value):
        x_in = np.array([self.RES_SCALE_MAP[res_idx], qp_value], dtype=np.float32)
        x_scaled = ((x_in - self.scaler_mean) / self.scaler_scale).reshape(1, 2)
        ort_inputs = {self.ort_session.get_inputs()[0].name: x_scaled}
        return float(self.ort_session.run(None, ort_inputs)[0][0][0])

    def _build_monotonic_vmaf_table(self, local):
        raw = np.empty((5, 5), dtype=np.float32)
        for res_idx in range(5):
            for rate_idx in range(5):
                qp_value = 20.0 if local else float(self.QP_MAP[rate_idx])
                raw[res_idx, rate_idx] = self._run_proxy_vmaf(res_idx, qp_value)

        # Enforce the physical ordering of the discrete controls while
        # preserving the proxy's fitted values as closely as possible.
        calibrated = np.maximum.accumulate(raw, axis=0)
        calibrated = np.maximum.accumulate(calibrated, axis=1)
        if local:
            calibrated = np.minimum(100.0, calibrated + 2.0)
        return np.clip(calibrated, 0.0, 100.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.is_eval:
            fraction = self.eval_start_fractions[
                self.eval_reset_index % len(self.eval_start_fractions)
            ]
            self.eval_reset_index += 1
            self.data_ptr = int(fraction * self.total_rows) % self.total_rows
            self.cloud_trace_loader.ptr = int(
                fraction * len(self.cloud_trace_loader.bandwidths)
            ) % len(self.cloud_trace_loader.bandwidths)
            self.edge_trace_loader.ptr = int(
                fraction * len(self.edge_trace_loader.bandwidths)
            ) % len(self.edge_trace_loader.bandwidths)
            self.local_capacity_factor = self.eval_c_local
            self.edge_capacity_factor = self.eval_c_edge
            self.cloud_capacity_factor = self.eval_c_cloud
        else:
            self.data_ptr = int(self.np_random.integers(0, self.total_rows))
            if self.ablation_mode == 'no_domain_randomization':
                self.local_capacity_factor = 1.0
                self.edge_capacity_factor = 1.0
                self.cloud_capacity_factor = 1.0
            else:
                self.local_capacity_factor = self.np_random.uniform(0.5, 1.5)
                self.edge_capacity_factor = self.np_random.uniform(0.7, 1.3)
                self.cloud_capacity_factor = self.np_random.uniform(0.8, 1.2)

        self.elapsed_steps = 0
        for history in self.bw_history.values():
            history.clear()
            history.extend([20.0] * 5)

        # Use a separate point of the same trace for Edge so policies observe
        # two correlated but non-identical links even with a single trace file.
        if not self.is_eval and len(self.edge_trace_loader.bandwidths) > 1:
            self.edge_trace_loader.ptr = int(
                self.np_random.integers(0, len(self.edge_trace_loader.bandwidths))
            )

        self.last_act_vec = np.zeros(ACTION_DIM, dtype=int)
        self.last_latency = 30.0
        self.last_vmaf = 80.0
        self.last_layer_latencies = np.full(3, 30.0, dtype=np.float32)
        self.last_layer_vmafs = np.full(3, 80.0, dtype=np.float32)
        self.node_telemetry = self._default_node_telemetry()

        self.current_obs = self._get_next_obs()
        return self.current_obs, {}

    def _predict_vmaf_and_payload(self, node, res_idx, rate_idx):
        if self.use_proxy:
            table = self.local_vmaf_table if node == 0 else self.remote_vmaf_table
            score = float(table[res_idx, rate_idx])
        else:
            base_scores = {0: 40, 1: 60, 2: 85, 3: 93, 4: 96}
            score = base_scores[res_idx]
            if node != 0:
                score -= (self.QP_MAP[rate_idx] - 20) * 1.5
            else:
                score += 2.0

            score = float(np.clip(score, 0, 100))

        if node == 0:
            payload_mbits = 0.0
        elif self.payload_model == 'lookup':
            payload_mbits = self.TARGET_BITRATE_KBPS[rate_idx] / (60.0 * 1000.0)
        else:
            pixels = self.RES_MAP[res_idx]
            qp = self.QP_MAP[rate_idx]
            payload_mbits = (
                self.content_kappa
                * pixels
                * self.PAYLOAD_BETA_REF
                * np.exp(-self.PAYLOAD_LAMBDA * (qp - self.PAYLOAD_QP_REF))
            )

        return score, float(payload_mbits)

    def _apply_ablation_to_action(self, action):
        applied = np.asarray(action, dtype=np.int64).copy()

        if self.ablation_mode == 'fixed_quality':
            for layer_index in range(NUM_LAYERS):
                offset = layer_offset(layer_index)
                applied[offset + RESOLUTION_OFFSET] = 2
                applied[offset + RATE_OFFSET] = 2
        elif self.ablation_mode == 'no_layer':
            applied[layer_offset(1)] = applied[layer_offset(0)]
            applied[layer_offset(2)] = applied[layer_offset(0)]

        # Local rendering does not traverse a video encoder. Canonicalizing its
        # unused network action avoids artificial switching penalties.
        for layer_index in range(NUM_LAYERS):
            offset = layer_offset(layer_index)
            if int(applied[offset]) == 0:
                applied[offset + RATE_OFFSET] = 4

        return applied

    def step(self, action):
        action = self._apply_ablation_to_action(action)
        safe_ptr = self.data_ptr % self.total_rows
        row = self.workload_data.iloc[safe_ptr]
        tris_list = [row['Near_Tris'], row['Mid_Tris'], row['Far_Tris']]
        layer_vmafs = self.last_layer_vmafs.astype(float).tolist()
        layer_latencies = self.last_layer_latencies.astype(float).tolist()
        active_layers = []
        node_tris = {0: 0.0, 1: 0.0, 2: 0.0}
        node_render_pixels = {0: 0.0, 1: 0.0, 2: 0.0}
        node_max_pixels = {0: 0.0, 1: 0.0, 2: 0.0}
        node_max_res_idx = {0: 0, 1: 0, 2: 0}
        node_max_rate_idx = {0: 0, 1: 0, 2: 0}
        node_payload_mbits = {0: 0.0, 1: 0.0, 2: 0.0}
        layer_nodes = {}

        total_weighted_vmaf = 0.0
        active_weight_sum = 0.0

        for i in range(3):
            tris = tris_list[i]
            if tris < 10:
                continue

            active_layers.append(i)
            node, res_idx, rate_idx = decode_layer_action(action, i)
            R = self.RES_MAP[res_idx]

            # Every layer is rendered at its requested resolution. The node
            # then scales the layer outputs to one shared packed canvas, whose
            # stream resolution and rate are the maximum assigned requests.
            node_tris[node] += tris
            node_render_pixels[node] += R
            node_max_pixels[node] = max(node_max_pixels[node], R)
            node_max_res_idx[node] = max(node_max_res_idx[node], res_idx)
            node_max_rate_idx[node] = max(node_max_rate_idx[node], rate_idx)
            layer_nodes[i] = node

        # A physical node emits one packed stream. Payload and codec costs use
        # the maximum canvas resolution and rate selected on that node.
        for node in (0, 1, 2):
            if node_tris[node] <= 0:
                continue
            _, node_payload_mbits[node] = self._predict_vmaf_and_payload(
                node, node_max_res_idx[node], node_max_rate_idx[node]
            )

        for i in active_layers:
            node = layer_nodes[i]
            _, res_idx, _ = decode_layer_action(action, i)
            vmaf_i, _ = self._predict_vmaf_and_payload(
                node, res_idx, node_max_rate_idx[node]
            )
            layer_vmafs[i] = vmaf_i

            total_weighted_vmaf += vmaf_i * self.VMAF_WEIGHTS[i]
            active_weight_sum += self.VMAF_WEIGHTS[i]

        capacity_factors = {
            0: self.local_capacity_factor,
            1: self.edge_capacity_factor,
            2: self.cloud_capacity_factor,
        }
        t_render = {
            node: (
                self.RENDER_COEFFICIENTS[node][0] * node_tris[node]
                + self.RENDER_COEFFICIENTS[node][1] * node_render_pixels[node]
            ) / max(capacity_factors[node], 1e-6)
            for node in (0, 1, 2)
        }
        t_encode = {0: 0.0, 1: 0.0, 2: 0.0}
        t_decode = {0: 0.0, 1: 0.0, 2: 0.0}
        t_transmit = {0: 0.0, 1: 0.0, 2: 0.0}
        node_finish = {0: t_render[0] if node_tris[0] > 0 else 0.0}

        for node, telemetry_name in ((1, "edge"), (2, "cloud")):
            if node_tris[node] <= 0:
                node_finish[node] = 0.0
                continue
            pixels = node_max_pixels[node]
            t_encode[node] = 0.5 + pixels / 2_000_000.0
            t_decode[node] = 1.0 + pixels / 2_000_000.0
            bandwidth = max(self.node_telemetry[telemetry_name]["bw"], 0.1)
            t_transmit[node] = min(node_payload_mbits[node] / bandwidth * 1000.0, 2000.0)
            one_way_propagation = self.node_telemetry[telemetry_name]["rtt"] / 2.0
            node_finish[node] = (
                t_render[node]
                + t_encode[node]
                + one_way_propagation
                + t_transmit[node]
                + t_decode[node]
            )

        for layer_index in active_layers:
            layer_node = int(action[layer_offset(layer_index)])
            layer_latencies[layer_index] = node_finish[layer_node]

        if active_weight_sum > 0:
            avg_vmaf = total_weighted_vmaf / active_weight_sum
            final_latency = (
                max(layer_latencies[i] for i in active_layers)
                + self.COMPOSITION_TIME_MS
            )
        else:
            avg_vmaf = self.last_vmaf
            final_latency = self.last_latency

        t_local_render = t_render[0]
        t_tx_active = t_transmit[1] + t_transmit[2]
        energy_mJ = (5.0 * t_local_render) + (1.5 * t_tx_active) + (1.0 * final_latency)

        self.last_vmaf = avg_vmaf
        self.last_latency = final_latency
        self.last_layer_vmafs = np.asarray(layer_vmafs, dtype=np.float32)
        self.last_layer_latencies = np.asarray(layer_latencies, dtype=np.float32)
        self.node_telemetry["local"]["render"] = t_render[0]
        for node, telemetry_name in ((1, "edge"), (2, "cloud")):
            self.node_telemetry[telemetry_name].update(
                render=t_render[node],
                encode=t_encode[node],
                decode=t_decode[node],
                tx=t_transmit[node],
            )

        # Reward balances perceptual quality, continuous latency, terminal
        # energy, and stability. The old reward barely differentiated 20 ms
        # from 90 ms and ignored energy, so full-resolution local rendering dominated.
        DEADLINE = 100.0
        r_qual = 0.15 * (avg_vmaf - 50.0)
        r_lat = -0.03 * final_latency
        r_energy = -0.004 * energy_mJ

        barrier_disabled = self.ablation_mode in ('no_barrier', 'no_barrier_no_switch')
        if final_latency > DEADLINE and not barrier_disabled:
            diff = final_latency - DEADLINE
            penalty = 2.0 + 0.08 * diff + 0.002 * (diff ** 2)
            r_lat -= penalty

        switch_disabled = self.ablation_mode in ('no_switch', 'no_barrier_no_switch')
        switch_pen = 0.0 if switch_disabled else (0.2 if np.any(action != self.last_act_vec) else 0.0)
        reward = np.clip(r_qual + r_lat + r_energy - switch_pen, -30.0, 15.0)

        self.last_act_vec = action.copy()
        self.data_ptr += 1
        self.elapsed_steps += 1

        truncated = self.elapsed_steps >= self.max_steps
        terminated = False
        self.current_obs = self._get_next_obs()

        return self.current_obs, reward, terminated, truncated, {
            "vmaf": avg_vmaf,
            "latency": final_latency,
            "energy_mJ": energy_mJ,
            "raw_reward": reward,
            "reward_quality": r_qual,
            "reward_latency": r_lat,
            "reward_energy": r_energy,
            "reward_switch": -switch_pen,
            "render_time_local": t_render[0],
            "render_time_edge": t_render[1],
            "render_time_cloud": t_render[2],
            "encode_time_edge": t_encode[1],
            "encode_time_cloud": t_encode[2],
            "decode_time_edge": t_decode[1],
            "decode_time_cloud": t_decode[2],
            "transmit_time_edge": t_transmit[1],
            "transmit_time_cloud": t_transmit[2],
            "payload_mbits_edge": node_payload_mbits[1],
            "payload_mbits_cloud": node_payload_mbits[2],
            "composition_time": self.COMPOSITION_TIME_MS,
            "payload_model": self.payload_model,
            "content_profile": self.content_profile,
            "content_kappa": self.content_kappa,
            "near_vmaf": layer_vmafs[0],
            "mid_vmaf": layer_vmafs[1],
            "far_vmaf": layer_vmafs[2],
            "near_latency": layer_latencies[0],
            "mid_latency": layer_latencies[1],
            "far_latency": layer_latencies[2],
            "near_node": action[0],
            "mid_node": action[3],
            "far_node": action[6],
            "near_resolution": action[1],
            "mid_resolution": action[4],
            "far_resolution": action[7],
            "near_rate": action[2],
            "mid_rate": action[5],
            "far_rate": action[8],
            # Rendering resolutions remain per-layer. Rates are node-level
            # effective values because co-located layers share one stream.
            "near_effective_resolution": int(action[1]),
            "mid_effective_resolution": int(action[4]),
            "far_effective_resolution": int(action[7]),
            "near_effective_rate": node_max_rate_idx.get(layer_nodes.get(0, -1), 0),
            "mid_effective_rate": node_max_rate_idx.get(layer_nodes.get(1, -1), 0),
            "far_effective_rate": node_max_rate_idx.get(layer_nodes.get(2, -1), 0),
            "applied_action": action.tolist(),
            "ablation_mode": self.ablation_mode
        }

    def _get_next_obs(self):
        safe_ptr = self.data_ptr % self.total_rows
        row = self.workload_data.iloc[safe_ptr]
        current_bandwidths = {
            1: self.edge_trace_loader.get_next_bandwidth(),
            2: self.cloud_trace_loader.get_next_bandwidth(),
        }
        for node, telemetry_name in ((1, "edge"), (2, "cloud")):
            bandwidth = current_bandwidths[node]
            self.bw_history[node].append(bandwidth)
            if node == 1:
                base_rtt = 8.0 if bandwidth >= 80 else (18.0 if bandwidth >= 20 else 45.0)
            else:
                base_rtt = 18.0 if bandwidth >= 80 else (35.0 if bandwidth >= 20 else 70.0)
            self.node_telemetry[telemetry_name]["bw"] = bandwidth
            self.node_telemetry[telemetry_name]["rtt"] = base_rtt
            self.node_telemetry[telemetry_name]["jitter"] = max(1.0, base_rtt * 0.12)

        cloud = self.node_telemetry["cloud"]
        edge = self.node_telemetry["edge"]
        local = self.node_telemetry["local"]

        if self.is_eval:
            cloud_valid = self.eval_cloud_telemetry_valid
            edge_valid = self.eval_edge_telemetry_valid
            local_valid = self.eval_local_telemetry_valid
        else:
            cloud_valid = float(self.np_random.random() >= 0.05)
            edge_valid = float(self.np_random.random() >= 0.05)
            local_valid = float(self.np_random.random() >= 0.03)

        runtime_metrics = np.array([
            cloud["bw"] / 100.0 * cloud_valid,
            cloud["rtt"] / 200.0 * cloud_valid,
            cloud["tx"] / 200.0 * cloud_valid,
            cloud["jitter"] / 100.0 * cloud_valid,
            cloud_valid,
            edge["bw"] / 100.0 * edge_valid,
            edge["rtt"] / 200.0 * edge_valid,
            edge["tx"] / 200.0 * edge_valid,
            edge["jitter"] / 100.0 * edge_valid,
            edge_valid,
            cloud["render"] / 200.0 * cloud_valid,
            cloud["encode"] / 50.0 * cloud_valid,
            cloud["decode"] / 50.0 * cloud_valid,
            edge["render"] / 200.0 * edge_valid,
            edge["encode"] / 50.0 * edge_valid,
            edge["decode"] / 50.0 * edge_valid,
            local["render"] / 200.0 * local_valid,
            local_valid,
            *(self.last_layer_latencies / 200.0),
            *(self.last_layer_vmafs / 100.0),
        ], dtype=np.float32)
        if self.ablation_mode == 'no_metrics':
            runtime_metrics[:] = 0.0

        normalized_action = np.array([
            self.last_act_vec[0] / 2.0,
            self.last_act_vec[1] / 4.0,
            self.last_act_vec[2] / 4.0,
            self.last_act_vec[3] / 2.0,
            self.last_act_vec[4] / 4.0,
            self.last_act_vec[5] / 4.0,
            self.last_act_vec[6] / 2.0,
            self.last_act_vec[7] / 4.0,
            self.last_act_vec[8] / 4.0,
        ], dtype=np.float32)

        obs = np.concatenate((np.array([
            np.log10(row['Near_Tris'] + 1),
            np.log10(row['Mid_Tris'] + 1),
            np.log10(row['Far_Tris'] + 1),
        ], dtype=np.float32), runtime_metrics, normalized_action))
        if obs.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"Observation shape mismatch: {obs.shape}")
        return obs
