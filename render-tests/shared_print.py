# shared_print.py
# 两个服务器脚本共用的打印工具
# 延迟统一在此计算，Unity/Monitor 侧不再计算预估延迟

import json
import numpy as np
from scheduling_config import (
    RESOLUTION_SCALES,
    TARGET_BITRATES_KBPS,
    VP9_QP_BY_RATE,
    decode_layer_action,
)

NODE_MAP  = {0: "CLOUD", 1: "EDGE ", 2: "LOCAL"}
NODE_ICON = {0: "☁️ ", 1: "🌐", 2: "📱"}
LAYER_NAMES = ["Near", "Mid ", "Far "]

SCALES = RESOLUTION_SCALES
BITRATES = TARGET_BITRATES_KBPS
QP_MAP = VP9_QP_BY_RATE


def _bw_tag(mbps):
    if mbps > 50: return "✅ 优秀"
    if mbps > 20: return "⚠️  良好"
    return "❌ 较差"

def _rtt_tag(ms, warn=30, bad=80):
    if ms < warn: return "✅"
    if ms < bad:  return "⚠️"
    return "❌"


# ──────────────────────────────────────────────────────────────
# 1. Unity 原始指标（纯展示，不计算延迟）
# ──────────────────────────────────────────────────────────────
def print_unity_state(stats):
    n  = stats.get('n', 0)
    m  = stats.get('m', 0)
    f  = stats.get('f', 0)
    pr = stats.get('pr', 0)
    pr_current = stats.get('pr_current', 0)
    pr_valid = bool(stats.get('pr_valid', 0))
    local_active = bool(stats.get('local_active', 0))
    pr_source = stats.get('pr_source', 'none')

    print("\n" + "=" * 70)
    print("📊 Unity 原始指标")
    print("=" * 70)

    print(f"\n🎨 场景复杂度:")
    print(f"   Near 物体数: {n}")
    print(f"   Mid  物体数: {m}")
    print(f"   Far  物体数: {f}")
    print(f"   总物体数:    {n + m + f}")

    print(f"\n☁️  Cloud 节点 (Node 0):")
    print(f"   可用带宽 bw0:  {stats.get('bw0', 0):.2f} Mbps  {_bw_tag(stats.get('bw0', 0))}")
    print(f"   RTT      r0:   {stats.get('r0',  0):.1f} ms    {_rtt_tag(stats.get('r0', 0))}")
    print(f"   单向传播 r0/2: {stats.get('r0',  0)/2:.1f} ms")
    print(f"   传输时延 t0:   {stats.get('t0',  0):.1f} ms")
    print(f"   PC渲染   c0r:  {stats.get('c0r', 0):.1f} ms")
    print(f"   PC编码   c0e:  {stats.get('c0e', 0):.1f} ms")
    print(f"   手机解码 d0:   {stats.get('d0',  0):.1f} ms")
    print(f"   Jitter   j0:   {stats.get('j0',  0):.1f} ms")

    print(f"\n🌐 Edge 节点 (Node 1):")
    print(f"   可用带宽 bw1:  {stats.get('bw1', 0):.2f} Mbps  {_bw_tag(stats.get('bw1', 0))}")
    print(f"   RTT      r1:   {stats.get('r1',  0):.1f} ms    {_rtt_tag(stats.get('r1', 0), warn=20, bad=50)}")
    print(f"   单向传播 r1/2: {stats.get('r1',  0)/2:.1f} ms")
    print(f"   传输时延 t1:   {stats.get('t1',  0):.1f} ms")
    print(f"   PC渲染   c1r:  {stats.get('c1r', 0):.1f} ms")
    print(f"   PC编码   c1e:  {stats.get('c1e', 0):.1f} ms")
    print(f"   手机解码 d1:   {stats.get('d1',  0):.1f} ms")
    print(f"   Jitter   j1:   {stats.get('j1',  0):.1f} ms")

    print(f"\n📱 Local 节点 (Node 2):")
    if pr_valid:
        source_label = "相机GPU" if pr_source == "camera-recorder" else "整帧GPU回退"
        print(f"   手机GPU估计 pr: {pr:.1f} ms  [{source_label}]")
        print(f"   当前本地GPU:    {pr_current:.1f} ms  " + ("正在渲染" if local_active else "纯云端，未渲染"))
        print(f"   算力系数:       {pr / 43.8:.2f}x")
    else:
        print("   手机GPU估计 pr: 尚无有效样本")
        print("   当前本地GPU:    0.0 ms")
        print("   算力系数:       尚未校准")


# ──────────────────────────────────────────────────────────────
# 2. 逐层决策（延迟在这里统一计算，带分解式）
#
#   端到端延迟公式：
#     Cloud/Edge: c_r + c_e + RTT/2 + t + d + pr
#     Local:      pr
#
#   其中：
#     c_r  = PC 渲染帧时间 (c0r / c1r)
#     c_e  = PC 编码耗时   (c0e / c1e)
#     RTT/2= 单向网络传播  (r0/2 / r1/2)
#     t    = 单帧传输时间  (t0 / t1)
#     d    = 手机解码耗时  (d0 / d1)
#     pr   = 手机本地渲染耗时
# ──────────────────────────────────────────────────────────────
def calc_latency(node, stats):
    """返回 (total_ms, breakdown_str)"""
    # 手机本地渲染 pr
    pr = stats.get('pr', 0)

    if node == 0:
        cr = stats.get('c0r', 0)
        ce = stats.get('c0e', 0)
        prop = stats.get('r0', 0) / 2.0  # 单向延迟
        t = stats.get('t0', 0)
        d = stats.get('d0', 0)
        # ✅ 修正：不再加 pr
        total = cr + ce + prop + t + d
        bd = f"c0r({cr:.0f})+c0e({ce:.0f})+r0/2({prop:.0f})+t0({t:.0f})+d0({d:.0f})"

    elif node == 1:
        cr = stats.get('c1r', 0)
        ce = stats.get('c1e', 0)
        prop = stats.get('r1', 0) / 2.0
        t = stats.get('t1', 0)
        d = stats.get('d1', 0)
        # ✅ 修正：不再加 pr
        total = cr + ce + prop + t + d
        bd = f"c1r({cr:.0f})+c1e({ce:.0f})+r1/2({prop:.0f})+t1({t:.0f})+d1({d:.0f})"

    else:  # Local Node 2
        total = pr
        bd = f"pr({pr:.0f})"

    return total, bd


def print_layer_decisions(action, stats, layer_vmafs, active_latencies,
                          ort_session, scaler_x, remote_vmaf_table=None,
                          local_vmaf_table=None):
    print("\n" + "=" * 70)
    print("🎮 逐层决策详情")
    print("=" * 70)

    requested = [decode_layer_action(action, i) for i in range(3)]
    effective_rate = {}
    for node, res_idx, rate_idx in requested:
        if node not in effective_rate:
            effective_rate[node] = rate_idx
        else:
            effective_rate[node] = max(effective_rate[node], rate_idx)

    for i, (node, requested_res_idx, requested_rate_idx) in enumerate(requested):
        # Resolution controls this layer's render target. Co-located layers
        # share the maximum requested target rate in the packed stream.
        res_idx = requested_res_idx
        rate_idx = effective_rate[node]
        res_scale = SCALES[res_idx]
        bitrate = BITRATES[rate_idx]
        qp_val = 20 if node == 2 else QP_MAP[rate_idx]

        # 延迟（统一公式，带分解）
        lat, breakdown = calc_latency(node, stats)
        active_latencies.append(lat)

        # Prefer the monotonic tables built by HybridRenderEnv so online
        # feedback is numerically identical to the training environment.
        vmaf_table = local_vmaf_table if node == 2 else remote_vmaf_table
        if vmaf_table is not None:
            vmaf_pred = float(vmaf_table[res_idx, rate_idx])
        else:
            import numpy as np
            proxy_input = np.array([[res_scale, qp_val]], dtype=np.float32)
            proxy_input_scaled = scaler_x.transform(proxy_input).astype(np.float32)
            vmaf_pred = float(ort_session.run(
                None, {ort_session.get_inputs()[0].name: proxy_input_scaled}
            )[0][0][0])
        layer_vmafs.append(vmaf_pred)

        node_label  = f"{NODE_ICON[node]} {NODE_MAP[node]}"
        quality_tag = ("🎯 超高" if rate_idx >= 4 else "✨ 高" if rate_idx >= 3
                       else "👍 中" if rate_idx >= 2 else "📉 低"
                       if rate_idx >= 1 else "💀 最低")

        print(f"\n  [{LAYER_NAMES[i].strip()}] 层:")
        print(f"    节点:    {node_label}  (Node {node})")
        if (requested_res_idx, requested_rate_idx) != (res_idx, rate_idx):
            print(
                f"    层请求:   分辨率档 {requested_res_idx} | 网络档 {requested_rate_idx} "
                f"(同节点共享流后提升)"
            )
        print(f"    分辨率档: {res_idx} ({res_scale:.2f}x)")
        print(f"    网络档:   {rate_idx} {quality_tag} | 码率: {bitrate:.0f} kbps | 代理 QP: {qp_val}")
        print(f"    VMAF:    {vmaf_pred:.2f}")
        print(f"    延迟:    {lat:.1f} ms  =  {breakdown}")


# ──────────────────────────────────────────────────────────────
# 3. 汇总
# ──────────────────────────────────────────────────────────────
def print_summary(layer_vmafs, active_latencies, state_bw, response_dict):
    avg_vmaf    = float(np.mean(layer_vmafs))
    max_latency = float(max(active_latencies))

    print("\n" + "=" * 70)
    print("✨ 汇总")
    print("=" * 70)
    print(f"   Avg VMAF:    {avg_vmaf:.2f}")
    print(f"   Max Latency: {max_latency:.1f} ms")
    print(f"   Min BW:      {state_bw:.2f} Mbps")
    print(f"\n📤 回传 Unity JSON:")
    print(f"   {json.dumps(response_dict)}")
    print("=" * 70)

    return avg_vmaf, max_latency
