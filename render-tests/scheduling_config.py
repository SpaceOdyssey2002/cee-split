"""Shared action layout and discrete streaming controls.

Each depth layer uses three independent decisions:
    [execution node, rendering resolution, network rate level]

The network rate level maps to both a WebRTC target bitrate (runtime) and
an offline VP9 QP/CRF value (VMAF proxy).  The two mappings share an index but
are not a claim that WebRTC exposes a fixed per-frame QP.
"""

from __future__ import annotations

import numpy as np


NUM_LAYERS = 3
ACTION_STRIDE = 3
ACTION_DIM = NUM_LAYERS * ACTION_STRIDE

NODE_OFFSET = 0
RESOLUTION_OFFSET = 1
RATE_OFFSET = 2

NODE_COUNT = 3
RESOLUTION_COUNT = 5
RATE_COUNT = 5
ACTION_NVECS = np.asarray(
    [NODE_COUNT, RESOLUTION_COUNT, RATE_COUNT] * NUM_LAYERS,
    dtype=np.int64,
)

RESOLUTION_SCALES = np.asarray([0.33, 0.67, 1.0, 1.33, 2.0], dtype=np.float32)
RESOLUTION_SIZES = (
    (640, 360),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)
RESOLUTION_PIXELS = np.asarray(
    [width * height for width, height in RESOLUTION_SIZES], dtype=np.int64
)

# Index 0 is the most bandwidth-saving network level; index 4 is the highest.
TARGET_BITRATES_KBPS = np.asarray(
    [1000.0, 2500.0, 5000.0, 8000.0, 10000.0], dtype=np.float32
)
VP9_QP_BY_RATE = np.asarray([40, 34, 28, 24, 20], dtype=np.int64)

LAYER_NAMES = ("near", "mid", "far")


def layer_offset(layer_index: int) -> int:
    if layer_index < 0 or layer_index >= NUM_LAYERS:
        raise IndexError(f"layer_index must be in [0, {NUM_LAYERS})")
    return layer_index * ACTION_STRIDE


def decode_layer_action(action, layer_index: int) -> tuple[int, int, int]:
    values = np.asarray(action, dtype=np.int64)
    if values.shape != (ACTION_DIM,):
        raise ValueError(f"Expected an action with shape ({ACTION_DIM},), got {values.shape}")
    offset = layer_offset(layer_index)
    return (
        int(values[offset + NODE_OFFSET]),
        int(values[offset + RESOLUTION_OFFSET]),
        int(values[offset + RATE_OFFSET]),
    )


def build_unity_response(action, node_map=None) -> dict:
    """Translate a nine-component action into Unity's per-layer JSON fields."""
    response = {}
    for layer_index, layer_name in enumerate(LAYER_NAMES):
        node, res_idx, rate_idx = decode_layer_action(action, layer_index)
        if node_map is not None:
            node = int(node_map[node])
        response[layer_name] = {
            "node": node,
            "scale": round(float(RESOLUTION_SCALES[res_idx]), 2),
            "bitrate": int(TARGET_BITRATES_KBPS[rate_idx]),
        }
    return response
