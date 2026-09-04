"""Probe: invert the VMAF proxy's bitrate head to find equivalent QPs.

For every (resolution scale, target bitrate) pair used by the scheduler,
this script bisects the proxy's bitrate output to find the QP that would
produce that bitrate, then reports the predicted VMAF at that QP. It also
flags whether the inversion stays inside the proxy's trained QP grid
[20, 40] and prints the analytic (Equation 11) inverse for comparison.

Run from anywhere:
    python probe_equivalent_qp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

from scheduling_config import (  # noqa: E402
    RESOLUTION_PIXELS,
    RESOLUTION_SCALES,
    TARGET_BITRATES_KBPS,
    VP9_QP_BY_RATE,
)

PROXY_DIR = TEST_DIR / "QoE代理（新版）" / "vp9_proxy"
ONNX_PATH = PROXY_DIR / "qoe_proxy_model.onnx"
SCALER_PATH = PROXY_DIR / "scaler_x.pkl"

# Eq. 11 analytic coefficients from the calibration JSON (nominal-pixel units).
PAYLOAD_BETA_REF = 2.3045533696200824e-6
PAYLOAD_LAMBDA = 0.051338898676894715
PAYLOAD_QP_REF = 20.0
CONTENT_KAPPA_MEAN = float(np.mean([1.0, 1.6807805756327179, 0.7644505223335383]))

TRAINED_QP_MIN, TRAINED_QP_MAX = 20.0, 40.0
QP_SEARCH_MIN, QP_SEARCH_MAX = 0.0, 100.0


class ProxyInverter:
    """Bisect the proxy's bitrate output to invert (scale, bitrate) -> QP."""

    def __init__(self) -> None:
        self.session = ort.InferenceSession(str(ONNX_PATH))
        self.scaler = joblib.load(str(SCALER_PATH))
        self.input_name = self.session.get_inputs()[0].name
        # Sanity-check output order: [VMAF, bitrate_kbps].
        probe = self._predict(1.0, 20.0)
        print(
            "Proxy output at (scale=1.0, QP=20): "
            f"VMAF={probe[0]:.2f}, bitrate={probe[1]:.0f} kbps"
        )

    def _predict(self, scale: float, qp: float) -> np.ndarray:
        x = self.scaler.transform(
            np.array([[scale, qp]], dtype=np.float32)
        ).astype(np.float32)
        return self.session.run(None, {self.input_name: x})[0][0]

    def bitrate(self, scale: float, qp: float) -> float:
        return float(self._predict(scale, qp)[1])

    def vmaf(self, scale: float, qp: float) -> float:
        return float(self._predict(scale, qp)[0])

    def positive_bitrate_qp_ceiling(self, scale: float) -> float:
        """Largest QP (0.5 step) whose predicted bitrate is still positive."""
        qp = QP_SEARCH_MIN
        for _ in range(int((QP_SEARCH_MAX - QP_SEARCH_MIN) / 0.5) + 1):
            if self.bitrate(scale, qp) <= 0.0:
                return qp
            qp += 0.5
        return QP_SEARCH_MAX

    def invert(self, scale: float, target_kbps: float) -> dict:
        """Return QP_hat whose predicted bitrate equals target_kbps."""
        lo, hi = QP_SEARCH_MIN, QP_SEARCH_MAX
        b_lo, b_hi = self.bitrate(scale, lo), self.bitrate(scale, hi)
        if b_lo < target_kbps:
            return {
                "qp_hat": lo,
                "vmaf_hat": self.vmaf(scale, lo),
                "status": "target_below_reachable_range",
                "b_lo": b_lo,
                "b_hi": b_hi,
            }
        if b_hi > target_kbps:
            return {
                "qp_hat": hi,
                "vmaf_hat": self.vmaf(scale, hi),
                "status": "target_above_reachable_range",
                "b_lo": b_lo,
                "b_hi": b_hi,
            }
        # Bisection: bitrate is monotonic non-increasing in QP.
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.bitrate(scale, mid) > target_kbps:
                lo = mid
            else:
                hi = mid
        qp_hat = 0.5 * (lo + hi)
        b_hat = self.bitrate(scale, qp_hat)
        if abs(b_hat - target_kbps) > max(1e-3, 1e-3 * target_kbps):
            status = "bisection_did_not_converge"
        elif qp_hat < TRAINED_QP_MIN or qp_hat > TRAINED_QP_MAX:
            status = "outside_trained_qp_grid"
        else:
            status = "in_trained_qp_grid"
        return {
            "qp_hat": qp_hat,
            "vmaf_hat": self.vmaf(scale, qp_hat),
            "status": status,
            "b_lo": b_lo,
            "b_hi": b_hi,
        }


def analytic_inverse_qp(scale_idx: int, bitrate_kbps: float) -> float:
    """Equation 11 inverse: QP that the fitted exponential model needs."""
    pixels = float(RESOLUTION_PIXELS[scale_idx])
    s_mbit = bitrate_kbps / 1000.0 / 60.0  # kbps -> Mbit/frame at 60 fps
    denom = CONTENT_KAPPA_MEAN * pixels * PAYLOAD_BETA_REF
    return PAYLOAD_QP_REF - np.log(s_mbit / denom) / PAYLOAD_LAMBDA


def main() -> None:
    inverter = ProxyInverter()

    print("\n=== Achievable bitrate range of the proxy per resolution ===")
    print(
        f"{'scale':>6} {'pixels':>8} | "
        f"{'bitrate(QP=0)':>14} {'bitrate(QP=20)':>14} "
        f"{'bitrate(QP=40)':>14} {'bitrate(QP=100)':>15} "
        f"{'maxQP(bitrate>0)':>17}"
    )
    for res_idx, (scale, pixels) in enumerate(
        zip(RESOLUTION_SCALES, RESOLUTION_PIXELS)
    ):
        row = " ".join(
            f"{inverter.bitrate(scale, qp):14.0f}"
            for qp in (0.0, 20.0, 40.0, 100.0)
        )
        ceiling = inverter.positive_bitrate_qp_ceiling(scale)
        print(f"{scale:6.2f} {pixels:8d} | {row} {ceiling:17.1f}")

    print("\n=== Inverted equivalent QP per (resolution, target bitrate) ===")
    print(
        f"{'rate':>4} {'target':>7} | "
        + " | ".join(
            f"res{idx} QP_hat/VMAF/status" for idx in range(len(RESOLUTION_SCALES))
        )
    )
    summary = []
    for rate_idx, target in enumerate(TARGET_BITRATES_KBPS):
        cells = []
        for res_idx, scale in enumerate(RESOLUTION_SCALES):
            result = inverter.invert(scale, float(target))
            cells.append(
                f"{result['qp_hat']:5.1f}/{result['vmaf_hat']:5.1f}/"
                f"{result['status'][:8]}"
            )
            summary.append(
                (rate_idx, res_idx, result["qp_hat"], result["status"])
            )
        print(f"{rate_idx:4d} {target:7.0f} | " + " | ".join(cells))

    print("\n=== Current hand-set QP labels (scheduling_config) for reference ===")
    print(
        "rate_idx: " + "  ".join(f"{i:3d}" for i in range(5))
    )
    print(
        "target  : " + "  ".join(f"{b:6.0f}" for b in TARGET_BITRATES_KBPS)
    )
    print(
        "QP label: " + "  ".join(f"{q:3d}" for q in VP9_QP_BY_RATE)
    )

    print("\n=== Eq. 11 analytic inverse (mean content kappa) ===")
    print(
        f"{'rate':>4} {'target':>7} | "
        + " | ".join(f"res{idx}" for idx in range(len(RESOLUTION_SCALES)))
    )
    for rate_idx, target in enumerate(TARGET_BITRATES_KBPS):
        qps = [analytic_inverse_qp(res_idx, float(target)) for res_idx in range(5)]
        print(
            f"{rate_idx:4d} {target:7.0f} | "
            + " | ".join(f"{qp:6.1f}" for qp in qps)
        )

    n_in_grid = sum(
        1 for _, _, qp, status in summary
        if status == "in_trained_qp_grid"
    )
    n_any = sum(
        1 for _, _, qp, status in summary
        if status in ("in_trained_qp_grid", "outside_trained_qp_grid")
    )
    print(
        f"\nSummary: {n_in_grid}/25 inversions stay inside the trained QP grid "
        f"[{TRAINED_QP_MIN:.0f}, {TRAINED_QP_MAX:.0f}]; "
        f"{n_any}/25 have a finite (but possibly extrapolated) solution."
    )


if __name__ == "__main__":
    main()
