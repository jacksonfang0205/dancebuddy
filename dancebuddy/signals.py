"""Continuous dancer signals, so a plan can make a bone *follow* the dancer, not only hit the accents.

Each signal is one number per frame with a plain meaning (an angle or a displacement). The `follow` primitive
rescales a signal to a unit range using the dancer's own 5th-95th percentile, so the planner only chooses which
signal drives which bone and how far (amplitude, in the bone channel's units).

  two-sided signals (lean, twist, sway, bounce...)  ->  about -1..1 around the dancer's typical pose
  one-sided signals (arm raise, foot lift)          ->  about  0..1 from rest to the dancer's typical extreme
"""

from __future__ import annotations

import numpy as np

from .beats import smooth
from .perception import L_AN, L_EL, L_HIP, L_SH, NOSE, R_AN, R_EL, R_HIP, R_SH, wrap_deg

SIGNAL_INFO: dict[str, dict] = {
    "arm_L_raise": {"units": "degrees", "two_sided": False, "meaning": "dancer's left upper arm lifting (0 = hanging down, 90 = horizontal)"},
    "arm_R_raise": {"units": "degrees", "two_sided": False, "meaning": "dancer's right upper arm lifting"},
    "torso_lean": {"units": "degrees", "two_sided": True, "meaning": "torso bending forward (+) or back (-)"},
    "torso_side": {"units": "degrees", "two_sided": True, "meaning": "torso leaning to the dancer's left (+) or right (-)"},
    "torso_twist": {"units": "degrees", "two_sided": True, "meaning": "shoulders twisting relative to the hips"},
    "head_nod": {"units": "degrees", "two_sided": True, "meaning": "head nodding forward (+) or back (-) relative to the torso"},
    "head_tilt": {"units": "degrees", "two_sided": True, "meaning": "head tilting sideways relative to the torso"},
    "hip_bounce": {"units": "torso lengths", "two_sided": True, "meaning": "hips moving up (+) or down (-): knee bends, hops"},
    "hip_sway": {"units": "torso lengths", "two_sided": True, "meaning": "hips shifting to the dancer's left (+) or right (-)"},
    "foot_L_lift": {"units": "torso lengths", "two_sided": False, "meaning": "left foot lifted off the floor (steps, kicks)"},
    "foot_R_lift": {"units": "torso lengths", "two_sided": False, "meaning": "right foot lifted off the floor"},
}
SIGNAL_NAMES = sorted(SIGNAL_INFO)


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(a, b))


def _rolling_max(x: np.ndarray, width: int) -> np.ndarray:
    half = max(1, width // 2)
    return np.array([x[max(0, i - half): i + half + 1].max() for i in range(len(x))])


def _rolling_mean(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(width, len(x)))
    pad = np.pad(x, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(pad, np.ones(width) / width, mode="valid")


def compute_signals(kp: dict) -> dict[str, np.ndarray]:
    """Per-frame signals from MediaPipe keypoints (world: hip-centred metres, y down; image: normalised)."""
    w, img, fps = kp["world"], kp["image"], float(kp["fps"])
    sh, hip = w[:, [L_SH, R_SH]].mean(1), w[:, [L_HIP, R_HIP]].mean(1)
    v = sh - hip
    up = -v[:, 1]
    out = {
        "torso_lean": _angle(-v[:, 2], up),
        "torso_side": _angle(v[:, 0], up),
    }
    sh_vec, hip_vec = w[:, L_SH] - w[:, R_SH], w[:, L_HIP] - w[:, R_HIP]
    out["torso_twist"] = wrap_deg(_angle(sh_vec[:, 2], sh_vec[:, 0]) - _angle(hip_vec[:, 2], hip_vec[:, 0]))
    h = w[:, NOSE] - sh
    out["head_nod"] = _angle(-h[:, 2], -h[:, 1]) - out["torso_lean"]
    out["head_tilt"] = _angle(h[:, 0], -h[:, 1]) - out["torso_side"]
    for side, s_i, e_i in (("L", L_SH, L_EL), ("R", R_SH, R_EL)):
        u = w[:, e_i] - w[:, s_i]
        out[f"arm_{side}_raise"] = _angle(np.abs(u[:, 0]) + np.abs(u[:, 2]), u[:, 1])  # 0 down, 90 out, 180 up

    torso_len = np.median(np.abs(img[:, [L_HIP, R_HIP], 1].mean(1) - img[:, [L_SH, R_SH], 1].mean(1))) + 1e-6
    hip_x, hip_y = img[:, [L_HIP, R_HIP], 0].mean(1), img[:, [L_HIP, R_HIP], 1].mean(1)
    drift = int(2.0 * fps)  # remove slow camera / walking drift
    out["hip_bounce"] = -(hip_y - _rolling_mean(hip_y, drift)) / torso_len
    out["hip_sway"] = (hip_x - _rolling_mean(hip_x, drift)) / torso_len
    for side, a_i in (("L", L_AN), ("R", R_AN)):
        y = img[:, a_i, 1]
        out[f"foot_{side}_lift"] = (_rolling_max(y, int(1.0 * fps)) - y) / torso_len
    return {k: smooth(np.asarray(v, float), 1.0) for k, v in out.items()}


def signal_stats(signals: dict[str, list | np.ndarray]) -> dict[str, dict]:
    """5th/95th percentiles used to normalise each signal (stored in the dance score)."""
    stats = {}
    for name, vals in signals.items():
        lo, hi = np.percentile(np.asarray(vals, float), [5, 95])
        stats[name] = {**SIGNAL_INFO[name], "p5": round(float(lo), 3), "p95": round(float(hi), 3)}
    return stats


def normalized(score: dict, name: str) -> np.ndarray:
    """The signal rescaled for `follow`: two-sided -> about -1..1, one-sided -> about 0..1."""
    if "signals" not in score:
        raise ValueError("this dance score has no signals; re-run `perceive` (or `synth`) to add them")
    if name not in score["signals"]:
        raise ValueError(f"unknown signal {name!r}; choose from {SIGNAL_NAMES}")
    x = np.asarray(score["signals"][name], float)
    st = score["signal_stats"][name]
    lo, hi = st["p5"], st["p95"]
    span = max(hi - lo, 1e-6)
    if st["two_sided"]:
        return np.clip((x - (hi + lo) / 2) / (span / 2), -1.5, 1.5)
    return np.clip((x - lo) / span, 0.0, 1.5)


def unit_activity(score: dict) -> list[dict]:
    """For each move unit: how much each signal moves there (in units of its whole-clip range)."""
    rows = []
    for u in score["units"]:
        s, e = u["start"], u["end"] + 1
        act = {}
        for name in SIGNAL_NAMES:
            x = np.asarray(score["signals"][name][s:e], float)
            st = score["signal_stats"][name]
            span = max(st["p95"] - st["p5"], 1e-6)
            act[name] = round(float(np.ptp(x) / span), 2) if len(x) else 0.0
        top = [k for k, v in sorted(act.items(), key=lambda kv: -kv[1]) if v >= 0.5][:4]
        rows.append({"unit_id": u["id"], "most_active_signals": top, "activity": act})
    return rows
