"""Baseline: direct pose copy.

Copies the dancer's spine, head and hip motion straight onto the character's
bone chain, frame by frame, the way a standard retargeter would. Arms and legs
have no matching bones on a non-humanoid body, so their motion is simply lost.
That loss is exactly what DanceBuddy's planner is supposed to fix.
"""

from __future__ import annotations

import numpy as np

from .beats import smooth
from .perception import L_EL, L_HIP, L_SH, NOSE, R_EL, R_HIP, R_SH, wrap_deg


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(a, b))


def pose_copy(kp: dict, card: dict) -> dict:
    w, img, fps = kp["world"], kp["image"], kp["fps"]
    n = len(w)
    sh = w[:, [L_SH, R_SH]].mean(1)
    hip = w[:, [L_HIP, R_HIP]].mean(1)
    v = sh - hip  # MediaPipe world: x right, y down, z toward/away from camera
    up = -v[:, 1]
    torso_lean = _angle(-v[:, 2], up)
    torso_side = _angle(v[:, 0], up)
    sh_vec, hip_vec = w[:, L_SH] - w[:, R_SH], w[:, L_HIP] - w[:, R_HIP]
    twist = wrap_deg(_angle(sh_vec[:, 2], sh_vec[:, 0]) - _angle(hip_vec[:, 2], hip_vec[:, 0]))
    h = w[:, NOSE] - sh
    head_lean = _angle(-h[:, 2], -h[:, 1]) - torso_lean
    head_side = _angle(h[:, 0], -h[:, 1]) - torso_side

    torso_len = np.median(np.abs(img[:, [L_HIP, R_HIP], 1].mean(1) - img[:, [L_SH, R_SH], 1].mean(1))) + 1e-6
    hip_y = img[:, [L_HIP, R_HIP], 1].mean(1)
    lift = np.clip((np.median(hip_y) - hip_y) / torso_len * 0.5, 0, None)

    def rel(x):  # motion relative to the dancer's typical pose, lightly smoothed
        return smooth(x - np.median(x), 1.0)

    bones = [b["name"] for b in card["bones"] if not b.get("limb")]
    root = card.get("root_bone", bones[0])
    chain = [b for b in bones if b != root]
    ch: dict[str, np.ndarray] = {f"{root}.lift": smooth(lift, 1.0)}
    # spread torso motion over the body bones, give the head motion to the top bone
    body, top = chain[:-1] or chain, chain[-1]
    for b in body:
        ch[f"{b}.lean"] = rel(torso_lean) / len(body)
        ch[f"{b}.sway"] = rel(torso_side) / len(body)
        ch[f"{b}.twist"] = rel(twist) / len(body)
    ch[f"{top}.lean"] = rel(head_lean)
    ch[f"{top}.sway"] = rel(head_side)
    # limbs: copy the dancer's upper-arm elevation onto arm bones of the same side
    for b in card["bones"]:
        if b.get("limb") != "arm":
            continue
        sh_i, el_i = (L_SH, L_EL) if b.get("side") == "left" else (R_SH, R_EL)
        u = w[:, el_i] - w[:, sh_i]
        elevation = _angle(np.abs(u[:, 0]), u[:, 1])  # 0 = arm hanging down, 90 = horizontal
        ch[f"{b['name']}.lean"] = rel(elevation)
    return {
        "fps": fps, "n_frames": n, "character": card.get("name", "character"),
        "channels": {k: [round(float(x), 5) for x in v] for k, v in sorted(ch.items())},
    }
