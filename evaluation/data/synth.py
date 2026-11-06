"""A synthetic stick-figure dancer, for tests and for running the pipeline
without a real video. It dances 3 sections at 110 BPM: arm hits, hip twists,
then bounces (footwork). It writes keypoints.npz in MediaPipe's format and a
stick-figure dancer.mp4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dancebuddy.perception.video import (L_AN, L_EL, L_HIP, L_KN, L_SH, L_WR, NOSE, R_AN, R_EL, R_HIP, R_KN, R_SH, R_WR)

REST = {  # world coords, metres, MediaPipe convention: x right, y down, z toward camera; hips at origin
    NOSE: (0, -0.62, 0), L_SH: (0.18, -0.45, 0), R_SH: (-0.18, -0.45, 0),
    L_EL: (0.22, -0.18, 0), R_EL: (-0.22, -0.18, 0), L_WR: (0.24, 0.08, 0), R_WR: (-0.24, 0.08, 0),
    L_HIP: (0.1, 0, 0), R_HIP: (-0.1, 0, 0), L_KN: (0.11, 0.42, 0), R_KN: (-0.11, 0.42, 0),
    L_AN: (0.11, 0.85, 0), R_AN: (-0.11, 0.85, 0),
}
BONES = [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_SH, L_HIP), (R_SH, R_HIP),
         (L_HIP, R_HIP), (L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN)]


def hit(t: np.ndarray, beat_times: np.ndarray, attack=0.12, release=0.35) -> np.ndarray:
    out = np.zeros_like(t)
    for b in beat_times:
        a = (t > b - attack) & (t <= b)
        out[a] = np.maximum(out[a], ((t[a] - (b - attack)) / attack) ** 2)
        r = (t > b) & (t < b + release)
        x = (t[r] - b) / release
        out[r] = np.maximum(out[r], 1 - (3 * x ** 2 - 2 * x ** 3))
    return out


def make(out_dir: str | Path, seconds: float = 12.0, fps: float = 30.0, bpm: float = 110.0) -> Path:
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    period = 60.0 / bpm
    beats = np.arange(0.5, seconds, period)
    third = seconds / 3
    sec1, sec2, sec3 = beats[beats < third], beats[(beats >= third) & (beats < 2 * third)], beats[beats >= 2 * third]

    world = np.zeros((n, 33, 3))
    for j, p in REST.items():
        world[:, j] = p
    # section 1: arm hits, arms punch up on each beat, alternating
    h1 = hit(t, sec1[::2]); h1b = hit(t, sec1[1::2])
    world[:, L_WR, 1] -= 0.55 * h1; world[:, L_EL, 1] -= 0.3 * h1
    world[:, R_WR, 1] -= 0.55 * h1b; world[:, R_EL, 1] -= 0.3 * h1b
    # section 2: torso twists, shoulders rotate +-35 deg with extremes on beats
    tw = np.zeros(n)
    for i, b in enumerate(sec2):
        tw += (1 if i % 2 == 0 else -1) * hit(t, [b], attack=0.2, release=0.25)
    ang = np.radians(35) * tw
    for j in (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, NOSE):
        x, z = world[:, j, 0].copy(), world[:, j, 2].copy()
        world[:, j, 0] = x * np.cos(ang) - z * np.sin(ang)
        world[:, j, 2] = x * np.sin(ang) + z * np.cos(ang)
    # section 3: footwork, knees bend and the body bounces, alternating feet
    h3 = hit(t, sec3)
    h3l = hit(t, sec3[::2]); h3r = hit(t, sec3[1::2])
    world[:, L_KN, 2] += 0.15 * h3l; world[:, L_AN, 1] -= 0.15 * h3l
    world[:, R_KN, 2] += 0.15 * h3r; world[:, R_AN, 1] -= 0.15 * h3r
    bounce = 0.08 * h3  # whole body dips (y down = positive)

    # image coords: simple orthographic projection into a 540x960 portrait frame
    H, W = 960, 540
    scale = 0.38  # normalised image units per metre (height)
    image = np.zeros_like(world)
    image[..., 0] = 0.5 + world[..., 0] * scale * H / W
    image[..., 1] = 0.52 + (world[..., 1] + bounce[:, None]) * scale
    image[..., 2] = world[..., 2]
    vis = np.ones((n, 33))
    np.savez_compressed(out / "keypoints.npz", world=world, image=image, visibility=vis, fps=fps)

    vw = cv2.VideoWriter(str(out / "dancer.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in range(n):
        img = np.full((H, W, 3), 245, np.uint8)
        P = lambda j: (int(image[f, j, 0] * W), int(image[f, j, 1] * H))
        for a, b in BONES:
            cv2.line(img, P(a), P(b), (40, 40, 40), 6, cv2.LINE_AA)
        hx, hy = P(NOSE)
        cv2.circle(img, (hx, hy - 10), 34, (40, 40, 40), 6, cv2.LINE_AA)
        cv2.putText(img, "synthetic dancer", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 120, 120), 2)
        vw.write(img)
    vw.release()
    print(f"synthetic dancer: {n} frames, {len(beats)} beats -> {out}")
    return out
