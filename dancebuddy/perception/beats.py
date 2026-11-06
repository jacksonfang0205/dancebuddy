"""Kinematic beat and accent detection.

A "hit" in dance is a moment where fast motion stops sharply. We find hits as
local minima of a (smoothed) speed curve, and score each minimum by how much
the speed dropped into it. The same detector is used on the dancer's
keypoints and on the character's animation curves, so the two are comparable.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


def smooth(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return x
    return gaussian_filter1d(x.astype(float), sigma, mode="nearest")


def detect_hits(
    speed: np.ndarray,
    fps: float,
    window_s: float = 0.15,
    min_gap_s: float = 0.2,
    sigma: float = 0.7,
    scale: float | None = None,
) -> list[tuple[int, float]]:
    """Return [(frame, strength)] for every local speed minimum.

    strength = (max speed in the preceding window - speed at the minimum) / scale,
    where scale defaults to the 90th percentile of speed. Strength is roughly
    0..1+; a sharp stop after fast motion scores high, a slow drift scores low.
    """
    s = smooth(np.asarray(speed, float), sigma)
    n = len(s)
    if n < 3:
        return []
    if scale is None:
        scale = float(np.percentile(s, 90))
    scale = max(scale, 1e-9)
    win = max(1, int(round(window_s * fps)))
    # local minima, including plateaus (take first index of a plateau)
    mins = argrelextrema(s, np.less_equal, order=1)[0]
    hits = []
    for f in mins:
        if f == 0 or f == n - 1:
            continue
        if s[f - 1] == s[f]:  # inside a plateau, keep only its first frame
            continue
        lo = max(0, f - win)
        strength = (s[lo:f].max() - s[f]) / scale
        hits.append((int(f), float(strength)))
    # non-maximum suppression by strength within min_gap
    gap = max(1, int(round(min_gap_s * fps)))
    hits.sort(key=lambda h: -h[1])
    kept: list[tuple[int, float]] = []
    for f, st in hits:
        if all(abs(f - k) >= gap for k, _ in kept):
            kept.append((f, st))
    kept.sort()
    return kept


def beats_and_accents(
    speed: np.ndarray,
    fps: float,
    beat_threshold: float = 0.25,
    local_ratio: float = 0.65,
    local_window_s: float = 1.0,
) -> tuple[list[int], list[int]]:
    """Beats are all hits above beat_threshold. Accents are the hits that are
    strong compared with their neighbours (>= local_ratio x the strongest hit
    within +-local_window_s), so a quiet section still gets its accents."""
    hits = [(f, st) for f, st in detect_hits(speed, fps) if st >= beat_threshold]
    beats = [f for f, _ in hits]
    w = local_window_s * fps
    accents = []
    for f, st in hits:
        local_max = max(s2 for f2, s2 in hits if abs(f2 - f) <= w)
        if st >= local_ratio * local_max:
            accents.append(f)
    return beats, accents


def match(reference: list[int], predicted: list[int], tol: int) -> tuple[list[int], list[int], list[float]]:
    """Greedy one-to-one matching within +-tol frames.

    Returns (matched reference frames, unmatched reference frames, offsets in
    frames for matched pairs, predicted - reference).
    """
    pred = sorted(predicted)
    used = [False] * len(pred)
    matched, missed, offsets = [], [], []
    for r in sorted(reference):
        best, best_d = None, None
        for i, p in enumerate(pred):
            if used[i]:
                continue
            d = abs(p - r)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is None:
            missed.append(r)
        else:
            used[best] = True
            matched.append(r)
            offsets.append(pred[best] - r)
    return matched, missed, offsets
