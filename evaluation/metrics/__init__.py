"""Metrics, the same for every system: they read only the character's motion (posed from curves.json in Blender,
blender/measure_motion.py), the dancer's keypoints and dance score, and the system's own record of its calls.

  rhythm     accent_recall      share of the dancer's accents the character hits (a stop after fast motion, ±0.1 s)
             beat_precision     share of the character's hits that land on a beat
             motion_corr        correlation of the character's and the dancer's amount of motion over time
  fidelity   arm_follow         correlation of each hand's position (relative to the shoulders) with the dancer's
                                wrist, over time, averaged over both hands and both axes (-1..1)
  integrity  hands_in_frame     share of frames with both hands inside the picture
             arm_folds / arm_sinking   frames where an arm folds / sinks into the body (blender/check_arm_shape.py)
  cost       llm_calls, cost_usd, attempts, passed (the system's own checks)
  pixels     px_accent_recall, px_beat_precision on rendered frames, when they exist (pixels.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dancebuddy.perception.beats import detect_hits, match, smooth
from dancebuddy.perception.video import L_HIP, L_SH, L_WR, R_HIP, R_SH, R_WR

TOL_S = 0.1


def _full(rows: list[dict], key, n: int, every: int) -> np.ndarray:
    """A measured series (every `every`-th frame) on all n frames."""
    xs = np.array([r["frame"] for r in rows], float)
    ys = np.array([key(r) for r in rows], float)
    return np.interp(np.arange(n), xs, ys)


def rhythm(measure: dict, score: dict) -> dict:
    n, fps = int(score["n_frames"]), float(score["fps"])
    energy = _full(measure["rows"], lambda r: r["energy"], n, measure["every"])
    s = smooth(energy, 1.0)
    scale = max(float(np.percentile(s, 90)), 1e-9)
    hits = [f for f, st in detect_hits(np.maximum(s, 0.1 * scale), fps, sigma=0, scale=scale) if st >= 0.25]
    tol = max(1, int(round(TOL_S * fps)))
    accents = [a for a in score["accents"] if a < n]
    matched, _, offsets = match(accents, hits, tol)
    on_beat, _, _ = match(hits, [b for b in score["beats"] if b < n], tol)
    dancer = np.asarray(score.get("motion_energy", []), float)[:n]
    corr = float(np.corrcoef(s[:len(dancer)], smooth(dancer, 1.0))[0, 1]) if len(dancer) > 3 and s.std() > 0 else 0.0
    return {"accent_recall": round(len(matched) / max(1, len(accents)), 3),
            "beat_precision": round(len(on_beat) / max(1, len(hits)), 3),
            "mean_offset_ms": round(float(np.mean(np.abs(offsets))) / fps * 1000, 1) if len(offsets) else None,
            "motion_corr": round(corr, 3)}


def fidelity(measure: dict, kp: dict) -> dict:
    rows = [r for r in measure["rows"] if "hands" in r]
    if not rows:
        return {"arm_follow": None}
    im = kp["image"]
    torso = np.linalg.norm(im[:, [L_SH, R_SH]].mean(1) - im[:, [L_HIP, R_HIP]].mean(1), axis=1) + 1e-6
    sh = im[:, [L_SH, R_SH]].mean(1)
    scores = []
    for side, wrist in (("L", L_WR), ("R", R_WR)):
        fr = [r["frame"] for r in rows if side in r["hands"]]
        if len(fr) < 4:
            continue
        ch = np.array([[(r["hands"][side][k] - r["shoulders"][k]) / max(r["height_px"], 1e-6) for k in (0, 1)]
                       for r in rows if side in r["hands"]])
        dn = (im[fr, wrist] - sh[fr]) / torso[fr, None]
        for k in (0, 1):
            if ch[:, k].std() > 1e-6 and dn[:, k].std() > 1e-6:
                scores.append(float(np.corrcoef(ch[:, k], dn[:, k])[0, 1]))
    return {"arm_follow": round(float(np.mean(scores)), 3) if scores else None}


def integrity(measure: dict, shape: dict | None = None) -> dict:
    rows = [r for r in measure["rows"] if "in_frame" in r]
    out = {"hands_in_frame": round(float(np.mean([all(r["in_frame"].values()) for r in rows])), 3) if rows else None}
    if shape:
        arm_frames = [r[s] for r in shape["rows"] for s in "LR"]
        out["arm_folds"] = round(float(np.mean([x["fold"] < 1.0 for x in arm_frames])), 3)
        out["arm_sinking"] = round(float(np.mean([x["inside_pts"] > 40 and x["inside_cm"] > 3 for x in arm_frames])), 3)
    return out


def cost(system_dir: Path) -> dict:
    p = Path(system_dir) / "summary.json"
    if not p.exists():
        return {"llm_calls": 0, "cost_usd": 0.0}
    s = json.loads(p.read_text())
    return {"llm_calls": s.get("llm_calls"), "cost_usd": s.get("cost_usd"), "attempts": s.get("attempts"),
            "passed": s.get("passed")}
