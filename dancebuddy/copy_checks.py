"""Checks on a copy (curves from retarget.py), reported per move unit so the agent knows where to revise.

  accents      each dancer accent should have a visible character hit (a stop after fast motion) within 0.1 s
  completeness in each unit the character should move about as much (relative to its own average) as the dancer
               (relative to hers); much less means the copy ate the move
  in_frame     the slide plus the body tilt must keep the character inside the picture
  range        body tilt / turn within what the character can do (motion spec "can_do")
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .beats import detect_hits, match, smooth

# frame_margin: how far (x body height) the top of the body may drift sideways before the hair leaves the picture:
# the camera (rig_soft.py: 45 mm lens at 3 body heights, portrait) sees +-0.68 heights, the hair is +-0.35 wide
THRESHOLDS = {"accent_recall": 0.6, "completeness": 0.6, "frame_margin": 0.32, "tol_s": 0.1}


def _rot_series(q: list) -> Rot:
    q = np.asarray(q, float)
    return Rot.from_quat(q[:, [1, 2, 3, 0]])


def character_speed(curves: dict) -> np.ndarray:
    """Per-frame motion of the character (deg/s of its main bones + m/s of its root, as one number)."""
    ch, fps = curves["channels"], float(curves["fps"])
    n = int(curves["n_frames"])
    sp = np.zeros(n)
    for name, w in (("root.quat", 1.0), ("body_upper.quat", 0.7), ("arm_L_1.quat", 0.5), ("arm_R_1.quat", 0.5),
                    ("arm_L_3.quat", 0.3), ("arm_R_3.quat", 0.3)):
        if name in ch:
            r = _rot_series(ch[name])
            d = (r[1:] * r[:-1].inv()).magnitude()
            sp += w * np.degrees(np.r_[d[:1], d]) * fps
    if "root.loc" in ch:
        loc = np.asarray(ch["root.loc"], float)
        v = np.linalg.norm(np.diff(loc, axis=0), axis=1) * fps
        sp += 100.0 * np.r_[v[:1], v]  # 1 m/s ~ 100 deg/s
    if "root.squash" in ch:
        sq = np.asarray(ch["root.squash"], float)
        sp += 100.0 * np.abs(np.gradient(sq)) * fps
    return smooth(sp, 1.0)


def run_copy_checks(curves: dict, score: dict, rig: dict, spec: dict | None = None) -> dict:
    fps = float(curves["fps"])
    n = int(curves["n_frames"])
    height = float(rig.get("height_m", 1.4))
    tol = max(1, int(round(THRESHOLDS["tol_s"] * fps)))
    sp = character_speed(curves)
    hits = [f for f, st in detect_hits(sp, fps, sigma=0.7, scale=max(float(np.percentile(sp, 90)), 1e-6)) if st >= 0.25]
    accents = [a for a in score["accents"] if a < n]
    matched, missed, _ = match(accents, hits, tol)
    recall = len(matched) / max(1, len(accents))

    dancer = np.asarray(score.get("motion_energy", np.ones(n)), float)[:n]
    m = Rot.from_matrix(np.array(next(b["matrix"] for b in rig["bones"] if b["name"] == rig.get("root_bone", "root"))))
    up = (m * _rot_series(curves["channels"]["root.quat"]) * m.inv()).apply([0, 0, 1])  # bone frame -> armature
    loc = np.asarray(curves["channels"].get("root.loc", np.zeros((n, 3))), float)
    reach = np.abs(loc[:, 0]) + 0.7 * height * np.abs(up[:, 0])  # sideways offset of the top of the body
    out_frames = np.nonzero(reach > THRESHOLDS["frame_margin"] * height)[0]
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    can = (spec or {}).get("parts", {}).get("body", {}).get("can_do", {})
    tilt_max = float(can.get("tilt_deg", 25)) + 5

    units = []
    for u in score["units"]:
        s, e = u["start"], min(n, u["end"] + 1)
        if e <= s:
            continue
        ratio = float((sp[s:e].mean() / (sp.mean() + 1e-9)) / (dancer[s:e].mean() / (dancer.mean() + 1e-9) + 1e-9))
        miss = [int(a) for a in missed if s <= a < e]
        outs = [int(f) for f in out_frames if s <= f < e]
        problems = []
        if miss:
            problems.append(f"no visible hit on {len(miss)} of {len([a for a in accents if s <= a < e])} accents "
                            f"(frames {miss})")
        if ratio < THRESHOLDS["completeness"]:
            problems.append(f"the character moves {ratio:.2f}x as much as the dancer here (relative to each one's "
                            "average): the copy eats this move")
        if outs:
            worst = max(outs, key=lambda f: reach[f])
            problems.append(f"the character leaves the picture on {len(outs)} frames; at the worst frame ({worst}) "
                            f"the slide is {abs(loc[worst, 0]):.2f} m and the body tilt moves its top "
                            f"{0.7 * height * abs(up[worst, 0]):.2f} m sideways (limit together "
                            f"{THRESHOLDS['frame_margin'] * height:.2f} m)")
        if tilt[s:e].max() > tilt_max:
            problems.append(f"body tilt reaches {tilt[s:e].max():.0f} deg, beyond what the character can do")
        units.append({"unit_id": u["id"], "time_s": [u["start_s"], u["end_s"]], "dancer": u.get("label"),
                      "dancer_regions": u.get("dominant_regions"), "completeness": round(ratio, 2),
                      "missed_accents": miss, "frames_out_of_picture": len(outs), "problems": problems})
    checks = {
        "accents": {"value": round(recall, 3), "threshold": THRESHOLDS["accent_recall"],
                    "passed": recall >= THRESHOLDS["accent_recall"]},
        "completeness": {"passed": all(x["completeness"] >= THRESHOLDS["completeness"] for x in units),
                         "lowest": min((x["completeness"] for x in units), default=1.0)},
        "in_frame": {"passed": len(out_frames) == 0, "frames_out": int(len(out_frames))},
        "range": {"passed": float(tilt.max()) <= tilt_max, "max_tilt_deg": round(float(tilt.max()), 1)},
    }
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "units": [x for x in units if x["problems"]], "all_units": units}


def rank(res: dict) -> tuple:
    """Higher is better: passed first, then fewer problems, then accent recall."""
    n_prob = sum(len(u["problems"]) for u in res["units"])
    return (res["passed"], -n_prob, res["checks"]["accents"]["value"])
