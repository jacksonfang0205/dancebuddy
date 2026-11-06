"""Code-based checks (used by the agent's verifier) and metrics (used by the eval).

Everything here works on curves.json, so it runs without Blender and scores
every system (agent, pose-copy baseline, ablations) the same way.
"""

from __future__ import annotations

import numpy as np

from .beats import detect_hits, match, smooth

# how much change per frame in each channel type counts as "1 unit" of motion
CHANNEL_SCALE = {"lift": 0.05, "squash": 0.05, "lean": 5.0, "twist": 5.0, "sway": 5.0}

THRESHOLDS = {
    "accent_recall": 0.75,      # share of dancer accents the character hits
    "beat_precision": 0.5,      # share of character hits that land on a dancer beat
    "max_abs_offset_frames": 3,  # tolerance for "on the beat" (3 frames = 100 ms at 30 fps)
    "idle_unit_energy": 0.05,   # a move unit with less mean motion than this is "idle"
}


def unit_report(curves: dict, score: dict, missed: list[int], off_beat: list[int]) -> list[dict]:
    """Per move unit: what the dancer does there and what went wrong, so a revision knows where to look."""
    speed = motion_speed(curves)
    dancer = np.asarray(score.get("motion_energy", []), float)
    rows = []
    for u in score["units"]:
        s, e = u["start"], u["end"] + 1
        char_e = float(speed[s:e].mean()) if e > s else 0.0
        row = {"unit_id": u["id"], "dancer": u.get("label"), "dancer_regions": u.get("dominant_regions"),
               "accents": len(u["accents"]), "missed_accents": [f for f in missed if s <= f < e],
               "off_beat_hits": [f for f in off_beat if s <= f < e], "character_motion": round(char_e, 3)}
        if len(dancer) >= e and e > s:
            row["dancer_motion_vs_clip"] = round(float(dancer[s:e].mean() / (dancer.mean() + 1e-9)), 2)
        problems = []
        if row["missed_accents"]:
            problems.append(f"missed {len(row['missed_accents'])} of {row['accents']} accents")
        if row["off_beat_hits"]:
            problems.append(f"{len(row['off_beat_hits'])} hits off the beat")
        if char_e < THRESHOLDS["idle_unit_energy"]:
            problems.append("character barely moves")
        row["problems"] = problems
        rows.append(row)
    return rows


def channel_kind(name: str) -> str:
    return name.split(".", 1)[1]


def motion_speed(curves: dict) -> np.ndarray:
    """Per-frame motion energy: sum over channels of |change| / channel scale, per second."""
    n = int(curves["n_frames"])
    fps = float(curves["fps"])
    speed = np.zeros(n)
    for name, vals in curves["channels"].items():
        v = np.asarray(vals, float)
        d = np.abs(np.gradient(v)) / CHANNEL_SCALE.get(channel_kind(name), 1.0)
        speed += d
    return speed * fps / 30.0  # normalise so numbers are comparable across fps


def character_hits(curves: dict, threshold: float = 0.25) -> list[int]:
    speed = motion_speed(curves)
    if speed.max() <= 1e-9:
        return []
    # Strength is relative to the motion around the hit (90th percentile within +-1 s, at least 0.5), not to
    # the whole clip: otherwise a unit with large continuous motion (e.g. `follow`) makes the hits of every
    # other unit look weak.
    # A held pose has speed exactly 0; without a floor the minimum of the smoothed curve lands a few frames
    # after the stop (where the smoothing tail reaches 0) instead of on it.
    fps = float(curves["fps"])
    floor = 0.05
    raw = detect_hits(np.maximum(smooth(speed, 0.7), floor), fps, sigma=0, scale=1.0)
    half = max(1, int(round(fps)))
    local = lambda f: max(float(np.percentile(speed[max(0, f - half): f + half + 1], 90)), 0.5)
    return [f for f, st in raw if st / local(f) >= threshold]


def run_checks(curves: dict, score: dict, card: dict) -> dict:
    fps = float(score["fps"])
    tol = THRESHOLDS["max_abs_offset_frames"]
    hits = character_hits(curves)
    accents = score["accents"]
    beats = score["beats"]

    matched, missed, offsets = match(accents, hits, tol)
    recall = len(matched) / max(1, len(accents))
    on_beat, off_beat, _ = match(hits, beats, tol)
    precision = len(on_beat) / max(1, len(hits))

    # limits
    limits = card.get("limits", {})
    violations = []
    for name, vals in curves["channels"].items():
        kind = channel_kind(name)
        lim = limits.get(name, limits.get(kind))
        if lim is None:
            continue
        v = np.asarray(vals, float)
        bad = np.where(np.abs(v) > lim)[0]
        if len(bad):
            violations.append({"channel": name, "limit": lim, "peak": round(float(np.abs(v).max()), 3),
                               "frames": [int(bad[0]), int(bad[-1])]})

    # idle units
    speed = motion_speed(curves)
    idle = []
    for u in score["units"]:
        e = float(speed[u["start"]:u["end"] + 1].mean()) if u["end"] >= u["start"] else 0.0
        if e < THRESHOLDS["idle_unit_energy"]:
            idle.append(u["id"])

    def unit_of(f):
        for u in score["units"]:
            if u["start"] <= f <= u["end"]:
                return u["id"]
        return None

    checks = {
        "accent_recall": {
            "value": round(recall, 3), "threshold": THRESHOLDS["accent_recall"],
            "passed": recall >= THRESHOLDS["accent_recall"],
            "missed_accents": [{"frame": f, "unit_id": unit_of(f)} for f in missed],
        },
        "beat_precision": {
            "value": round(precision, 3), "threshold": THRESHOLDS["beat_precision"],
            "passed": precision >= THRESHOLDS["beat_precision"] or len(hits) == 0,
            "off_beat_hits": [{"frame": f, "unit_id": unit_of(f)} for f in off_beat][:20],
        },
        "limits": {"passed": not violations, "violations": violations},
        "idle_units": {"passed": not idle, "unit_ids": idle},
    }
    return {
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
        "summary": {
            "n_accents": len(accents), "n_character_hits": len(hits),
            "mean_offset_ms": round(float(np.mean(offsets)) * 1000 / fps, 1) if offsets else None,
        },
        "units": [r for r in unit_report(curves, score, missed, off_beat) if r["problems"]],
    }


def metrics(curves: dict, score: dict, card: dict) -> dict:
    """Numbers for the results table (a superset of the checks)."""
    res = run_checks(curves, score, card)
    c = res["checks"]
    speed = motion_speed(curves)
    hits = character_hits(curves)
    # beat alignment score (Li et al. 2021 style): mean over character hits of
    # exp(-d^2 / (2 sigma^2)), d = distance to nearest dancer beat, sigma = 3 frames
    beats = np.asarray(score["beats"], float)
    if len(hits) and len(beats):
        d = np.array([np.min(np.abs(beats - h)) for h in hits])
        bas = float(np.mean(np.exp(-(d ** 2) / (2 * 3.0 ** 2))))
    else:
        bas = 0.0
    jerk = float(np.mean(np.abs(np.diff(speed, 2)))) if len(speed) > 3 else 0.0
    return {
        "accent_recall": c["accent_recall"]["value"],
        "beat_precision": c["beat_precision"]["value"],
        "beat_alignment": round(bas, 3),
        "mean_offset_ms": res["summary"]["mean_offset_ms"],
        "limit_violations": len(c["limits"]["violations"]),
        "idle_units": len(c["idle_units"]["unit_ids"]),
        "mean_motion": round(float(speed.mean()), 3),
        "jerk": round(jerk, 3),
        "all_checks_passed": res["passed"],
    }
