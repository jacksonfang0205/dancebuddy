"""Leg translation: the character has no legs, so every leg movement of the dancer becomes a visible movement of
the parts it does have, before anything else sees the motion (a renderer, or a video model that would otherwise
invent legs to show the footwork).

Events are found from the dancer's foot lift and hip height (signals.py, in torso lengths) and translated with the
character's motion spec, using classic animation timing so each one reads clearly:

  jump      both feet off the floor  -> squash to anticipate, stretch and hop (exaggerated), squash on landing
  kick      one foot high            -> lean away from the kicking leg, a small hop on the "support" side
  step      one foot lifts a little  -> a small bob with a squash when it lands, a lean onto the stepping side
  squat     hips go down             -> squash
  rise      hips go up, feet down    -> stretch
  travel    the dancer moves across  -> the whole character slides across

Output, per frame: lift (body heights), squash (+ squash / - stretch), tilt (deg, + toward the character's left)
and shift (body heights, + toward the character's left).
"""

from __future__ import annotations

import numpy as np

from dancebuddy.perception.beats import smooth
from dancebuddy.perception.video import L_AN, L_HIP, L_SH, R_AN, R_HIP, R_SH

RULES = {
    "seen": 0.8,           # a foot counts only where the pose model sees it this surely (wide trousers, occlusion)
    "air": 0.15,           # both feet higher than this (torso lengths) = in the air
    "kick": 0.25,          # one foot higher than this, the other down = kick
    "step": 0.07,          # one foot higher than this = step
    "down": 0.05,          # a foot lower than this is on the floor
    "jump_height": (0.4, 0.08, 0.22),   # hop height in body heights = gain x dancer's foot lift, clipped
    "anticipate_s": 0.15, "anticipate_squash": 0.12,
    "land_squash": 0.16, "land_s": 0.3,
    "takeoff_stretch": 0.08,
    "kick_tilt_deg": 16.0, "kick_hop": 0.03,
    "step_hop": 0.03, "step_squash": 0.05, "step_tilt_deg": 5.0,
    "squat_gain": 0.6, "squat_max": 0.15, "squash_max": 0.18,
    "rise_gain": 0.3, "rise_max": 0.05,
    "travel_per_torso": 0.25, "travel_max": 0.15,
}


def _segments(mask: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def floor_lifts(kp: dict) -> tuple[np.ndarray, np.ndarray]:
    """Height of each foot above the floor, in torso lengths, robust to the dancer walking toward or away from
    the camera and to cuts.

    With a fixed camera and a flat floor, a foot on the floor appears lower in the picture the closer the dancer
    is, and the dancer's size grows in step: floor_y = a + b * size. That line is fitted to the lower foot of
    every frame (per shot; a shot change shows as a sudden jump in size), keeping the frames on the floor. A foot
    is off the floor when it is above the line for the dancer's current size; a jump moves the whole body up
    without changing its size, walking away does not."""
    im = kp["image"]
    sh, hip = im[:, [L_SH, R_SH], :2].mean(1), im[:, [L_HIP, R_HIP], :2].mean(1)
    size = smooth(np.linalg.norm(sh - hip, axis=1), 2.0) + 1e-6
    ay = im[:, [L_AN, R_AN], 1]
    low = ay.max(1)  # image y grows downward: the lower foot is the one more likely on the floor
    n = len(size)
    rel = np.abs(np.diff(size)) / size[:-1]
    cuts = [0] + [i + 1 for i in np.nonzero(rel > 0.08)[0]] + [n]
    floor = np.zeros(n)
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        if s1 - s0 < 3:
            floor[s0:s1] = low[s0:s1]
            continue
        X = np.c_[np.ones(s1 - s0), size[s0:s1]]
        y = low[s0:s1]
        keep = np.ones(s1 - s0, bool)
        for _ in range(4):  # refit on the frames at or below the line (feet on the floor)
            coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            keep = (y - X @ coef) > -0.08 * size[s0:s1]
        floor[s0:s1] = X @ coef
    return (np.clip((floor - ay[:, 0]) / size, 0, None), np.clip((floor - ay[:, 1]) / size, 0, None))


def detect_events(kp: dict, signals: dict) -> list[dict]:
    """Leg events: {"type", "start", "peak", "end", "side" (L/R or None), "size" (torso lengths)}."""
    if "floor_lift_L" in signals:  # precomputed (e.g. tests)
        fl, fr = (np.asarray(signals[k], float) for k in ("floor_lift_L", "floor_lift_R"))
    else:
        fl, fr = (smooth(x, 1.0) for x in floor_lifts(kp))
    vis = kp.get("visibility")
    seen_l = vis[:, L_AN] >= RULES["seen"] if vis is not None else np.ones(len(fl), bool)
    seen_r = vis[:, R_AN] >= RULES["seen"] if vis is not None else np.ones(len(fl), bool)
    # a foot we cannot see is not evidence of a leg move: treat it as on the floor
    fl, fr = np.where(seen_l, fl, 0.0), np.where(seen_r, fr, 0.0)
    both = np.minimum(fl, fr)
    events = []
    air = both > RULES["air"]
    for s, e in _segments(air):
        events.append({"type": "jump", "start": s, "end": e, "peak": s + int(np.argmax(both[s:e + 1])),
                       "side": None, "size": float(both[s:e + 1].max())})
    for side, me, other in (("L", fl, fr), ("R", fr, fl)):
        solo = (me > RULES["step"]) & ~air
        for s, e in _segments(solo):
            pk = s + int(np.argmax(me[s:e + 1]))
            size = float(me[pk])
            kind = "kick" if size > RULES["kick"] and other[pk] < RULES["kick"] / 2 else "step"
            events.append({"type": kind, "start": s, "end": e, "peak": pk, "side": side, "size": size})
    # a foot leaving or touching the floor at the edge of a jump is part of the jump, not a step or a kick
    jumps = [ev for ev in events if ev["type"] == "jump"]
    edge = max(3, int(0.12 * float(kp["fps"])))
    events = [ev for ev in events if ev["type"] == "jump"
              or not any(j["start"] - edge <= ev["peak"] <= j["end"] + edge for j in jumps)]
    return sorted(events, key=lambda ev: ev["start"])


def _hip_height(kp: dict) -> tuple[np.ndarray, np.ndarray]:
    im = kp["image"]
    hip = im[:, [L_HIP, R_HIP]].mean(1)
    torso = np.median(np.abs(hip[:, 1] - im[:, [L_SH, R_SH], 1].mean(1))) + 1e-6
    height = -(hip[:, 1] - np.median(hip[:, 1])) / torso      # + = hips higher than usual
    across = (hip[:, 0] - hip[0, 0]) / torso                   # + = toward the image right = dancer's left
    return smooth(height, 1.0), smooth(across, 3.0)


def translate(kp: dict, signals: dict, overrides: dict[int, dict] | None = None) -> dict:
    """Per-frame lift / squash / tilt / shift for a legless body, plus the events found."""
    fps = float(kp["fps"])
    n = len(kp["world"])
    t = np.arange(n)
    lift, squash, tilt = np.zeros(n), np.zeros(n), np.zeros(n)
    events = detect_events(kp, signals)
    hip_h, across = _hip_height(kp)
    in_air = np.zeros(n, bool)
    landing = np.zeros(n)

    def bump(center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((t - center) / max(width, 1.0)) ** 2)

    overrides = overrides or {}
    for idx, ev in enumerate(events):
        ov = overrides.get(idx, {})
        if ov.get("skip"):
            continue
        k = float(ov.get("size", 1.0))  # the agent's choice: scale this event's translation
        s, e, pk = ev["start"], ev["end"], ev["peak"]
        if ev["type"] == "jump":
            in_air[s:e + 1] = True
            g, lo, hi = RULES["jump_height"]
            h = float(np.clip(g * ev["size"], lo, hi)) * k
            # stretch the hop over the whole time off the floor (a little longer reads better)
            s2, e2 = max(0, s - 1), min(n - 1, e + 1)
            phase = np.clip((t - s2) / max(1, e2 - s2), 0, 1)
            arc = np.where((t >= s2) & (t <= e2), np.sin(np.pi * phase), 0)
            lift = np.maximum(lift, h * arc)
            a = max(2, int(RULES["anticipate_s"] * fps))
            squash += k * RULES["anticipate_squash"] * np.where((t >= s2 - a) & (t < s2), (t - (s2 - a)) / a, 0)
            squash -= k * RULES["takeoff_stretch"] * np.where((t >= s2) & (t <= e2), np.sin(np.pi * phase), 0)
            land = k * RULES["land_squash"] * np.where(t > e2, np.exp(-(t - e2) / (RULES["land_s"] * fps / 2)), 0)
            landing = np.maximum(landing, land)
        elif ev["type"] == "kick":
            away = 1.0 if ev["side"] == "R" else -1.0          # lean away from the kicking leg
            env = bump(pk, (e - s) / 2.5)
            size = min(1.0, ev["size"] / 0.5)
            tilt += k * away * RULES["kick_tilt_deg"] * size * env
            lift = np.maximum(lift, k * RULES["kick_hop"] * size * env)
        else:  # step
            onto = 1.0 if ev["side"] == "L" else -1.0           # weight goes onto the stepping side
            env = bump(pk, max(2, (e - s) / 2.5))
            size = min(1.0, ev["size"] / RULES["kick"])
            lift = np.maximum(lift, k * RULES["step_hop"] * size * env)
            squash += k * RULES["step_squash"] * size * bump(e + 2, 2)
            tilt += k * onto * RULES["step_tilt_deg"] * size * env

    # squat and rise from the hip height, except in the air (the hop already shows that)
    squat = np.clip(-hip_h * RULES["squat_gain"], 0, RULES["squat_max"])
    rise = np.clip(hip_h * RULES["rise_gain"], 0, RULES["rise_max"])
    # a landing and the squat that follows it are one squash, not two
    squash += np.where(in_air, 0, np.maximum(squat, landing) - rise)
    shift = np.clip(across * RULES["travel_per_torso"], -RULES["travel_max"], RULES["travel_max"])
    return {"lift": smooth(lift, 0.8), "squash": np.clip(smooth(squash, 0.8), -0.1, RULES["squash_max"]),
            "tilt": smooth(tilt, 1.0), "shift": shift, "events": events}
