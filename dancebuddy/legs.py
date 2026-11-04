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

from .beats import smooth
from .perception import L_HIP, L_SH, R_HIP, R_SH

RULES = {
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


def detect_events(kp: dict, signals: dict) -> list[dict]:
    """Leg events: {"type", "start", "peak", "end", "side" (L/R or None), "size" (torso lengths)}."""
    fl = np.asarray(signals["foot_L_lift"], float)
    fr = np.asarray(signals["foot_R_lift"], float)
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
    # a foot leaving or touching the floor at the edge of a jump is part of the jump, not a step
    jumps = [ev for ev in events if ev["type"] == "jump"]
    events = [ev for ev in events if ev["type"] != "step"
              or not any(j["start"] - 3 <= ev["peak"] <= j["end"] + 3 for j in jumps)]
    return sorted(events, key=lambda ev: ev["start"])


def _hip_height(kp: dict) -> tuple[np.ndarray, np.ndarray]:
    im = kp["image"]
    hip = im[:, [L_HIP, R_HIP]].mean(1)
    torso = np.median(np.abs(hip[:, 1] - im[:, [L_SH, R_SH], 1].mean(1))) + 1e-6
    height = -(hip[:, 1] - np.median(hip[:, 1])) / torso      # + = hips higher than usual
    across = (hip[:, 0] - hip[0, 0]) / torso                   # + = toward the image right = dancer's left
    return smooth(height, 1.0), smooth(across, 3.0)


def translate(kp: dict, signals: dict) -> dict:
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

    for ev in events:
        s, e, pk = ev["start"], ev["end"], ev["peak"]
        if ev["type"] == "jump":
            in_air[s:e + 1] = True
            g, lo, hi = RULES["jump_height"]
            h = float(np.clip(g * ev["size"], lo, hi))
            # stretch the hop over the whole time off the floor (a little longer reads better)
            s2, e2 = max(0, s - 1), min(n - 1, e + 1)
            phase = np.clip((t - s2) / max(1, e2 - s2), 0, 1)
            arc = np.where((t >= s2) & (t <= e2), np.sin(np.pi * phase), 0)
            lift = np.maximum(lift, h * arc)
            a = max(2, int(RULES["anticipate_s"] * fps))
            squash += RULES["anticipate_squash"] * np.where((t >= s2 - a) & (t < s2), (t - (s2 - a)) / a, 0)
            squash -= RULES["takeoff_stretch"] * np.where((t >= s2) & (t <= e2), np.sin(np.pi * phase), 0)
            land = RULES["land_squash"] * np.where(t > e2, np.exp(-(t - e2) / (RULES["land_s"] * fps / 2)), 0)
            landing = np.maximum(landing, land)
        elif ev["type"] == "kick":
            away = 1.0 if ev["side"] == "R" else -1.0          # lean away from the kicking leg
            env = bump(pk, (e - s) / 2.5)
            size = min(1.0, ev["size"] / 0.5)
            tilt += away * RULES["kick_tilt_deg"] * size * env
            lift = np.maximum(lift, RULES["kick_hop"] * size * env)
        else:  # step
            onto = 1.0 if ev["side"] == "L" else -1.0           # weight goes onto the stepping side
            env = bump(pk, max(2, (e - s) / 2.5))
            size = min(1.0, ev["size"] / RULES["kick"])
            lift = np.maximum(lift, RULES["step_hop"] * size * env)
            squash += RULES["step_squash"] * size * bump(e + 2, 2)
            tilt += onto * RULES["step_tilt_deg"] * size * env

    # squat and rise from the hip height, except in the air (the hop already shows that)
    squat = np.clip(-hip_h * RULES["squat_gain"], 0, RULES["squat_max"])
    rise = np.clip(hip_h * RULES["rise_gain"], 0, RULES["rise_max"])
    # a landing and the squat that follows it are one squash, not two
    squash += np.where(in_air, 0, np.maximum(squat, landing) - rise)
    shift = np.clip(across * RULES["travel_per_torso"], -RULES["travel_max"], RULES["travel_max"])
    return {"lift": smooth(lift, 0.8), "squash": np.clip(smooth(squash, 0.8), -0.1, RULES["squash_max"]),
            "tilt": smooth(tilt, 1.0), "shift": shift, "events": events}
