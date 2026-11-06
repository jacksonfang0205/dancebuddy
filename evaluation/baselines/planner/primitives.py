"""Motion primitive library and the plan -> curves compiler.

The agent never writes keyframes. It writes a *plan*: for each move unit of
the dance, a list of primitives, each placed on some bones and timed to
accent frames. This module turns a plan into per-frame channel curves
("curves.json"), which Blender then applies one frame at a time.

Channels (all relative to the rest pose):
  <bone>.lean    rotation forward/back, degrees   (bone-local X)
  <bone>.twist   rotation around the bone axis, degrees (bone-local Y)
  <bone>.sway    rotation side to side, degrees   (bone-local Z)
  root.lift      vertical offset, metres
  root.squash    0 = rest, 0.2 = 20% shorter (volume preserving)

Primitive families:
  * hits    (squash, nod, tilt): a fast ease-in that stops dead on
            the accent, then a slow release. bounce is a hop that lands on
            the accent. The sudden stop is what reads
            as a "hit" (and what the beat detector picks up).
  * swings  (sway, twist, rock): alternate between +amplitude and
            -amplitude, reaching each extreme exactly on an accent.
  * holds   (lean_hold): ramp to a pose, hold it for the span, ramp back.
  * follow  (follow): a bone continuously tracks one of the dancer's signals
            (signals.py), so the character copies the shape of a move (an arm
            lifting, the torso leaning) instead of only hitting its accents.
"""

from __future__ import annotations

import numpy as np

from dancebuddy.perception.beats import smooth
from dancebuddy.perception.signals import SIGNAL_NAMES, normalized

# primitive name -> (channel on each bone, family, units, sensible amplitude range)
PRIMITIVES: dict[str, dict] = {
    "bounce": {"channel": "lift", "family": "land", "units": "metres (hop height)", "range": [0.03, 0.3], "root_only": True,
               "doc": "whole body hops up and lands on each accent. Good for steps, footwork, jumps."},
    "squash": {"channel": "squash", "family": "hit", "units": "fraction", "range": [0.05, 0.35], "root_only": True,
               "doc": "body compresses on each accent then stretches back. Good for stomps, drops, landing a jump."},
    "nod": {"channel": "lean", "family": "hit", "units": "degrees", "range": [5, 35],
            "doc": "bones snap forward on each accent then return. Good for head nods, chest pops, punches."},
    "tilt": {"channel": "sway", "family": "hit_alternating", "units": "degrees", "range": [5, 35],
             "doc": "bones snap to the side on each accent, alternating left/right. Good for hip hits, shoulder pops."},
    "sway": {"channel": "sway", "family": "swing", "units": "degrees", "range": [5, 35],
             "doc": "smooth side-to-side swing, reaching each side on an accent. Good for arm waves, grooves, body rolls."},
    "twist": {"channel": "twist", "family": "swing", "units": "degrees", "range": [5, 45],
              "doc": "smooth turn left/right around the vertical axis, extremes on accents. Good for hip twists, turns, shoulder shimmies."},
    "rock": {"channel": "lean", "family": "swing", "units": "degrees", "range": [5, 30],
             "doc": "smooth forward/back rocking, extremes on accents. Good for body waves, steps forward and back."},
    "lean_hold": {"channel": "lean", "family": "hold", "units": "degrees", "range": [-30, 30],
                  "doc": "lean to a pose and hold it for the whole span (negative = backwards). Good for poses and freezes."},
    "follow": {"channel": None, "family": "follow", "units": "degrees at the dancer's typical extreme", "range": [-60, 60],
               "needs": ["signal", "channel", "bones"],
               "doc": "bones continuously track one dancer signal (see get_dance_signals), e.g. an arm bone following "
                      "arm_L_raise, or the body's sway following torso_side. Copies the shape of the move, not just its "
                      "accents. Choose 'channel' (lean | twist | sway). Negative amplitude mirrors the motion."},
}
FOLLOW_CHANNELS = ("lean", "twist", "sway")
FOLLOW_EDGE_S = 0.15  # follow fades in/out over this long outside its span, so units join smoothly

ATTACK_S = 0.12   # time to reach a hit
RELEASE_S = 0.45  # time to relax after a hit


def _hit_kernel(fps: float) -> tuple[np.ndarray, int]:
    """Shape of one hit, peak value 1 at index `peak`."""
    a = max(2, int(round(ATTACK_S * fps)))
    r = max(3, int(round(RELEASE_S * fps)))
    t_a = np.linspace(0, 1, a + 1)
    attack = t_a ** 2  # ease-in: fastest right at the accent, then a dead stop
    t_r = np.linspace(0, 1, r + 1)[1:]
    release = 1 - (3 * t_r ** 2 - 2 * t_r ** 3)  # smoothstep back to rest
    return np.concatenate([attack, release]), a


def _land_kernel(fps: float) -> tuple[np.ndarray, int]:
    """A hop: rise, then fall faster and faster, landing (value 0) on the accent."""
    u = max(2, int(round(0.18 * fps)))
    d = max(2, int(round(0.15 * fps)))
    t_u = np.linspace(0, 1, u + 1)[:-1]
    up = 3 * t_u ** 2 - 2 * t_u ** 3
    t_d = np.linspace(0, 1, d + 1)
    down = 1 - t_d ** 2
    return np.concatenate([up, down]), u + d


def _add_hits(curve: np.ndarray, accents: list[int], amp: float, fps: float, alternate: bool = False,
              land: bool = False) -> None:
    kern, peak = _land_kernel(fps) if land else _hit_kernel(fps)
    n = len(curve)
    for i, a in enumerate(accents):
        sign = (-1) ** i if alternate else 1
        start = a - peak
        for k, v in enumerate(kern):
            f = start + k
            if 0 <= f < n:
                # take the stronger of overlapping hits rather than summing them
                cand = sign * amp * v
                if abs(cand) > abs(curve[f]):
                    curve[f] = cand


def _swing(n: int, accents: list[int], amp: float, start: int, end: int, fps: float) -> np.ndarray:
    out = np.zeros(n)
    if not accents:
        return out
    keys = [(a, amp * (1 if i % 2 == 0 else -1)) for i, a in enumerate(accents)]
    ramp = max(3, int(round(0.25 * fps)))
    pts = [(max(start, keys[0][0] - ramp), 0.0)] + keys + [(min(end, keys[-1][0] + ramp), 0.0)]
    for (f0, v0), (f1, v1) in zip(pts[:-1], pts[1:]):
        if f1 <= f0:
            continue
        t = np.linspace(0, 1, f1 - f0 + 1)
        seg = v0 + (v1 - v0) * (3 * t ** 2 - 2 * t ** 3)  # zero velocity at each key
        lo, hi = max(0, f0), min(n - 1, f1)
        out[lo:hi + 1] = seg[lo - f0: hi - f0 + 1]
    return out


def _hold(n: int, amp: float, start: int, end: int, fps: float) -> np.ndarray:
    out = np.zeros(n)
    ramp = max(2, min(int(round(0.2 * fps)), (end - start) // 3))
    for f in range(max(0, start), min(n, end + 1)):
        if f < start + ramp:
            t = (f - start) / ramp
        elif f > end - ramp:
            t = (end - f) / ramp
        else:
            t = 1.0
        t = min(max(t, 0.0), 1.0)
        out[f] = amp * (3 * t ** 2 - 2 * t ** 3)
    return out


def resolve_accents(spec, unit: dict) -> list[int]:
    acc = sorted(unit.get("accents", []))
    if spec in (None, "all"):
        return acc
    if spec == "every_other":
        return acc[::2]
    if spec == "first":
        return acc[:1]
    if spec == "last":
        return acc[-1:]
    if isinstance(spec, list):
        return sorted(int(x) for x in spec)
    raise ValueError(f"unknown accents spec {spec!r}")


def compile_plan(plan: dict, score: dict, card: dict) -> dict:
    """Plan + dance score + character card -> curves dict (see module doc)."""
    fps = float(score["fps"])
    n = int(score["n_frames"])
    units = {u["id"]: u for u in score["units"]}
    bones = [b["name"] for b in card["bones"]]
    root = card.get("root_bone", bones[0])
    channels: dict[str, np.ndarray] = {}
    # follow: per (bone, channel, signal), the strongest weight seen at each frame and its amplitude
    follows: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    edge = max(1, int(round(FOLLOW_EDGE_S * fps)))

    def ch(name: str) -> np.ndarray:
        if name not in channels:
            channels[name] = np.zeros(n)
        return channels[name]

    for unit_plan in plan["units"]:
        unit = units.get(unit_plan["unit_id"])
        if unit is None:
            raise ValueError(f"plan refers to unknown unit_id {unit_plan['unit_id']}")
        for p in unit_plan["primitives"]:
            name = p["primitive"]
            if name not in PRIMITIVES:
                raise ValueError(f"unknown primitive {name!r}; choose from {sorted(PRIMITIVES)}")
            meta = PRIMITIVES[name]
            target_bones = [root] if meta.get("root_only") else p.get("bones", [])
            if not target_bones:
                raise ValueError(f"primitive {name!r} in unit {unit['id']} needs a non-empty 'bones' list")
            for b in target_bones:
                if b not in bones:
                    raise ValueError(f"unknown bone {b!r}; this character has {bones}")
            start = int(p.get("start", unit["start"]))
            end = int(p.get("end", unit["end"]))
            if meta["family"] == "follow":
                sig, chan = p.get("signal"), p.get("channel")
                if sig not in SIGNAL_NAMES:
                    raise ValueError(f"follow in unit {unit['id']} needs 'signal', one of {SIGNAL_NAMES}")
                if chan not in FOLLOW_CHANNELS:
                    raise ValueError(f"follow in unit {unit['id']} needs 'channel', one of {list(FOLLOW_CHANNELS)}")
                frames = np.arange(n)
                w = np.clip(1 - np.maximum(start - frames, frames - end) / edge, 0, 1)
                w[(frames >= start) & (frames <= end)] = 1.0
                for b in target_bones:
                    wt, amp_c = follows.setdefault((b, chan, sig), (np.zeros(n), np.zeros(n)))
                    take = w > wt
                    wt[take] = w[take]
                    amp_c[take] = float(p["amplitude"])
                continue
            acc = [a for a in resolve_accents(p.get("accents", "all"), unit) if start <= a <= end]
            amp = float(p["amplitude"])
            for b in target_bones:
                c = ch(f"{b}.{meta['channel']}")
                fam = meta["family"]
                if fam in ("hit", "land"):
                    tmp = np.zeros(n)
                    _add_hits(tmp, acc, amp, fps, land=(fam == "land"))
                    c += tmp
                elif fam == "hit_alternating":
                    tmp = np.zeros(n)
                    _add_hits(tmp, acc, amp, fps, alternate=True)
                    c += tmp
                elif fam == "swing":
                    c += _swing(n, acc, amp, start, end, fps)
                elif fam == "hold":
                    c += _hold(n, amp, start, end, fps)

    for (b, chan, sig), (wt, amp_c) in follows.items():
        sigma = max(1.0, 0.08 * fps)  # soften amplitude changes between units
        ch(f"{b}.{chan}")[:] += smooth(amp_c, sigma) * wt * normalized(score, sig)

    return {
        "fps": fps,
        "n_frames": n,
        "character": card.get("name", "character"),
        "channels": {k: [round(float(x), 5) for x in v] for k, v in sorted(channels.items())},
    }
