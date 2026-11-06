"""The agent's own moves: a few lines of code that put the character in a clear pose at a given time.

The copy (retarget.py) follows the dancer; where the agent sees that a move does not read (hands not above the
head, hands not meeting at the chest, a lean too small), it writes what the character should do there. Each move
blends in before its time, holds, and blends back into the copy. The code is parsed, never run: only these calls
with constant arguments are accepted, one per line (# comments allowed).

  arm(side, t, pose=None, raise_=None, forward=None, bend=None, fist=None, hold=0.2, attack=0.15, release=0.25)
      side "L" (the character's left = the picture's right), "R" or "both"; t in seconds
      pose: a named pose (POSES), or give the hand's direction from the shoulder:
        raise_  0 = hanging, 90 = straight out to the side, 180 = straight up (degrees)
        forward 0 = out to the side in the picture plane, 90 = toward the camera, beyond 90 on across the front
                toward the other side (180 = across the body) (degrees)
        bend    0 = straight arm, up to 160 = curled; the forearm curls toward the head and the middle
      fist True / False to close or open the hand during the move
  body(t, tilt=0, lean=0, turn=0, squash=0, lift=0, hold=0.2, attack=0.15, release=0.25)
      added to the copy: tilt + = toward the character's left (deg), lean + = forward (deg), turn (deg),
      squash 0..0.25, lift = a hop (body heights)
  hit(t, move, size=0.6)
      an accent move landing at t: squash, hop, nod, tilt_left, tilt_right, twist
"""

from __future__ import annotations

import ast

import numpy as np

POSES = {  # name: (raise_, forward, bend, fist) for one arm; "both" mirrors it
    "hanging": (15, 10, 10, False),
    "out_side": (90, 10, 10, False),
    "overhead": (172, 15, 0, True),
    "raised_bent": (120, 10, 100, True),        # upper arm out, forearm up: a soft elbow, fist
    "hands_together_chest": (60, 130, 40, False),
    "hands_to_face": (115, 125, 60, False),
    "across_chest": (80, 145, 30, False),
    "point_forward": (90, 75, 0, False),
}
ARM_KEYS = {"pose", "raise_", "forward", "bend", "fist", "hold", "attack", "release"}
BODY_KEYS = {"tilt", "lean", "turn", "squash", "lift", "hold", "attack", "release"}
HIT_MOVES = ("squash", "hop", "nod", "tilt_left", "tilt_right", "twist")


class MoveError(ValueError):
    pass


def _const(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except ValueError as e:
        raise MoveError(f"only constant arguments are allowed ({ast.unparse(node)})") from e


def parse(code: str) -> list[dict]:
    """Lines of move calls -> list of {"fn", "args"}; raises MoveError with the line number on anything else."""
    moves = []
    for no, line in enumerate((code or "").splitlines(), 1):
        src = line.split("#", 1)[0].strip()
        if not src:
            continue
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as e:
            raise MoveError(f"line {no}: not a call: {src}") from e
        call = tree.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id not in ("arm", "body", "hit"):
            raise MoveError(f"line {no}: only arm(...), body(...) and hit(...) are allowed: {src}")
        fn = call.func.id
        pos = [_const(a) for a in call.args]
        kw = {k.arg: _const(k.value) for k in call.keywords}
        if fn == "arm":
            if len(pos) < 2:
                raise MoveError(f"line {no}: arm(side, t, ...) needs a side and a time")
            side, t = pos[0], pos[1]
            if side not in ("L", "R", "both"):
                raise MoveError(f"line {no}: side must be 'L', 'R' or 'both'")
            bad = set(kw) - ARM_KEYS
            if bad:
                raise MoveError(f"line {no}: unknown arguments {sorted(bad)}; allowed {sorted(ARM_KEYS)}")
            if "pose" in kw and kw["pose"] not in POSES:
                raise MoveError(f"line {no}: unknown pose {kw['pose']!r}; known: {sorted(POSES)}")
            if "pose" not in kw and "raise_" not in kw:
                raise MoveError(f"line {no}: give a pose or raise_")
            moves.append({"fn": "arm", "side": side, "t": float(t), **kw})
        elif fn == "body":
            if len(pos) < 1:
                raise MoveError(f"line {no}: body(t, ...) needs a time")
            bad = set(kw) - BODY_KEYS
            if bad:
                raise MoveError(f"line {no}: unknown arguments {sorted(bad)}; allowed {sorted(BODY_KEYS)}")
            moves.append({"fn": "body", "t": float(pos[0]), **kw})
        else:
            if len(pos) < 2 or pos[1] not in HIT_MOVES:
                raise MoveError(f"line {no}: hit(t, move, size) with move one of {HIT_MOVES}")
            moves.append({"fn": "hit", "t": float(pos[0]), "move": pos[1],
                          "size": float(pos[2]) if len(pos) > 2 else float(kw.get("size", 0.6))})
    return moves


def _envelope(n: int, fps: float, t: float, hold: float, attack: float, release: float) -> np.ndarray:
    """0..1 weight: rises over `attack` before t, holds `hold` after t, falls over `release` (smooth steps)."""
    x = np.arange(n) / fps
    a = np.clip((x - (t - attack)) / max(attack, 1e-3), 0, 1)
    r = np.clip(1 - (x - (t + hold)) / max(release, 1e-3), 0, 1)
    w = np.minimum(a, r)
    return w * w * (3 - 2 * w)


def arm_directions(side: str, raise_: float, forward: float, bend: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper-arm and forearm directions (character axes: x = its left, y = away from the camera, z = up) whose
    soft-arm arc points the hand at (raise_, forward) from the shoulder and bends by `bend`, curling toward the
    head and the middle."""
    sign = 1.0 if side == "L" else -1.0
    r, fw = np.radians(raise_), np.radians(forward)
    chord = np.array([sign * np.sin(r) * np.cos(fw), -np.sin(fw) * np.sin(r), -np.cos(r)])
    chord = chord / np.linalg.norm(chord)
    toward = np.array([-sign, -0.3, 0.6])                      # the middle, a little forward, up
    m = toward - chord * (toward @ chord)
    if np.linalg.norm(m) < 1e-6:
        m = np.array([-sign, 0.0, 0.0])
    m = m / np.linalg.norm(m)
    b = np.radians(np.clip(bend, 0, 160)) / 2
    f = chord * np.cos(b) + m * np.sin(b)                      # the forearm curls toward the middle
    u = chord * np.cos(b) - m * np.sin(b)
    return u, f


def curves(moves: list[dict], n: int, fps: float) -> dict:
    """Per frame: for each arm a weight and target directions (and a fist value or NaN); body offsets; hits as
    emphasis entries (actions/plan.py format)."""
    out = {"arm": {s: {"w": np.zeros(n), "u": np.zeros((n, 3)), "f": np.zeros((n, 3)), "fist": np.full(n, np.nan)}
                   for s in "LR"},
           "body": {k: np.zeros(n) for k in ("tilt", "lean", "turn", "squash", "lift")}, "emphasis": []}
    for mv in moves:
        hold, attack, release = (float(mv.get(k, d)) for k, d in (("hold", 0.2), ("attack", 0.15), ("release", 0.25)))
        env = _envelope(n, fps, mv["t"], hold, attack, release)
        if mv["fn"] == "arm":
            p = POSES.get(mv.get("pose"), (None, None, None, None))
            rz = float(mv.get("raise_", p[0] if p[0] is not None else 90))
            fw = float(mv.get("forward", p[1] if p[1] is not None else 10))
            bd = float(mv.get("bend", p[2] if p[2] is not None else 0))
            fist = mv.get("fist", p[3])
            for s in ("L", "R") if mv["side"] == "both" else (mv["side"],):
                u, f = arm_directions(s, rz, fw, bd)
                a = out["arm"][s]
                take = env > a["w"]                             # the strongest move wins where moves overlap
                a["u"][take], a["f"][take], a["w"][take] = u, f, env[take]
                if fist is not None:
                    a["fist"][env > 0.5] = 1.0 if fist else 0.0
        elif mv["fn"] == "body":
            for k in ("tilt", "lean", "turn", "squash", "lift"):
                out["body"][k] += float(mv.get(k, 0.0)) * env
        else:
            out["emphasis"].append({"frame": int(round(mv["t"] * fps)), "move": mv["move"], "size": mv["size"]})
    return out
