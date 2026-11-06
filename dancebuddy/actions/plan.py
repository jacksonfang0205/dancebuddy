"""The agent's plan for copying a dancer onto a character: the decisions, not the keyframes.

The copy itself (retarget.py, legs.py) is deterministic. The plan sets, per move unit, how strongly each layer
is used, how each leg event is translated, and which accents get an extra emphasis move. The default plan
(every knob at its default) reproduces the plain copy exactly.

  units[]     body (x dancer's torso turn/tilt/lean), arms (x how far the arms move from their rest pose),
              groove (x beat wiggle), head (x the dancer's head nods and tilts moved onto the upper body),
              travel (x the slide across the floor)
  legs[]      per leg event (index from get_leg_events): skip it, or scale its translation
  emphasis[]  an extra move on an accent frame: squash, hop, nod, tilt_left, tilt_right, twist
  moves       the agent's own moves, as lines of code (moves.py): a pose for an arm or the body at a time
"""

from __future__ import annotations

import numpy as np

from dancebuddy.perception.beats import smooth

KNOBS = {  # name: (default, min, max, meaning)
    "body": (1.0, 0.0, 1.6, "x the dancer's torso turn / side tilt / lean (still softly limited by the character)"),
    "arms": (1.0, 0.5, 1.4, "x how far the arms move away from their rest pose (>1 exaggerates)"),
    "groove": (1.0, 0.0, 2.0, "x the beat-synced wiggle of the soft body"),
    "head": (0.0, 0.0, 1.5, "x the dancer's head nods and tilts, moved onto the upper body (the character has no neck)"),
    "travel": (1.0, 0.0, 1.5, "x how far the character slides across when the dancer moves across the floor"),
}
EMPHASIS_MOVES = {  # move: (what it does, full size)
    "squash": ("the body squashes on the accent", 0.15),
    "hop": ("a small hop landing on the accent (body heights)", 0.12),
    "nod": ("the upper body nods forward on the accent (deg)", 20.0),
    "tilt_left": ("the body tilts to the character's left on the accent (deg)", 15.0),
    "tilt_right": ("the body tilts to the character's right on the accent (deg)", 15.0),
    "twist": ("the body twists on the accent (deg)", 25.0),
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "units": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "unit_id": {"type": "integer"},
                "intent": {"type": "string", "description": "one sentence: what the dancer does here and how the character shows it"},
                **{k: {"type": "number", "minimum": v[1], "maximum": v[2], "description": v[3]} for k, v in KNOBS.items()},
            },
            "required": ["unit_id", "intent"]}},
        "legs": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "leg event index from get_leg_events"},
                "skip": {"type": "boolean"},
                "size": {"type": "number", "minimum": 0.3, "maximum": 1.6, "description": "x the default translation"},
            },
            "required": ["index"]}},
        "emphasis": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "frame": {"type": "integer", "description": "accent frame (from get_dance_score)"},
                "move": {"type": "string", "enum": list(EMPHASIS_MOVES)},
                "size": {"type": "number", "minimum": 0.1, "maximum": 1.0, "description": "fraction of the move's full size"},
            },
            "required": ["frame", "move"]}},
        "moves": {"type": "string", "description": "the agent's own moves: lines of arm(...), body(...), hit(...) "
                  "calls (see get_plan_format); they take over from the copy around their times"},
        "summary": {"type": "string"},
    },
    "required": ["units"],
}


def default_plan(score: dict) -> dict:
    return {"units": [{"unit_id": u["id"], "intent": "plain copy", **{k: v[0] for k, v in KNOBS.items()}}
                      for u in score["units"]], "legs": [], "emphasis": []}


def knob_curves(plan: dict | None, score: dict | None, n: int, fps: float) -> dict[str, np.ndarray]:
    """Per-frame value of each knob; units blend into each other over ~0.15 s."""
    out = {k: np.full(n, v[0], float) for k, v in KNOBS.items()}
    if not plan or not score:
        return out
    by_id = {u["unit_id"]: u for u in plan.get("units", [])}
    for u in score["units"]:
        p = by_id.get(u["id"], {})
        s, e = max(0, u["start"]), min(n, u["end"] + 1)
        for k, (d, lo, hi, _) in KNOBS.items():
            out[k][s:e] = float(np.clip(p.get(k, d), lo, hi))
    sig = max(1.0, 0.15 * fps / 2)
    return {k: smooth(v, sig) for k, v in out.items()}


def leg_overrides(plan: dict | None) -> dict[int, dict]:
    if not plan:
        return {}
    return {int(x["index"]): {"skip": bool(x.get("skip", False)), "size": float(np.clip(x.get("size", 1.0), 0.3, 1.6))}
            for x in plan.get("legs", [])}


def emphasis_curves(plan: dict | None, n: int, fps: float, height: float) -> dict[str, np.ndarray]:
    """Extra accent moves: quick attack landing on the accent frame, then a release."""
    out = {k: np.zeros(n) for k in ("squash", "lift", "nod", "tilt", "twist")}
    if not plan:
        return out
    t = np.arange(n)
    attack, release = max(1.0, 0.08 * fps), max(2.0, 0.25 * fps)
    for em in plan.get("emphasis", []):
        f, move = int(em["frame"]), em["move"]
        if move not in EMPHASIS_MOVES or not 0 <= f < n:
            continue
        size = float(np.clip(em.get("size", 0.6), 0.1, 1.0)) * EMPHASIS_MOVES[move][1]
        env = np.where(t <= f, np.exp(-0.5 * ((t - f) / attack) ** 2), np.exp(-(t - f) / release))
        if move == "squash":
            out["squash"] += size * env
        elif move == "hop":  # in the air before the accent, landing on it
            out["lift"] += size * np.where(t <= f, np.exp(-0.5 * ((t - (f - attack * 1.5)) / attack) ** 2), 0)
        elif move == "nod":
            out["nod"] += size * env
        elif move == "tilt_left":
            out["tilt"] += size * env
        elif move == "tilt_right":
            out["tilt"] -= size * env
        elif move == "twist":
            out["twist"] += size * env
    return out
