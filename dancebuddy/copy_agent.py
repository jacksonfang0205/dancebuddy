"""The DanceBuddy agent on the copy pipeline: an LLM decides how the dancer's moves are shown on the character.

The model reads the dance (move units, accents, the dancer's signals incl. head moves, the leg events found) and
the character's motion spec, and submits a copy plan (copy_plan.py): per unit how strongly to use the body, arms,
groove and head layers, how to translate each leg event, and which accents get an extra emphasis move. Each plan
is run through the deterministic copy (retarget.py) and checked (copy_checks.py); the model revises from the
per-unit report. The best attempt is kept. Rendering is a separate step (`render ... --systems agent_copy`).
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from . import llm
from .copy_checks import rank, run_copy_checks
from .copy_plan import EMPHASIS_MOVES, KNOBS, PLAN_SCHEMA, default_plan
from .retarget import retarget

SYSTEM = """You are DanceBuddy, an animation agent. A non-humanoid plush character copies a human dancer. A
deterministic copy already exists: the character's arms follow the dancer's arms, its body follows the dancer's
torso, every leg move is translated (the character has no legs) and the body grooves on the beat. You decide how
the dancer's moves are SHOWN on this body, unit by unit, by setting the knobs of a copy plan.

How to work:
1. Read the dance (get_dance_score, get_dance_signals, get_leg_events) and the character (get_character_spec).
   Note what each unit is about: arms, footwork, head and hair flips, turns.
2. Read the plan format (get_plan_format). Every knob defaults to the plain copy.
3. For each unit decide what matters most and how this body can show it, following the character's spec:
   - moves the body can copy: keep them, exaggerate a little if the unit is about them (a small round character
     needs bigger moves to read as strongly as a person)
   - moves it cannot copy (head flips: it has no neck; footwork: it has no legs): move them onto parts it has
     (head -> upper body via the head knob, or emphasis nods/tilts; strong steps -> bigger leg translation)
   - accents: every accent should show as a clear hit; add an emphasis move where the copy has none
   - keep the character inside the picture and within its range
4. submit_copy_plan. It returns checks per unit: missed accents, completeness (does the character move about
   as much as the dancer, relatively), frames out of the picture, range. Fix what the report points at and keep
   what works. Do not change units that pass unless you have a reason.
5. When all checks pass, or you are out of attempts, reply with a short summary of your decisions and stop."""

TOOLS = [
    llm.tool_spec("get_dance_score", "Move units (time, label, which body regions move), accents and beats, as frames and seconds.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("get_dance_signals", "The dancer's continuous signals (arm raise, torso lean/side/twist, head nod/tilt, hip bounce/sway, foot lift): meaning, typical range, and which move most in each unit.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("get_leg_events", "The leg moves found (jump, kick, step) with their index, time, side and size, and how each is translated by default.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("get_character_spec", "How this character moves: its parts and how they connect, its pose vocabulary, its range, and how human parts translate.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("get_plan_format", "The copy plan's knobs, their ranges and defaults, and the emphasis moves.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("submit_copy_plan", "Submit a copy plan. It is run through the copy and checked; returns the checks per unit.",
                  PLAN_SCHEMA),
]


def _score_view(score: dict) -> dict:
    fps = float(score["fps"])
    return {"fps": fps, "duration_s": score["duration_s"], "n_beats": len(score["beats"]),
            "accents": [{"frame": a, "time_s": round(a / fps, 2)} for a in score["accents"]],
            "units": [{k: u[k] for k in ("id", "start", "end", "start_s", "end_s", "label", "dominant_regions",
                                         "region_share", "features", "accents") if k in u} for u in score["units"]]}


LEG_DEFAULTS = {"jump": "squash to wind up, stretch and hop, squash on landing",
                "kick": "a big lean away from the kicking leg with a small hop",
                "step": "a small bob with a lean onto the stepping side"}


def run_copy_agent(kp: dict, score: dict, rig: dict, spec: dict, out_dir: str | Path, max_attempts: int = 4,
                   max_turns: int = 16, model: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    validator = Draft202012Validator(PLAN_SCHEMA)
    fps = float(score["fps"])
    from .signals import compute_signals, unit_activity
    signals = compute_signals(kp)
    base = retarget(kp, rig, signals=signals, beats=score["beats"], score=score)
    events = base["leg_events"]
    spec_view = {k: v for k, v in spec.items() if k not in ("physics", "rig")}
    attempts: list[dict] = []

    def tool(name: str, args: dict) -> str:
        if name == "get_dance_score":
            return json.dumps(_score_view(score))
        if name == "get_dance_signals":
            return json.dumps({"signals": score.get("signal_stats"), "units": unit_activity(score)})
        if name == "get_leg_events":
            return json.dumps([{"index": i, "type": e["type"], "side": e["side"], "time_s": round(e["peak"] / fps, 2),
                                "size_torso_lengths": round(e["size"], 2), "default": LEG_DEFAULTS[e["type"]]}
                               for i, e in enumerate(events)])
        if name == "get_character_spec":
            return json.dumps(spec_view)
        if name == "get_plan_format":
            return json.dumps({"unit_knobs": {k: {"default": v[0], "range": [v[1], v[2]], "meaning": v[3]}
                                              for k, v in KNOBS.items()},
                               "legs": "per leg event index: skip (bool) and size (x default translation, 0.3-1.6)",
                               "emphasis_moves": {k: v[0] for k, v in EMPHASIS_MOVES.items()},
                               "emphasis_size": "0.1-1.0 of the move's full size",
                               "default_plan": default_plan(score)})
        if name == "submit_copy_plan":
            if len(attempts) >= max_attempts:
                return json.dumps({"error": "no attempts left; stop and summarise"})
            errs = [e.message for e in validator.iter_errors(args)]
            if errs:
                return json.dumps({"error": "plan does not match the schema", "details": errs[:5]})
            curves = retarget(kp, rig, signals=signals, beats=score["beats"], plan=args, score=score)
            res = run_copy_checks(curves, score, rig, spec)
            k = len(attempts) + 1
            d = out / f"attempt_{k}"
            d.mkdir(exist_ok=True)
            (d / "plan.json").write_text(json.dumps(args, indent=1))
            (d / "curves.json").write_text(json.dumps(curves))
            (d / "checks.json").write_text(json.dumps(res, indent=1))
            attempts.append({"attempt": k, "dir": str(d), "result": res})
            print(f"  attempt {k}: passed={res['passed']} "
                  + " ".join(f"{n}={'ok' if c['passed'] else 'FAIL'}" for n, c in res["checks"].items()))
            view = {kk: v for kk, v in res.items() if kk != "all_units"}
            return json.dumps({"attempt": k, "attempts_left": max_attempts - k, **view})
        return json.dumps({"error": f"unknown tool {name}"})

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Make the character dance this dance. Start by reading the dance and the character."}]
    final_text = ""
    for _ in range(max_turns):
        resp = llm.call(messages, tools=TOOLS, model=model)
        calls = resp["tool_calls"]
        assistant = {"role": "assistant", "content": resp["text"] or None}
        if calls:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                       for c in calls]
        messages.append(assistant)
        if not calls:
            final_text = resp["text"]
            break
        for c in calls:
            a = c["arguments"]
            result = (json.dumps({"error": "arguments were not valid JSON", "details": a["_parse_error"]})
                      if "_parse_error" in a else tool(c["name"], a))
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    (out / "trace.json").write_text(json.dumps(messages, indent=1))
    if not attempts:
        raise RuntimeError("the agent never submitted a valid plan; see trace.json")
    best = max(attempts, key=lambda x: rank(x["result"]))
    for f in ("plan.json", "curves.json", "checks.json"):
        (out / f).write_text((Path(best["dir"]) / f).read_text())
    summary = {"attempts": len(attempts), "best_attempt": best["attempt"], "passed": best["result"]["passed"],
               "history": [{"attempt": x["attempt"], "passed": x["result"]["passed"],
                            "problems": sum(len(u["problems"]) for u in x["result"]["units"])} for x in attempts],
               "final_message": final_text, "model": model or llm.DEFAULT_MODEL}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"agent: {len(attempts)} attempt(s), best = {best['attempt']}, passed = {best['result']['passed']}")
    return summary
