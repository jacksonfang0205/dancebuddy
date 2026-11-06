"""The DanceBuddy agent on the copy pipeline: an LLM decides how the dancer's moves are shown on the character.

The model reads the dance (move units, accents, the dancer's signals incl. head moves, the leg events found) and
the character's motion spec, and submits a copy plan (copy_plan.py): per unit how strongly to use the body, arms,
groove and head layers, how to translate each leg event, and which accents get an extra emphasis move. Each plan
is run through the deterministic copy (retarget.py) and checked (copy_checks.py); the model revises from the
per-unit report. Beyond the numbers it can check the arms' geometry on the posed character and look at a few
rendered frames next to the dancer (inspect_tools.py), choosing what to check and where. The best attempt is kept.
Rendering the whole dance is a separate step (`render ... --systems agent_copy`).
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from . import llm
from .copy_checks import rank, run_copy_checks
from .copy_plan import EMPHASIS_MOVES, KNOBS, PLAN_SCHEMA, default_plan
from .inspect_tools import check_arm_geometry, look_at_frames, pick_frames
from . import memory as memory_mod
from . import moves as moves_mod
from .retarget import retarget

SYSTEM = """You are DanceBuddy, an animation agent. A non-humanoid plush character copies a human dancer. A
deterministic copy already exists: the character's arms follow the dancer's arms, its body follows the dancer's
torso, every leg move is translated (the character has no legs) and the body grooves on the beat. You decide how
the dancer's moves are SHOWN on this body, unit by unit, by setting the knobs of a copy plan.

How to work:
0. Read your memory of this character (get_memory): lessons from your earlier runs (what worked, what failed,
   tips) and their recent results. Use them: start from what worked, avoid what failed.
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
5. The numbers do not show what the character looks like. Check the latest attempt where it matters, cheapest
   first, and decide yourself where to look:
   - check_arm_geometry (a few seconds): arms folding or sinking into the body, on chosen frames or a unit
   - look_at_frames (about half a minute; limited number of looks): renders chosen frames small next to the
     dancer and returns a reviewer's notes per frame: does the character show the move, what looks wrong, a fix.
     Look at units with big arm moves, turns or accents, and at units the checks flagged.
   Turn what you learn into changes and submit again. Knobs scale whole units; for a particular moment write a
   move in the plan's `moves` (code lines, see get_plan_format): e.g. arm("both", 3.5, pose="overhead") when the
   hands should go above the head, arm("both", 0.7, pose="hands_together_chest"), body(2.7, tilt=-12). A move
   takes over from the copy around its time and blends back, so write moves only where the copy does not read,
   time them on the dancer's moment (seconds), and look again to confirm they worked.
6. When the checks pass and what you looked at is fine, or you are out of attempts: call remember once with the
   lessons of this run that will help next time with this character (short, general, not tied to this dance's
   timing: e.g. "works: pose overhead with fist for raised accents", "fails: knob arms > 1.3 pushes hands out of the
   picture"), and, only if what you saw shows the motion spec is wrong or missing something, one spec_proposal
   (a person reviews it). Then reply with a short summary of your decisions, what you checked and what is still
   wrong, and stop."""

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
    llm.tool_spec("check_arm_geometry", "Check the arms on the posed character (no rendering, a few seconds): where an "
                  "arm folds or sinks into the body. On the latest attempt unless `attempt` is given; on `frames` "
                  "(at most 24) or on a unit's key frames.",
                  {"type": "object", "properties": {"attempt": {"type": "integer"}, "unit_id": {"type": "integer"},
                                                    "frames": {"type": "array", "items": {"type": "integer"}}}}),
    llm.tool_spec("get_memory", "Your memory of this character: lessons from earlier runs, their recent results, "
                  "and spec changes you proposed that are still waiting for a person.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("remember", "Save this run's lessons for the next runs with this character, and optionally "
                  "propose a change to its motion spec (a person approves it).",
                  {"type": "object", "properties": {
                      "lessons": {"type": "array", "items": {"type": "object", "properties": {
                          "kind": {"type": "string", "enum": list(memory_mod.KINDS)},
                          "text": {"type": "string"}}, "required": ["kind", "text"]}},
                      "spec_proposal": {"type": "object", "properties": {
                          "change": {"type": "string"}, "why": {"type": "string"}}}},
                   "required": ["lessons"]}),
    llm.tool_spec("look_at_frames", "Render up to 8 frames of an attempt small, next to the dancer at the same "
                  "moments, and get a reviewer's notes per frame (copy 0-2, problems, a fix). About half a minute; "
                  "limited number of looks per run. On the latest attempt unless `attempt` is given; on `frames` or on "
                  "a unit's accents and beats.",
                  {"type": "object", "properties": {"attempt": {"type": "integer"}, "unit_id": {"type": "integer"},
                                                    "frames": {"type": "array", "items": {"type": "integer"}}}}),
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
                   max_turns: int = 24, model: str | None = None, blend: str | None = None,
                   source: str | None = None, max_looks: int = 4, memory_path: Path | None = None) -> dict:
    """blend: the rigged character (for the geometry check and the looks); source: the dance video (for the
    looks). Without them the agent only has the numeric checks."""
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
    looks: list[dict] = []
    mem = memory_mod.load(memory_path) if memory_path else None
    remembered: list[dict] = []

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
                               "moves": {"how": moves_mod.__doc__.split("\n\n", 1)[1],
                                         "poses": {k: dict(zip(("raise_", "forward", "bend", "fist"), v))
                                                   for k, v in moves_mod.POSES.items()}},
                               "default_plan": default_plan(score)})
        if name == "submit_copy_plan":
            if len(attempts) >= max_attempts:
                return json.dumps({"error": "no attempts left; stop and summarise"})
            errs = [e.message for e in validator.iter_errors(args)]
            if errs:
                return json.dumps({"error": "plan does not match the schema", "details": errs[:5]})
            try:
                moves_mod.parse(args.get("moves", ""))
            except moves_mod.MoveError as e:
                return json.dumps({"error": "moves do not parse", "details": str(e)})
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
        if name == "get_memory":
            return json.dumps(memory_mod.view(mem) if mem is not None else {"note": "no memory in this run"})
        if name == "remember":
            if mem is None:
                return json.dumps({"note": "no memory in this run"})
            res = memory_mod.remember(mem, args.get("lessons", []), args.get("spec_proposal"), out.name)
            remembered.append(args)
            return json.dumps(res)
        if name in ("check_arm_geometry", "look_at_frames"):
            if not attempts:
                return json.dumps({"error": "submit a plan first"})
            if not blend or (name == "look_at_frames" and not source):
                return json.dumps({"error": "this check is not available in this run"})
            k = int(args.get("attempt") or len(attempts))
            att = next((x for x in attempts if x["attempt"] == k), attempts[-1])
            n = len(kp["image"])
            work = Path(att["dir"]) / ("geometry" if name == "check_arm_geometry" else f"look_{len(looks) + 1}")
            if name == "check_arm_geometry":
                frames = pick_frames(score, n, args.get("frames"), args.get("unit_id"), most=24)
                if not args.get("frames") and args.get("unit_id") is None:
                    frames = list(range(0, n, max(1, n // 24)))[:24]
                res = check_arm_geometry(Path(att["dir"]) / "curves.json", blend, score, frames, work)
                print(f"  geometry on attempt {att['attempt']}: {len(res['problems'])} problems in {len(frames)} frames")
                return json.dumps({"attempt": att["attempt"], **res})
            if len(looks) >= max_looks:
                return json.dumps({"error": "no looks left; decide from what you have"})
            frames = pick_frames(score, n, args.get("frames"), args.get("unit_id"), most=8)
            res = look_at_frames(Path(att["dir"]) / "curves.json", blend, score, spec, source, frames, work, model)
            looks.append({"attempt": att["attempt"], **res})
            print(f"  look {len(looks)} at attempt {att['attempt']}, frames {frames}")
            return json.dumps({"attempt": att["attempt"], "looks_left": max_looks - len(looks), "frames": frames,
                               "notes": res["notes"]})
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
               "looks": [{"attempt": x["attempt"], "frames": x["frames"], "picture": x["picture"]} for x in looks],
               "final_message": final_text, "model": model or llm.DEFAULT_MODEL}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if mem is not None:  # what this run did, recorded whether or not the agent wrote lessons
        best_plan = json.loads((Path(best["dir"]) / "plan.json").read_text())
        last_look = looks[-1]["notes"].get("overall") if looks and isinstance(looks[-1].get("notes"), dict) else None
        mem["runs"].append({"run": str(out), "passed": best["result"]["passed"],
                            "accents": best["result"]["checks"]["accents"]["value"],
                            "attempts": len(attempts), "looks": len(looks),
                            "moves_written": len([x for x in best_plan.get("moves", "").splitlines() if x.strip()]),
                            "moves": best_plan.get("moves", ""), "reviewer": last_look,
                            "wrote_lessons": bool(remembered)})
        memory_mod.save(memory_path, mem)
    print(f"agent: {len(attempts)} attempt(s), best = {best['attempt']}, passed = {best['result']['passed']}")
    return summary
