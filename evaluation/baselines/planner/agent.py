"""The DanceBuddy agent: plan -> check -> revise, with LLM tool calling (via OpenRouter).

The model (any OpenRouter model with tool calling) gets three read tools (dance score, character card, primitive
library) and one action tool, submit_plan. Each submission is validated,
compiled to curves, and scored by the code-based checks; the check results go
back to the model, which revises until everything passes or it runs out of
attempts. Every attempt (plan, curves, checks) is saved, plus the full trace.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from dancebuddy import llm
from evaluation.baselines.planner.checks import run_checks
from evaluation.baselines.planner.primitives import FOLLOW_CHANNELS, PRIMITIVES, compile_plan
from dancebuddy.perception.signals import SIGNAL_NAMES, unit_activity

SYSTEM = """You are DanceBuddy, an animation agent. You make a non-humanoid 3D character dance along with a human
dancer, in the same rhythm, by writing a motion plan.

How to work:
1. Read the dance score (move units, their accents and which body regions move) and the character card
   (what body the character has and which human parts it is missing).
2. Read the primitive library and the dancer's continuous signals (which signals move most in each unit).
3. For every move unit, decide how that move should be expressed on THIS body while keeping its rhythm and intent.
   Copy what the body can do: if the character has a part the dancer is moving (an arm, the torso), use `follow` so
   that part tracks the dancer's signal and the move keeps its shape. Translate what it cannot do: the character
   usually lacks arms or legs, so e.g. footwork can become bounces, arm hits can become nods or tilts of the upper
   body, a hip twist can become a twist of the lower bones. Put the strongest motion on the accents. Vary the moves
   between units the way the dancer does.
4. Call submit_plan. It returns automatic checks: accent_recall (did the character hit the dancer's accents),
   beat_precision (are the character's hits on the beat, not in between), limits, idle_units, plus a per-unit report
   of what went wrong where (missed accents, off-beat hits, units where the character barely moves).
5. If a check fails, read the details (missed accent frames, off-beat hits, limit violations) and submit a revised
   plan. Change what the check points at; keep what works.
6. When all checks pass, or you are out of attempts, reply with a short summary of your plan and stop.

Each unit needs at least one primitive. Layering two primitives in a unit is fine (e.g. a bounce on every accent plus
a slow sway), but too many hits that are not on accents will lower beat_precision."""

PLAN_SCHEMA = {
    "type": "object",
    "required": ["units"],
    "properties": {
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["unit_id", "intent", "primitives"],
                "properties": {
                    "unit_id": {"type": "integer"},
                    "intent": {"type": "string", "description": "what the dancer does and how you express it on this body"},
                    "primitives": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["primitive", "amplitude"],
                            "properties": {
                                "primitive": {"type": "string", "enum": sorted(PRIMITIVES)},
                                "bones": {"type": "array", "items": {"type": "string"},
                                          "description": "bones to move; ignored for root-only primitives"},
                                "amplitude": {"type": "number"},
                                "accents": {
                                    "description": "which of the unit's accent frames to hit: 'all', 'every_other', 'first', 'last', or an explicit list of frames",
                                    "anyOf": [{"type": "string", "enum": ["all", "every_other", "first", "last"]},
                                              {"type": "array", "items": {"type": "integer"}}],
                                },
                                "signal": {"type": "string", "enum": SIGNAL_NAMES,
                                           "description": "follow only: the dancer signal to track"},
                                "channel": {"type": "string", "enum": list(FOLLOW_CHANNELS),
                                            "description": "follow only: which rotation of the bones tracks the signal"},
                                "start": {"type": "integer", "description": "optional start frame (defaults to unit start)"},
                                "end": {"type": "integer", "description": "optional end frame (defaults to unit end)"},
                            },
                        },
                    },
                },
            },
        }
    },
}

TOOLS = [
    llm.tool_spec("get_dance_score", "The dancer's analysed motion: fps, frame count, beats, accents, and move units with their accents, dominant body regions and features.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("get_character_card", "The character's rig (bones, channels, limits) and a description of its body: visible parts, missing human parts, suggested substitutions.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("list_primitives", "The motion primitives a plan can use, with their channel, amplitude units and sensible range.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("get_dance_signals", "The dancer's continuous signals (arm raise, torso lean/side/twist, head nod/tilt, hip bounce/sway, foot lift): what each means, its typical range, and which signals move most in each unit. Use them with the follow primitive.",
                  {"type": "object", "properties": {}}),
    llm.tool_spec("submit_plan", "Submit a full motion plan (all units). It is compiled to animation curves and checked; returns the check results.",
                  PLAN_SCHEMA),
]


def _score_view(score: dict) -> dict:
    return {k: score[k] for k in ("fps", "n_frames", "duration_s", "beats", "accents", "units")}


def _rank(res: dict) -> tuple:
    c = res["checks"]
    return (res["passed"], sum(x["passed"] for x in c.values()),
            c["accent_recall"]["value"] + c["beat_precision"]["value"])


def run_agent(score: dict, card: dict, out_dir: str | Path, max_attempts: int = 4, max_turns: int = 14,
              use_card_description: bool = True, model: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    validator = Draft202012Validator(PLAN_SCHEMA)
    card_view = dict(card)
    if not use_card_description:  # ablation: the planner only sees bone names and limits
        card_view["description"] = None
    attempts: list[dict] = []

    def tool(name: str, args: dict) -> str:
        if name == "get_dance_score":
            return json.dumps(_score_view(score))
        if name == "get_character_card":
            return json.dumps(card_view)
        if name == "list_primitives":
            return json.dumps(PRIMITIVES)
        if name == "get_dance_signals":
            if "signals" not in score:
                return json.dumps({"error": "this dance score has no signals; the follow primitive is unavailable"})
            return json.dumps({"signals": score["signal_stats"], "units": unit_activity(score)})
        if name == "submit_plan":
            if len(attempts) >= max_attempts:
                return json.dumps({"error": "no attempts left; stop and summarise"})
            errs = [e.message for e in validator.iter_errors(args)]
            if errs:
                return json.dumps({"error": "plan does not match the schema", "details": errs[:5]})
            try:
                curves = compile_plan(args, score, card)
            except ValueError as e:
                return json.dumps({"error": str(e)})
            res = run_checks(curves, score, card)
            k = len(attempts) + 1
            d = out / f"attempt_{k}"
            d.mkdir(exist_ok=True)
            (d / "plan.json").write_text(json.dumps(args, indent=1))
            (d / "curves.json").write_text(json.dumps(curves))
            (d / "checks.json").write_text(json.dumps(res, indent=1))
            attempts.append({"attempt": k, "dir": str(d), "result": res})
            print(f"  attempt {k}: passed={res['passed']} "
                  + " ".join(f"{n}={'ok' if c['passed'] else 'FAIL'}" for n, c in res["checks"].items()))
            return json.dumps({"attempt": k, "attempts_left": max_attempts - k, **res})
        return json.dumps({"error": f"unknown tool {name}"})

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Make the character dance along with the dancer. Start by reading the dance score and the character card."},
    ]
    final_text = ""
    for turn in range(max_turns):
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
            args = c["arguments"]
            result = (json.dumps({"error": "arguments were not valid JSON", "details": args["_parse_error"]})
                      if "_parse_error" in args else tool(c["name"], args))
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    (out / "trace.json").write_text(json.dumps(messages, indent=1))
    if not attempts:
        raise RuntimeError("the agent never submitted a valid plan; see trace.json")
    best = max(attempts, key=lambda a: _rank(a["result"]))
    best_dir = Path(best["dir"])
    for f in ("plan.json", "curves.json", "checks.json"):
        (out / f).write_text((best_dir / f).read_text())
    summary = {
        "attempts": len(attempts),
        "best_attempt": best["attempt"],
        "passed": best["result"]["passed"],
        "history": [{"attempt": a["attempt"], "passed": a["result"]["passed"],
                     **{n: c.get("value", c["passed"]) for n, c in a["result"]["checks"].items()}} for a in attempts],
        "final_message": final_text,
        "model": model or llm.DEFAULT_MODEL,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"agent: {len(attempts)} attempt(s), best = {best['attempt']}, passed = {best['result']['passed']}")
    return summary
