"""Prompts and the text descriptions of the dance and the character that the agent reasons over."""

from __future__ import annotations

import json

import numpy as np

from ..characters.spec import CharacterSpec
from ..motion.plan import EVENT_SOURCES
from ..perception.features import DanceFeatures

# What each dancer signal means, so the planner can pick sources by meaning rather than by name.
SOURCE_DOCS = {
    "torso_lean_side": "degrees; torso leaning to the dancer's left (+) or right (-)",
    "torso_lean_fwd": "degrees; torso bending forward (+) or back (-)",
    "torso_twist": "degrees; shoulders twisting relative to hips",
    "head_nod": "degrees; head nodding forward (+) or back (-)",
    "head_tilt": "degrees; head tilting to the dancer's left (+) or right (-)",
    "arm_L_raise": "degrees; left upper arm elevation (0 down, 90 horizontal, 180 overhead)",
    "arm_L_forward": "degrees; left upper arm swinging forward (+) or back (-)",
    "arm_L_elbow": "degrees; left elbow bend",
    "arm_R_raise": "degrees; right upper arm elevation",
    "arm_R_forward": "degrees; right upper arm swinging forward (+) or back (-)",
    "arm_R_elbow": "degrees; right elbow bend",
    "hip_L_flex": "degrees; left thigh swinging forward (+)",
    "hip_R_flex": "degrees; right thigh swinging forward (+)",
    "knee_L_bend": "degrees; left knee bend",
    "knee_R_bend": "degrees; right knee bend",
    "bounce": "dancer torso lengths; hips moving up (+) or down (-) around their running average",
    "sway": "dancer torso lengths; hips moving toward the dancer's left (+) or right (-)",
    "foot_L_lift": "dancer torso lengths; left foot lifted off its recent lowest point",
    "foot_R_lift": "dancer torso lengths; right foot lifted off its recent lowest point",
}
EVENT_DOCS = {
    "accents": "whole-body accents: moments where a move hits or holds (the rhythm the character must show)",
    "steps": "foot plants of either foot",
    "step_L": "left foot plants",
    "step_R": "right foot plants",
    "arm_L": "left arm hits",
    "arm_R": "right arm hits",
    "arm_hits": "either arm hitting",
}

UNDERSTANDING_SYSTEM = """You are the character-understanding step of DanceBuddy, a system that makes a 3D cartoon character dance along with a human dancer in a video.

You receive renders of the character and its control interface (the channels the animation system can drive). Describe the body as an animator would: what each part can express, which human body parts it lacks or merges, and how motion of those missing parts could still be shown on the parts it has. Only refer to channels that exist in the interface."""

PLANNER_SYSTEM = """You are the motion planner of DanceBuddy. A 3D cartoon character dances beside a human dancer, and it must feel like it is dancing the same dance in the same rhythm, even when its body is not human-shaped.

You write a motion plan: for each segment of the dance, a list of mappings from dancer signals to character channels. Code then computes exact per-frame values, so you decide what should move and how strongly; timing comes from the dancer's signals.

Mapping modes:
- copy: target = gain * source + offset. Use when the source and target mean the same thing in the same units (for example arm raise to arm raise, in degrees).
- follow: target moves with the shape of the source, rescaled to about -1..1, times gain (gain is the amplitude in the target's units). Use to carry a movement onto a different body part, for example the dancer's hip sway onto a body tilt.
- pulse: a bump of height gain, width width_sec, peaking on each event of an event stream. Use to turn discrete hits into motion, for example a hop on every step or a squash on every accent. A negative gain pushes the other way.

Guidance:
- Keep what the character can copy directly, and translate what it cannot: motion of body parts the character lacks should reappear on parts it has, at the same moments.
- The dancer's accents are the rhythm. Every accent should produce a visible change on the character.
- Stay within each channel's range; contributions to one channel add up.
- For copy and follow, set width_sec to 0.3 (it is ignored). For pulse, set offset to 0.
- Segments must cover the whole dance without gaps; you may merge or split the suggested segments.
- Explain your choices briefly in rationale and in each segment's intent."""


def _round(x: float, n: int = 2) -> float:
    return round(float(x), n)


def describe_dance(features: DanceFeatures) -> dict:
    """Dance description for the planner: global summary plus per-segment activity."""
    summary = features.summary()
    t = features.times
    segments = []
    for seg in features.segments:
        mask = (t >= seg.start) & (t < seg.end)
        energy = {p: _round(features.energy[p][mask].mean()) for p in ("arms", "legs", "torso", "head")}
        segments.append({
            **seg.to_dict(),
            "mean_energy": energy,
            "accents": [_round(a) for a in features.accents.times if seg.start <= a < seg.end],
            "steps": [_round(a) for a in features.event_times("steps") if seg.start <= a < seg.end],
            "arm_hits": [_round(a) for a in features.event_times("arm_hits") if seg.start <= a < seg.end],
        })
    return {
        "duration_sec": summary["duration_sec"],
        "num_accents": summary["num_accents"],
        "segments": segments,
        "signals": {
            name: {"meaning": SOURCE_DOCS.get(name, ""), "p5_p95": summary["channel_ranges"][name]}
            for name in features.channels
        },
        "event_streams": {name: {"meaning": EVENT_DOCS[name], "count": len(features.event_times(name))} for name in EVENT_SOURCES},
    }


def planning_request(features: DanceFeatures, spec: CharacterSpec, understanding: dict | None) -> str:
    parts = [
        "Character control interface:",
        json.dumps(spec.interface_for_llm(), indent=1),
    ]
    if understanding:
        parts += ["Character understanding from the previous step:", json.dumps(understanding, indent=1)]
    parts += ["Dance description:", json.dumps(describe_dance(features), indent=1), "Write the motion plan."]
    return "\n\n".join(parts)


def understanding_request(spec: CharacterSpec, image_blocks: list[dict]) -> list[dict]:
    text = "Control interface:\n" + json.dumps(spec.interface_for_llm(), indent=1)
    if image_blocks:
        text = f"Renders of the character from {len(image_blocks)} angles are attached.\n\n" + text
    return image_blocks + [{"type": "text", "text": text}]


def revision_request(report: dict) -> str:
    return (
        "The plan was compiled, animated and checked. The verifier's report is below. "
        "Revise the plan to fix the failed checks without breaking what already works, and return the full revised plan.\n\n"
        + json.dumps(report, indent=1, default=lambda o: _round(o) if isinstance(o, (float, np.floating)) else str(o))
    )


def invalid_plan_request(errors: list[str]) -> str:
    return "The plan could not be compiled:\n- " + "\n- ".join(errors) + "\nReturn a corrected full plan."
