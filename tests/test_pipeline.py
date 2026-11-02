"""Fast tests for planning, compiling, metrics and the agent loop. No Blender, MediaPipe or API key needed."""

import numpy as np
import pytest

from dancebuddy.agent.agent import DanceBuddyAgent
from dancebuddy.agent.llm import ScriptedLLM
from dancebuddy.characters.spec import CharacterSpec, available_characters
from dancebuddy.eval.metrics import accent_scores, evaluate
from dancebuddy.motion.plan import Mapping, MotionPlan, compile_plan, whole_clip
from dancebuddy.perception.features import Accents, DanceFeatures, Segment, detect_accents
from dancebuddy.planners.baselines import direct_copy_plan, fixed_rules_plan

FPS = 30.0


def synthetic_features(seconds: float = 6.0) -> DanceFeatures:
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    beats = np.arange(0.5, seconds - 0.25, 0.5)
    channels = {
        "torso_lean_side": 10 * np.sin(2 * np.pi * t),
        "torso_lean_fwd": 5 * np.sin(np.pi * t),
        "torso_twist": 15 * np.sin(np.pi * t / 2),
        "head_nod": 10 * np.sin(2 * np.pi * t),
        "head_tilt": np.zeros(n),
        "bounce": 0.1 * np.sin(4 * np.pi * t),
        "sway": 0.2 * np.sin(np.pi * t),
    }
    for side in ("L", "R"):
        channels[f"arm_{side}_raise"] = 80 + 60 * np.sin(2 * np.pi * t)
        channels[f"arm_{side}_forward"] = np.zeros(n)
        channels[f"arm_{side}_elbow"] = np.full(n, 20.0)
        channels[f"hip_{side}_flex"] = 10 * np.abs(np.sin(2 * np.pi * t))
        channels[f"knee_{side}_bend"] = np.full(n, 10.0)
        channels[f"foot_{side}_lift"] = np.zeros(n)
    speed = np.abs(np.sin(2 * np.pi * t)) * 2  # minima every 0.5 s
    energy = {k: speed.copy() for k in ("arms", "legs", "torso", "head", "step_L", "step_R", "arm_L", "arm_R", "total")}
    accents = Accents(beats, np.ones(len(beats)))
    part = {k: Accents(beats, np.ones(len(beats))) for k in ("step_L", "step_R", "arm_L", "arm_R", "arm_hits")}
    return DanceFeatures(FPS, n, channels, energy, accents, part, [Segment(0, seconds, "legs")])


def test_specs_load():
    assert {"doudou", "capsule", "humanoid"} <= set(available_characters())
    spec = CharacterSpec.load("doudou")
    assert "legs" in spec.parts_missing and "squash" in spec.channels


def test_detect_accents_finds_speed_minima():
    t = np.arange(int(4 * FPS)) / FPS
    acc = detect_accents(np.abs(np.sin(2 * np.pi * t)) * 2, FPS)
    assert len(acc) >= 6
    assert np.all(np.min(np.abs(acc.times[:, None] - np.arange(0.5, 4, 0.5)[None]), axis=1) < 0.05)


def test_pulse_peaks_on_events_and_respects_range():
    f = synthetic_features()
    spec = CharacterSpec.load("doudou")
    plan = whole_clip(f, [Mapping("root_bounce", "steps", "pulse", gain=1.0, width_sec=0.3)])
    motion = compile_plan(plan, spec, f)
    bounce = motion.channels["root_bounce"]
    assert bounce.max() <= spec.channels["root_bounce"].max + 1e-9
    assert motion.clip_fraction["root_bounce"] > 0  # gain 1.0 exceeds the 0.35 limit, so it clips
    peak_frame = int(round(1.0 * FPS))
    assert bounce[peak_frame] == pytest.approx(bounce.max(), abs=1e-6)


def test_invalid_plan_reports_errors():
    f = synthetic_features()
    spec = CharacterSpec.load("doudou")
    plan = whole_clip(f, [Mapping("leg_L_lift", "hip_L_flex", "copy"), Mapping("squash", "bounce", "pulse")])
    errors = plan.validate(spec, f)
    assert any("unknown target channel" in e for e in errors)
    assert any("pulse source must be an event stream" in e for e in errors)


def test_accent_scores_perfect_and_empty():
    ref = np.array([1.0, 2.0, 3.0])
    assert accent_scores(ref, ref)["coverage"] == 1.0
    assert accent_scores(ref, np.array([]))["coverage"] == 0.0


def test_baselines_compile_for_every_character():
    f = synthetic_features()
    for name in available_characters():
        spec = CharacterSpec.load(name)
        for plan in (direct_copy_plan(spec, f), fixed_rules_plan(spec, f)):
            motion = compile_plan(plan, spec, f)
            metrics = evaluate(f, motion, spec)
            assert 0.0 <= metrics["accent_coverage"] <= 1.0


def test_agent_loop_revises_and_keeps_best():
    f = synthetic_features()
    spec = CharacterSpec.load("doudou")
    understanding = {
        "body_summary": "egg", "parts_present": ["body"], "parts_missing_or_merged": ["legs"],
        "expressive_channels": [], "translation_ideas": ["steps become hops"],
    }
    weak = MotionPlan.from_dict({"segments": [{"start": 0, "end": 7, "label": "all", "intent": "", "mappings": []}]})
    bad = {"rationale": "", "segments": [{"start": 0, "end": 7, "label": "all", "intent": "",
                                          "mappings": [{"target": "legs", "source": "bounce", "mode": "copy",
                                                        "gain": 1, "offset": 0, "width_sec": 0.3}]}]}
    good = direct_copy_plan(spec, f).to_dict()
    llm = ScriptedLLM([understanding, weak.to_dict(), bad, good])
    result = DanceBuddyAgent(llm, max_rounds=3).run(f, spec)
    steps = [s["step"] for s in result.trace]
    assert steps[0] == "understand" and steps.count("plan") == 3
    assert any("errors" in s for s in result.trace)  # the invalid plan was sent back, not compiled
    assert result.report.score >= 0.0
    assert len(result.plan.segments[0].mappings) > 0  # the best plan, not the empty one
