"""Fast tests: no model calls, no Blender. Run with `uv run pytest`."""

import json
from pathlib import Path

import pytest

from dancebuddy import agent, llm
from dancebuddy.baseline import pose_copy
from dancebuddy.checks import metrics, run_checks
from dancebuddy.perception import build_dance_score, load_keypoints
from dancebuddy.primitives import compile_plan
from dancebuddy.synth import make

ROOT = Path(__file__).resolve().parent.parent
CARD = {"name": "capsule", "root_bone": "root",
        "bones": [{"name": n} for n in ("root", "lower", "upper", "top")],
        "limits": {"lean": 35, "twist": 45, "sway": 35, "lift": 0.35, "squash": 0.35}}
PLAN = json.loads((ROOT / "examples" / "handwritten_plan_synth.json").read_text())


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = make(tmp_path_factory.mktemp("synth"))
    kp = load_keypoints(d / "keypoints.npz")
    return kp, build_dance_score(kp)


def test_accents_match_the_synthetic_beats(synth):
    _, score = synth
    truth = [round((0.5 + i * 60 / 110) * 30) for i in range(22)]
    hits = sum(any(abs(a - t) <= 3 for a in score["accents"]) for t in truth)
    assert hits / len(truth) >= 0.75
    assert len(score["units"]) == 6


def test_handwritten_plan_passes_checks(synth):
    _, score = synth
    res = run_checks(compile_plan(PLAN, score, CARD), score, CARD)
    assert res["passed"], res["checks"]


def test_pose_copy_loses_arm_sections(synth):
    kp, score = synth
    m = metrics(pose_copy(kp, CARD), score, CARD)
    assert m["idle_units"] >= 2 and m["accent_recall"] < 0.5


def test_limit_violation_is_reported(synth):
    _, score = synth
    bad = json.loads(json.dumps(PLAN))
    bad["units"][0]["primitives"][0]["amplitude"] = 80
    res = run_checks(compile_plan(bad, score, CARD), score, CARD)
    assert not res["checks"]["limits"]["passed"]


def test_agent_revises_after_failed_check(synth, tmp_path, monkeypatch):
    _, score = synth
    bad = json.loads(json.dumps(PLAN))
    bad["units"][0]["primitives"][0]["amplitude"] = 80
    script = iter([
        {"text": "", "tool_calls": [{"id": "a", "name": "get_dance_score", "arguments": {}}]},
        {"text": "", "tool_calls": [{"id": "b", "name": "submit_plan", "arguments": bad}]},
        {"text": "", "tool_calls": [{"id": "c", "name": "submit_plan", "arguments": PLAN}]},
        {"text": "done", "tool_calls": []},
    ])
    monkeypatch.setattr(llm, "call", lambda *a, **k: next(script))
    summary = agent.run_agent(score, CARD, tmp_path)
    assert summary["attempts"] == 2 and summary["passed"] and summary["best_attempt"] == 2


def test_cut_out_finds_the_character_on_a_plain_background(tmp_path):
    from dancebuddy.puppet import cut_out
    info = cut_out(ROOT / "examples" / "sample_character.png", tmp_path / "char")
    filled = sum(map(sum, info["grid"])) / (info["rows"] * info["cols"])
    assert 0.5 < filled < 0.95 and info["height_px"] > info["width_px"]


def test_follow_tracks_the_dancer_signal(synth):
    import numpy as np
    from dancebuddy.signals import normalized

    _, score = synth
    unit = score["units"][2]  # torso-twist section
    plan = {"units": [{"unit_id": 2, "intent": "copy the twist", "primitives": [
        {"primitive": "follow", "signal": "torso_twist", "channel": "twist", "bones": ["upper"], "amplitude": 30}]}]}
    curve = np.array(compile_plan(plan, score, CARD)["channels"]["upper.twist"])
    s, e = unit["start"] + 5, unit["end"] - 5
    target = normalized(score, "torso_twist")[s:e]
    assert np.corrcoef(curve[s:e], target)[0, 1] > 0.95
    assert abs(curve[:unit["start"] - 10]).max() < 1e-6  # nothing outside the unit (beyond the fade)


def test_follow_needs_a_valid_signal(synth):
    _, score = synth
    for p in ({"primitive": "follow", "channel": "twist", "bones": ["upper"], "amplitude": 20},
              {"primitive": "follow", "signal": "no_such", "channel": "twist", "bones": ["upper"], "amplitude": 20}):
        with pytest.raises(ValueError):
            compile_plan({"units": [{"unit_id": 0, "intent": "x", "primitives": [p]}]}, score, CARD)


def test_get_dance_signals_tool(synth, tmp_path, monkeypatch):
    _, score = synth
    seen = {}
    script = iter([
        {"text": "", "tool_calls": [{"id": "a", "name": "get_dance_signals", "arguments": {}}]},
        {"text": "", "tool_calls": [{"id": "b", "name": "submit_plan", "arguments": PLAN}]},
        {"text": "done", "tool_calls": []},
    ])

    def fake_call(messages, *a, **k):
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if tool_msgs and "signals" not in seen:
            seen["signals"] = json.loads(tool_msgs[0]["content"])
        return next(script)

    monkeypatch.setattr(llm, "call", fake_call)
    agent.run_agent(score, CARD, tmp_path)
    reply = seen["signals"]
    assert "torso_twist" in reply["signals"] and len(reply["units"]) == len(score["units"])
    assert "torso_twist" in reply["units"][2]["most_active_signals"]


def test_pixel_metric_finds_hits_in_rendered_frames(synth):
    import numpy as np
    from dancebuddy.video_metrics import motion_from_frames, pixel_metrics

    _, score = synth
    n, acc = score["n_frames"], score["accents"]
    # a square that darts to the other side before each accent and stops dead on it (ease-out move)
    pos, side, prev = np.zeros(n), 0.0, 0
    for a in acc:
        start = max(prev, a - 6)
        pos[prev:start] = side
        new = 50.0 - side
        t = np.linspace(0, 1, a - start + 1)
        pos[start:a + 1] = side + (new - side) * (1 - (1 - t) ** 2)
        side, prev = new, a + 1
    pos[prev:] = side
    frames = []
    for i in range(n):
        img = np.full((90, 160, 3), 255, np.uint8)
        img[30:60, 40 + int(pos[i]):70 + int(pos[i])] = 0
        frames.append(img)
    good = pixel_metrics(motion_from_frames(frames), score)
    still = pixel_metrics(motion_from_frames([frames[0]] * n), score)
    assert good["px_accent_recall"] >= 0.6
    assert still["px_accent_recall"] == 0 and still["px_hits"] == 0


def test_follow_plan_keeps_hits_in_other_units(synth):
    # continuous follow motion in units 2-3 must not hide the hops of units 4-5
    _, score = synth
    plan = json.loads((ROOT / "examples" / "handwritten_plan_synth_follow.json").read_text())
    res = run_checks(compile_plan(plan, score, CARD), score, CARD)
    assert res["passed"], res["checks"]
    assert not [u for u in res["units"] if u["unit_id"] in (4, 5)]


def test_unit_report_names_the_failing_units(synth):
    kp, score = synth
    res = run_checks(pose_copy(kp, CARD), score, CARD)
    bad = {u["unit_id"] for u in res["units"]}
    assert bad and all(u["problems"] for u in res["units"])
    assert bad <= {u["id"] for u in score["units"]}


def test_copy_raises_the_arm_the_dancer_raises(synth):
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot

    from dancebuddy.perception import L_EL, L_SH, L_WR
    from dancebuddy.retarget import retarget

    kp, _ = synth
    kp = {**kp, "world": kp["world"].copy(), "image": kp["image"].copy()}
    w, im = kp["world"], kp["image"]
    half = len(w) // 2
    for i in range(len(w)):  # y points down in both: first half arm hanging, second half arm straight up
        d = 1 if i < half else -1
        w[i, L_EL] = w[i, L_SH] + [0, 0.28 * d, 0]
        w[i, L_WR] = w[i, L_SH] + [0, 0.55 * d, 0]
        im[i, L_EL] = im[i, L_SH] + [0, 0.12 * d, 0]
        im[i, L_WR] = im[i, L_SH] + [0, 0.24 * d, 0]
    rig = json.loads((ROOT / "assets" / "plush_soft_rig.json").read_text())
    curves = retarget(kp, rig)
    assert all(len(v) == curves["n_frames"] for v in curves["channels"].values())
    q = np.array(curves["channels"]["arm_L_1.quat"])
    assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-3)
    mat = {b["name"]: Rot.from_matrix(np.array(b["matrix"])) for b in rig["bones"]}
    rot = Rot.from_quat(q[:, [1, 2, 3, 0]])
    if "arm_L_0.quat" in curves["channels"]:  # the shoulder takes part of the turn: compose it
        q0 = np.array(curves["channels"]["arm_L_0.quat"])
        sh = mat["arm_L_0"] * Rot.from_quat(q0[:, [1, 2, 3, 0]]) * mat["arm_L_0"].inv()
    else:
        sh = Rot.identity(len(q))
    direction = (sh * mat["arm_L_1"] * rot).apply([0, 1, 0])  # bone axis in armature space
    down, up = direction[half // 2], direction[half + half // 2]
    # hanging (along the side of the egg, as in the model: straight down would go through its belly) vs raised
    assert down[2] < -0.65 and up[2] > 0.5
    assert up[1] < 0  # raised arm stays in front of the hair


def test_legs_become_hops_leans_and_squash(synth):
    import numpy as np
    from dancebuddy.legs import translate
    from dancebuddy.signals import compute_signals

    kp, _ = synth
    n = len(kp["world"])
    sig = {k: np.zeros(n) for k in ("floor_lift_L", "floor_lift_R")}
    sig["floor_lift_L"][40:50] = sig["floor_lift_R"][40:50] = 0.5   # a jump
    sig["floor_lift_R"][100:115] = np.r_[np.linspace(0.05, 0.5, 8), np.linspace(0.5, 0.05, 7)]  # a right kick
    sig["floor_lift_L"][200:204] = 0.12                              # a left step
    out = translate(kp, {**compute_signals(kp), **sig})
    kinds = [(e["type"], e["side"]) for e in out["events"]]
    assert ("jump", None) in kinds and ("kick", "R") in kinds and ("step", "L") in kinds
    assert out["lift"][40:50].max() > 0.08                      # a visible hop
    assert out["squash"][51:60].max() > 0.05                    # squash on landing
    assert out["tilt"][100:115].max() > 8                       # lean away from the right leg (toward its left)


def test_default_copy_plan_is_the_plain_copy(synth):
    import numpy as np
    from dancebuddy.copy_plan import default_plan
    from dancebuddy.retarget import retarget

    kp, score = synth
    rig = json.loads((ROOT / "assets" / "plush_soft_rig.json").read_text())
    a = retarget(kp, rig, beats=score["beats"], score=score)["channels"]
    b = retarget(kp, rig, beats=score["beats"], score=score, plan=default_plan(score))["channels"]
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_copy_agent_revises_and_keeps_the_best(synth, tmp_path, monkeypatch):
    from dancebuddy.copy_agent import run_copy_agent
    from dancebuddy.copy_plan import default_plan

    kp, score = synth
    rig = json.loads((ROOT / "assets" / "plush_soft_rig.json").read_text())
    spec = json.loads((ROOT / "assets" / "plush_motion_spec.json").read_text())
    first = default_plan(score)
    second = json.loads(json.dumps(first))
    second["units"][0]["head"] = 1.0
    second["emphasis"] = [{"frame": score["accents"][0], "move": "squash", "size": 0.8}]
    seen = []
    script = iter([
        {"text": "", "tool_calls": [{"id": "a", "name": "get_dance_score", "arguments": {}},
                                    {"id": "b", "name": "get_leg_events", "arguments": {}}]},
        {"text": "", "tool_calls": [{"id": "c", "name": "submit_copy_plan", "arguments": first}]},
        {"text": "", "tool_calls": [{"id": "d", "name": "submit_copy_plan", "arguments": second}]},
        {"text": "done", "tool_calls": []},
    ])

    def fake(messages, *a, **k):
        seen.append(messages[-1])
        return next(script)

    monkeypatch.setattr(llm, "call", fake)
    summary = run_copy_agent(kp, score, rig, spec, tmp_path)
    assert summary["attempts"] == 2
    report = json.loads(seen[2]["content"])          # what the model saw after its first plan
    assert "checks" in report and "units" in report
    assert (tmp_path / "curves.json").exists() and (tmp_path / "plan.json").exists()


def test_hands_ball_up_for_raised_bent_arms():
    """Motion spec: fists for raised, bent arms; open hands when the arm hangs, reaches out or lies across."""
    import numpy as np
    from dancebuddy.retarget import elbow_amount, fist_amount
    out, up, down, across = map(np.array, ((1.0, 0, 0), (0, 0, 1.0), (0, 0, -1.0), (-1.0, 0, 0)))
    assert fist_amount(out, up) > 0.95                 # upper arm out, forearm up: a soft elbow and a fist
    assert elbow_amount(out, up) == 1.0
    assert fist_amount(down, down) == 0.0              # hanging
    assert fist_amount(out, out) == 0.0                # reaching out
    assert fist_amount(out, across) == 0.0             # across the chest
    assert 0.4 < fist_amount(up, up) < 0.8             # straight up: partly closed
    assert elbow_amount(out, (0.95, 0, 0.3)) == 0.0    # a small bend stays one curve


def test_soft_arm_is_a_round_arc_that_keeps_its_length():
    """An arm bends as one circular arc (no crease) and its arc keeps the arm's length."""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from dancebuddy.retarget import arm_centre_line, arm_pose
    M = {"arm_L_1": np.eye(3), "arm_L_1__head": [0.0, 0.0, 0.0], "arm_L_1__tail": [0.0, 0.5, 0.0]}
    info = {}
    q, curve, chord = arm_pose(M, "arm_L_1", np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 1, info=info)
    assert info["theta"] > 1.0 and 0.6 < chord < 1.0
    pts = arm_centre_line(np.zeros(3), info, np.linspace(0, 1, 200))
    assert abs(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() - 0.5) < 0.005
    seg = np.diff(pts, axis=0)
    turn = np.degrees(np.arccos(np.clip((seg[1:] * seg[:-1]).sum(1) / np.linalg.norm(seg[1:], axis=1)
                                        / np.linalg.norm(seg[:-1], axis=1), -1, 1)))
    assert turn.max() - turn.min() < 0.2          # the same curvature all along
    assert abs(np.linalg.norm(pts[-1]) - 0.5 * chord) < 1e-3
    assert Rot.from_quat(np.r_[q[1:], q[:1]]).magnitude() > 0.1


def test_inspect_picks_a_units_key_frames():
    from dancebuddy.inspect_tools import pick_frames
    score = {"units": [{"id": 0, "start": 0, "end": 40, "accents": [10], "beats": [5, 15, 25, 35]},
                       {"id": 1, "start": 41, "end": 90, "accents": [60, 70], "beats": [45, 55, 65, 75, 85]}]}
    f = pick_frames(score, 91, unit_id=1)
    assert f[0] == 41 and f[-1] == 90 and 60 in f and 70 in f and len(f) <= 8
    assert pick_frames(score, 91, frames=[5, 200, 5, 3]) == [3, 5]
