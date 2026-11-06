"""Fast tests of the MVP: no model calls, no Blender. Run with `uv run pytest`."""

import json
from pathlib import Path

import pytest

from dancebuddy import llm
from dancebuddy.perception import build_dance_score, load_keypoints
from dancebuddy.synth import make

ROOT = Path(__file__).resolve().parent.parent


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


def test_moves_parse_safely_and_take_over_the_arm():
    import numpy as np
    from dancebuddy.moves import MoveError, curves, parse
    for bad in ("import os", "open('x')", "arm('L', 1.0, raise_=__import__('os'))", "arm('L', 1.0, pose='fly')",
                "arm('X', 1.0, raise_=90)"):
        with pytest.raises(MoveError):
            parse(bad)
    mv = parse('arm("both", 1.0, pose="overhead", hold=0.2)  # hands up\nbody(2.0, tilt=-10)\nhit(1.0, "squash")')
    c = curves(mv, 90, 30.0)
    w = c["arm"]["L"]["w"]
    assert w[30] == 1.0 and w[0] == 0.0 and w[80] == 0.0       # in at 1 s, back to the copy later
    assert c["arm"]["L"]["u"][30][2] > 0.9                      # overhead: straight up
    assert c["arm"]["L"]["fist"][30] == 1.0
    assert abs(c["body"]["tilt"][60] + 10) < 1e-6 and c["emphasis"][0]["frame"] == 30


def test_command_line_options():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "dancebuddy", "copy-agent", "--help"], capture_output=True, text=True,
                         cwd=ROOT).stdout
    assert "--attempts" in out and "--looks" in out
    out = subprocess.run([sys.executable, "-m", "dancebuddy", "--help"], capture_output=True, text=True, cwd=ROOT).stdout
    for cmd in ("perceive", "copy", "copy-agent", "render", "compose", "demo"):
        assert cmd in out
