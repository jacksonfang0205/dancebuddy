"""The systems an evaluation compares, all on the same dance and the same character. Each writes
<run>/<system>/curves.json (and, for the LLM systems, summary.json with its calls and cost).

  pose_copy        no LLM: the dancer's torso angles copied straight onto the character's bones
  plain_copy       no LLM: the DanceBuddy copy (perception, leg translation, retarget) with no plan
  planner          LLM, no copying: plans primitive moves per unit from a description of the body (the first
                   prototype's agent), compiled to curves, checked and revised
  agent            the DanceBuddy agent with every tool
  agent_no_looks   ablation: no looking at rendered frames
  agent_no_moves   ablation: knobs only, no own moves
  agent_numbers    ablation: numbers only (no looks, no geometry check, no own moves)
"""

from __future__ import annotations

import json
from pathlib import Path

from dancebuddy.knowledge import character

LLM_SYSTEMS = {"planner", "agent", "agent_no_looks", "agent_no_moves", "agent_numbers"}
ABLATIONS = {"agent": {}, "agent_no_looks": {"use_looks": False}, "agent_no_moves": {"use_moves": False},
             "agent_numbers": {"use_looks": False, "use_geometry": False, "use_moves": False}}


def _load(run_dir: Path):
    from dancebuddy.perception.video import load_keypoints
    return load_keypoints(run_dir / "keypoints.npz"), json.loads((run_dir / "dance_score.json").read_text())


def pose_copy(run_dir: Path, char: Path, out: Path, **_) -> None:
    from evaluation.baselines.pose_copy import pose_copy as copy
    kp, _score = _load(run_dir)
    (out / "curves.json").write_text(json.dumps(copy(kp, character.card(char))))


def plain_copy(run_dir: Path, char: Path, out: Path, **_) -> None:
    from dancebuddy.executor.retarget import retarget
    kp, score = _load(run_dir)
    (out / "curves.json").write_text(json.dumps(retarget(kp, character.rig(char), beats=score["beats"])))


def planner(run_dir: Path, char: Path, out: Path, attempts: int = 4, model: str | None = None, **_) -> None:
    from evaluation.baselines.planner.agent import run_agent
    _kp, score = _load(run_dir)
    run_agent(score, character.card(char), out, max_attempts=attempts, model=model)


def agent(run_dir: Path, char: Path, out: Path, attempts: int = 4, looks: int = 4, model: str | None = None,
          system: str = "agent", **_) -> None:
    from dancebuddy.planner.loop import run_agent
    kp, score = _load(run_dir)
    src = (run_dir / "source.txt").read_text().strip() if (run_dir / "source.txt").exists() else score.get("source")
    run_agent(kp, score, character.rig(char), character.spec(char), out, max_attempts=attempts, model=model,
              blend=character.blend(char), source=src if src and Path(src).exists() else None, max_looks=looks,
              **ABLATIONS[system])


SYSTEMS = {"pose_copy": pose_copy, "plain_copy": plain_copy, "planner": planner,
           **{name: agent for name in ABLATIONS}}


def run_system(name: str, run_dir: Path, char: Path, **kw) -> Path:
    out = Path(run_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    SYSTEMS[name](Path(run_dir), Path(char), out, system=name, **kw)
    return out
