"""The DanceBuddy agent: understand the character, plan the motion, compile it, verify it, and revise.

Code orchestrates the loop; the LLM makes the judgment calls (what the body can express, how each move translates).
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..characters.spec import CharacterSpec
from ..motion.plan import CharacterMotion, MotionPlan, compile_plan
from ..perception.features import DanceFeatures
from . import prompts, schemas
from .llm import LLM, Conversation, LLMError
from .verifier import VerificationReport, verify


@dataclass
class AgentResult:
    plan: MotionPlan
    motion: CharacterMotion
    report: VerificationReport
    understanding: dict | None
    trace: list[dict] = field(default_factory=list)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps(self.plan.to_dict(), indent=1))
        self.motion.save(out / "motion.json")
        (out / "verification.json").write_text(json.dumps(self.report.to_dict(), indent=1))
        if self.understanding is not None:
            (out / "understanding.json").write_text(json.dumps(self.understanding, indent=1))
        (out / "trace.json").write_text(json.dumps(self.trace, indent=1, default=str))


def image_blocks(paths: list[str | Path]) -> list[dict]:
    blocks = []
    for p in paths:
        data = base64.standard_b64encode(Path(p).read_bytes()).decode("utf-8")
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    return blocks


class DanceBuddyAgent:
    def __init__(self, llm: LLM, max_rounds: int = 3, use_understanding: bool = True, use_verification: bool = True):
        self.llm = llm
        self.max_rounds = max_rounds
        self.use_understanding = use_understanding
        self.use_verification = use_verification  # off = single-pass ablation

    def understand(self, spec: CharacterSpec, views: list[str | Path], trace: list) -> dict:
        convo = Conversation(system=prompts.UNDERSTANDING_SYSTEM)
        result = self.llm.ask_json(convo, prompts.understanding_request(spec, image_blocks(views)), schemas.CHARACTER_UNDERSTANDING)
        trace.append({"step": "understand", "output": result})
        return result

    def run(self, features: DanceFeatures, spec: CharacterSpec, views: list[str | Path] | None = None) -> AgentResult:
        trace: list[dict] = []
        understanding = self.understand(spec, views or [], trace) if self.use_understanding else None

        convo = Conversation(system=prompts.PLANNER_SYSTEM)
        request = prompts.planning_request(features, spec, understanding)
        best: tuple[MotionPlan, CharacterMotion, VerificationReport] | None = None
        rounds = self.max_rounds if self.use_verification else 1
        for round_idx in range(rounds):
            t0 = time.time()
            raw = self.llm.ask_json(convo, request, schemas.MOTION_PLAN)
            plan = MotionPlan.from_dict(raw)
            errors = plan.validate(spec, features)
            if errors:  # an uncompilable plan goes straight back with the errors; it still costs a round
                trace.append({"step": "plan", "round": round_idx, "plan": raw, "errors": errors})
                request = prompts.invalid_plan_request(errors)
                continue
            motion = compile_plan(plan, spec, features)
            report = verify(features, motion, spec, plan)
            trace.append({
                "step": "plan",
                "round": round_idx,
                "seconds": round(time.time() - t0, 1),
                "plan": raw,
                "verification": report.to_dict(),
            })
            if best is None or report.score > best[2].score:
                best = (plan, motion, report)
            if report.passed or not self.use_verification:
                break
            request = prompts.revision_request(report.to_dict())

        if best is None:
            raise LLMError(f"no compilable plan after {rounds} rounds")
        trace.append({"step": "usage", "calls": getattr(self.llm, "usage", [])})
        return AgentResult(plan=best[0], motion=best[1], report=best[2], understanding=understanding, trace=trace)
