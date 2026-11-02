"""Checks a compiled motion and turns failures into concrete feedback for the planner.

The checks the agent optimizes against are deliberately a subset of what the final evaluation reports; human ratings
remain the primary result so the agent is not graded only by its own signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..characters.spec import CharacterSpec
from ..eval.metrics import DEFAULT_TOLERANCE_SEC, character_accents, character_speed, evaluate, match_events
from ..motion.plan import CharacterMotion, MotionPlan
from ..perception.features import DanceFeatures, smooth

THRESHOLDS = {
    "accent_coverage": ("min", 0.7),
    "accent_precision": ("min", 0.4),
    "idle_while_dancer_moves": ("max", 0.25),
    "clip_rate": ("max", 0.15),
}


@dataclass
class VerificationReport:
    passed: bool
    score: float
    metrics: dict
    failed_checks: list[str]
    segment_issues: list[dict] = field(default_factory=list)
    channel_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": round(self.score, 3),
            "metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.metrics.items()},
            "thresholds": {k: f"{op} {v}" for k, (op, v) in THRESHOLDS.items()},
            "failed_checks": self.failed_checks,
            "segment_issues": self.segment_issues,
            "channel_issues": self.channel_issues,
        }


def _score(m: dict) -> float:
    """Single number used to keep the best plan across revision rounds."""
    return (
        m["accent_coverage"]
        + 0.5 * m["accent_precision"]
        - 0.5 * m["idle_while_dancer_moves"]
        - 0.5 * m["clip_rate"]
    )


def verify(features: DanceFeatures, motion: CharacterMotion, spec: CharacterSpec, plan: MotionPlan) -> VerificationReport:
    m = evaluate(features, motion, spec)
    failed = []
    for name, (op, limit) in THRESHOLDS.items():
        value = m[name]
        if (op == "min" and not value >= limit) or (op == "max" and not value <= limit):
            failed.append(f"{name} = {value:.2f}, needs {op} {limit}")

    char_acc = character_accents(motion, spec).times
    matched = {r for r, _ in match_events(features.accents.times, char_acc, DEFAULT_TOLERANCE_SEC)}
    speed = smooth(character_speed(motion, spec), motion.fps, 0.5)
    t = np.arange(motion.num_frames) / motion.fps
    still = speed < 0.1 * (np.percentile(speed, 95) + 1e-9)

    segment_issues = []
    for i, seg in enumerate(plan.segments):
        in_seg = [a for a in features.accents.times if seg.start <= a < seg.end]
        missed = [round(float(a), 2) for a in in_seg if a not in matched]
        mask = (t >= seg.start) & (t < seg.end)
        idle = float(still[mask].mean()) if mask.any() else 0.0
        if missed or idle > 0.3:
            segment_issues.append({
                "segment": i,
                "label": seg.label,
                "time": [round(seg.start, 2), round(min(seg.end, motion.num_frames / motion.fps), 2)],
                "dancer_accents": len(in_seg),
                "missed_accent_times": missed,
                "character_still_fraction": round(idle, 2),
            })

    channel_issues = [
        f"{name} sits at its range limit {frac:.0%} of the time (range {spec.channels[name].min}..{spec.channels[name].max})"
        for name, frac in motion.clip_fraction.items()
        if frac > 0.2
    ]
    unused = [name for name, v in motion.channels.items() if np.ptp(v) < 1e-6]
    if unused:
        channel_issues.append(f"channels that never move: {unused}")

    return VerificationReport(
        passed=not failed,
        score=_score(m),
        metrics=m,
        failed_checks=failed,
        segment_issues=segment_issues,
        channel_issues=channel_issues,
    )
