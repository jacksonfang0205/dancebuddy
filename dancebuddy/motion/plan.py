"""Motion plans and the compiler that turns them into per-frame character channel values.

A plan says, for each segment of the dance, which dancer signal drives which character channel and how.
Planners (rule-based or LLM) only decide *what* to map; this module computes the exact timing and values.

Plan JSON:
{
  "rationale": "...",
  "segments": [
    {"start": 0.0, "end": 3.5, "label": "footwork",
     "mappings": [
       {"target": "root_bounce", "source": "steps", "mode": "pulse", "gain": 0.12},
       {"target": "arm_L_raise", "source": "arm_L_raise", "mode": "copy", "gain": 1.0},
       {"target": "body_tilt", "source": "sway", "mode": "follow", "gain": 15}
     ]}
  ]
}

Modes
  copy    target += gain * feature            (feature and target share units, e.g. degrees)
  follow  target += gain * normalized feature (feature rescaled to about [-1, 1]; gain is the amplitude in target units)
  pulse   target += gain * bump at each event (source is an event stream; the bump peaks on the event)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..characters.spec import CharacterSpec
from ..perception.features import DanceFeatures, smooth

MODES = ("copy", "follow", "pulse")
EVENT_SOURCES = ("accents", "steps", "step_L", "step_R", "arm_L", "arm_R", "arm_hits")
CROSSFADE_SEC = 0.15


@dataclass
class Mapping:
    target: str
    source: str
    mode: str
    gain: float = 1.0
    offset: float = 0.0
    width_sec: float = 0.3  # pulse width

    @classmethod
    def from_dict(cls, d: dict) -> "Mapping":
        return cls(**{k: d[k] for k in ("target", "source", "mode", "gain", "offset", "width_sec") if k in d})


@dataclass
class PlanSegment:
    start: float
    end: float
    label: str
    mappings: list[Mapping]
    intent: str = ""


@dataclass
class MotionPlan:
    segments: list[PlanSegment]
    rationale: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "MotionPlan":
        segs = [
            PlanSegment(
                start=float(s["start"]),
                end=float(s["end"]),
                label=str(s.get("label", "")),
                mappings=[Mapping.from_dict(m) for m in s.get("mappings", [])],
                intent=str(s.get("intent", "")),
            )
            for s in d.get("segments", [])
        ]
        return cls(segments=segs, rationale=str(d.get("rationale", "")))

    def to_dict(self) -> dict:
        return {
            "rationale": self.rationale,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "label": s.label,
                    "intent": s.intent,
                    "mappings": [m.__dict__ for m in s.mappings],
                }
                for s in self.segments
            ],
        }

    def validate(self, spec: CharacterSpec, features: DanceFeatures) -> list[str]:
        """Problems that make the plan impossible to compile, phrased so a planner can fix them."""
        errors = []
        for i, seg in enumerate(self.segments):
            if seg.end <= seg.start:
                errors.append(f"segment {i}: end must be after start")
            for m in seg.mappings:
                if m.target not in spec.channels:
                    errors.append(f"segment {i}: unknown target channel {m.target!r}; available: {sorted(spec.channels)}")
                if m.mode not in MODES:
                    errors.append(f"segment {i}: unknown mode {m.mode!r}; use one of {MODES}")
                elif m.mode == "pulse" and m.source not in EVENT_SOURCES:
                    errors.append(f"segment {i}: pulse source must be an event stream {EVENT_SOURCES}, got {m.source!r}")
                elif m.mode != "pulse" and m.source not in features.channels and m.source not in features.energy:
                    errors.append(f"segment {i}: unknown feature {m.source!r}; available: {sorted(features.channels)}")
        return errors


@dataclass
class CharacterMotion:
    """Per-frame values for each character channel, in the channel's own units."""

    fps: float
    channels: dict[str, np.ndarray]
    clip_fraction: dict[str, float] = field(default_factory=dict)  # share of frames that hit a range limit

    @property
    def num_frames(self) -> int:
        return len(next(iter(self.channels.values())))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "fps": self.fps,
            "channels": {k: [round(float(x), 4) for x in v] for k, v in self.channels.items()},
            "clip_fraction": self.clip_fraction,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "CharacterMotion":
        raw = json.loads(Path(path).read_text())
        return cls(raw["fps"], {k: np.array(v) for k, v in raw["channels"].items()}, raw.get("clip_fraction", {}))


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [5, 95])
    half = (hi - lo) / 2 + 1e-6
    return (x - (hi + lo) / 2) / half


def _pulse_train(event_times: np.ndarray, num_frames: int, fps: float, width_sec: float) -> np.ndarray:
    """Raised-cosine bumps of height 1 peaking on each event."""
    t = np.arange(num_frames) / fps
    out = np.zeros(num_frames)
    half = width_sec / 2
    for e in event_times:
        d = np.abs(t - e)
        bump = np.where(d < half, 0.5 * (1 + np.cos(np.pi * d / half)), 0.0)
        out = np.maximum(out, bump)
    return out


def _segment_weight(seg: PlanSegment, num_frames: int, fps: float) -> np.ndarray:
    """1 inside the segment, ramping over CROSSFADE_SEC at the edges so segments blend."""
    t = np.arange(num_frames) / fps
    duration = num_frames / fps
    rise = np.ones(num_frames) if seg.start <= 0 else np.clip((t - seg.start) / CROSSFADE_SEC + 0.5, 0, 1)
    fall = np.ones(num_frames) if seg.end >= duration else np.clip((seg.end - t) / CROSSFADE_SEC + 0.5, 0, 1)
    return np.minimum(rise, fall)


def compile_plan(plan: MotionPlan, spec: CharacterSpec, features: DanceFeatures) -> CharacterMotion:
    errors = plan.validate(spec, features)
    if errors:
        raise ValueError("invalid plan:\n" + "\n".join(errors))
    n, fps = features.num_frames, features.fps
    deltas = {name: np.zeros(n) for name in spec.channels}
    for seg in plan.segments:
        weight = _segment_weight(seg, n, fps)
        for m in seg.mappings:
            if m.mode == "pulse":
                signal = _pulse_train(features.event_times(m.source), n, fps, m.width_sec)
            else:
                src = features.channels.get(m.source)
                if src is None:
                    src = features.energy[m.source]
                signal = src if m.mode == "copy" else _normalize(src)
            contribution = m.gain * signal + m.offset
            if m.mode == "copy":  # copy sets the absolute value, so it replaces the rest pose
                contribution = contribution - spec.channels[m.target].rest
            deltas[m.target] += weight * contribution

    out, clip = {}, {}
    for name, ch in spec.channels.items():
        raw = smooth(ch.rest + deltas[name], fps, 0.12)
        clip[name] = float(np.mean((raw <= ch.min) | (raw >= ch.max)))
        out[name] = ch.clip(raw)
    return CharacterMotion(fps=fps, channels=out, clip_fraction=clip)


def whole_clip(features: DanceFeatures, mappings: list[Mapping], label: str = "whole clip") -> MotionPlan:
    """A one-segment plan covering the whole dance."""
    return MotionPlan(segments=[PlanSegment(0.0, features.num_frames / features.fps + 1.0, label, mappings)])
