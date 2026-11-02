"""Character control interfaces: which channels a character has, their ranges, and how they bind to the rig."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SPEC_DIR = Path(__file__).parent / "specs"


@dataclass(frozen=True)
class Channel:
    name: str
    min: float
    max: float
    rest: float
    unit: str
    description: str
    bind: dict

    def clip(self, value):
        import numpy as np

        return np.clip(value, self.min, self.max)


@dataclass
class CharacterSpec:
    name: str
    display_name: str
    category: str
    description: str
    parts_present: list[str]
    parts_missing: list[str]
    channels: dict[str, Channel]
    body_height_m: float = 1.0

    @classmethod
    def load(cls, name_or_path: str) -> "CharacterSpec":
        path = Path(name_or_path)
        if not path.exists():
            path = SPEC_DIR / f"{name_or_path}.json"
        raw = json.loads(path.read_text())
        channels = {k: Channel(name=k, **v) for k, v in raw.pop("channels").items()}
        return cls(channels=channels, **raw)

    def interface_for_llm(self) -> dict:
        """The control interface as shown to the planner (no rig bindings)."""
        return {
            "name": self.display_name,
            "description": self.description,
            "parts_present": self.parts_present,
            "parts_missing": self.parts_missing,
            "channels": {
                c.name: {"min": c.min, "max": c.max, "rest": c.rest, "unit": c.unit, "description": c.description}
                for c in self.channels.values()
            },
        }


def available_characters() -> list[str]:
    return sorted(p.stem for p in SPEC_DIR.glob("*.json"))
