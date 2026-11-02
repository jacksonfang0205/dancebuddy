"""Container for per-frame body landmarks extracted from a dance video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MediaPipe BlazePose landmark indices (33-point topology). "Left" is the dancer's left.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32
NUM_LANDMARKS = 33


@dataclass
class PoseSequence:
    """Landmarks for T frames.

    world:      (T, 33, 3) metric coordinates centered at the hips (x right, y down, z toward camera).
    image:      (T, 33, 3) normalized image coordinates (x, y in [0, 1]; y points down).
    visibility: (T, 33) landmark confidence in [0, 1].
    fps:        frames per second of the source video.
    """

    world: np.ndarray
    image: np.ndarray
    visibility: np.ndarray
    fps: float
    source: str = ""

    @property
    def num_frames(self) -> int:
        return self.world.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            world=self.world,
            image=self.image,
            visibility=self.visibility,
            fps=np.array(self.fps),
            source=np.array(self.source),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        data = np.load(path)
        return cls(
            world=data["world"],
            image=data["image"],
            visibility=data["visibility"],
            fps=float(data["fps"]),
            source=str(data["source"]),
        )
