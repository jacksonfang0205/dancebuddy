"""Turn raw landmarks into interpretable dance features, motion energy, accents and segments.

Everything here is plain numpy, so it runs without Blender or MediaPipe and is easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from . import pose_sequence as P
from .pose_sequence import PoseSequence

VISIBILITY_THRESHOLD = 0.3

PART_LANDMARKS = {
    "arms": [P.L_ELBOW, P.R_ELBOW, P.L_WRIST, P.R_WRIST],
    "legs": [P.L_KNEE, P.R_KNEE, P.L_ANKLE, P.R_ANKLE, P.L_FOOT, P.R_FOOT],
    "torso": [P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP],
    "head": [P.NOSE],
}
PART_WEIGHTS = {"arms": 1.0, "legs": 1.0, "torso": 1.5, "head": 0.5}


@dataclass
class Accents:
    """Moments where a movement hits or holds: local speed minima right after a burst of motion."""

    times: np.ndarray  # seconds
    strengths: np.ndarray  # speed drop into the minimum, in torso lengths per second

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class Segment:
    start: float
    end: float
    dominant_part: str  # arms | legs | torso | head | still

    def to_dict(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2), "dominant_part": self.dominant_part}


@dataclass
class DanceFeatures:
    fps: float
    num_frames: int
    channels: dict[str, np.ndarray]  # interpretable signals, one value per frame
    energy: dict[str, np.ndarray]  # per-part speed, torso lengths per second
    accents: Accents  # whole-body accents
    part_accents: dict[str, Accents] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.num_frames) / self.fps

    def event_times(self, name: str) -> np.ndarray:
        """Event streams a plan can reference: 'accents', 'steps', 'step_L', 'step_R', 'arm_hits', ..."""
        if name == "accents":
            return self.accents.times
        if name == "steps":  # both feet, with near-simultaneous left/right events merged
            both = np.sort(np.concatenate([self.part_accents["step_L"].times, self.part_accents["step_R"].times]))
            keep = np.concatenate([[True], np.diff(both) > 0.12]) if len(both) else np.array([], dtype=bool)
            return both[keep]
        if name in self.part_accents:
            return self.part_accents[name].times
        raise KeyError(f"unknown event stream {name!r}")

    def summary(self) -> dict:
        """Compact description of the dance for a language model."""
        return {
            "duration_sec": round(self.num_frames / self.fps, 2),
            "fps": self.fps,
            "num_accents": len(self.accents),
            "accent_times_sec": [round(float(t), 2) for t in self.accents.times],
            "num_steps": len(self.event_times("steps")),
            "segments": [s.to_dict() for s in self.segments],
            "channel_ranges": {
                k: [round(float(np.percentile(v, 5)), 1), round(float(np.percentile(v, 95)), 1)]
                for k, v in self.channels.items()
            },
            "mean_part_energy": {k: round(float(v.mean()), 3) for k, v in self.energy.items()},
        }


# ---------------------------------------------------------------- preprocessing


def _odd(n: float, minimum: int = 5) -> int:
    n = max(int(round(n)), minimum)
    return n if n % 2 == 1 else n + 1


def _fill_gaps(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate frames where valid is False. x has shape (T, ...)."""
    if valid.all() or not valid.any():
        return x
    t = np.arange(len(x))
    flat = x.reshape(len(x), -1).copy()
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, t[valid], flat[valid, j])
    return flat.reshape(x.shape)


def smooth(x: np.ndarray, fps: float, window_sec: float = 0.25) -> np.ndarray:
    window = _odd(window_sec * fps)
    if len(x) <= window:
        return x
    return savgol_filter(x, window, polyorder=2, axis=0)


def _rolling_mean(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(width, len(x)))
    kernel = np.ones(width) / width
    padded = np.pad(x, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def clean(seq: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Gap-filled, smoothed world and image landmarks."""
    world, image = seq.world.copy(), seq.image.copy()
    for j in range(P.NUM_LANDMARKS):
        valid = seq.visibility[:, j] >= VISIBILITY_THRESHOLD
        world[:, j] = _fill_gaps(world[:, j], valid)
        image[:, j] = _fill_gaps(image[:, j], valid)
    return smooth(world, seq.fps), smooth(image, seq.fps)


# ---------------------------------------------------------------- geometry helpers


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle in degrees between row vectors."""
    cos = np.sum(_unit(a) * _unit(b), axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _joint_bend(a: np.ndarray, joint: np.ndarray, c: np.ndarray) -> np.ndarray:
    """0 when the limb is straight, larger when bent."""
    return 180.0 - _angle(a - joint, c - joint)


# ---------------------------------------------------------------- features


def compute_channels(world: np.ndarray, image: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Interpretable per-frame signals. Angles are in degrees; translations in torso lengths."""
    w = world
    hip_mid = (w[:, P.L_HIP] + w[:, P.R_HIP]) / 2
    sh_mid = (w[:, P.L_SHOULDER] + w[:, P.R_SHOULDER]) / 2
    up = _unit(sh_mid - hip_mid)

    # Body axes. MediaPipe world space has y pointing down, so gravity "up" is -y.
    world_up = np.array([0.0, -1.0, 0.0])
    lat = w[:, P.L_HIP] - w[:, P.R_HIP]  # toward the dancer's left
    lat_h = _unit(lat * np.array([1.0, 0.0, 1.0]))
    fwd_h = _unit(np.cross(lat_h, world_up))  # toward where the dancer faces

    ch: dict[str, np.ndarray] = {}
    ch["torso_lean_side"] = np.degrees(np.arcsin(np.clip(np.sum(up * lat_h, -1), -1, 1)))
    ch["torso_lean_fwd"] = np.degrees(np.arcsin(np.clip(np.sum(up * fwd_h, -1), -1, 1)))

    sh_line = (w[:, P.L_SHOULDER] - w[:, P.R_SHOULDER]) * np.array([1.0, 0.0, 1.0])
    hip_line = lat * np.array([1.0, 0.0, 1.0])
    twist = _angle(sh_line, hip_line)
    sign = np.sign(np.cross(hip_line, sh_line)[:, 1] * -1.0)
    ch["torso_twist"] = twist * np.where(sign == 0, 1.0, sign)

    head = w[:, P.NOSE] - sh_mid
    ch["head_nod"] = np.degrees(np.arctan2(np.sum(head * fwd_h, -1), np.sum(head * up, -1)))
    ch["head_tilt"] = np.degrees(np.arctan2(np.sum(head * lat_h, -1), np.sum(head * up, -1)))

    for side, s, e, wr in (("L", P.L_SHOULDER, P.L_ELBOW, P.L_WRIST), ("R", P.R_SHOULDER, P.R_ELBOW, P.R_WRIST)):
        upper = w[:, e] - w[:, s]
        ch[f"arm_{side}_raise"] = _angle(upper, -up)  # 0 down, 90 horizontal, 180 overhead
        ch[f"arm_{side}_forward"] = np.degrees(np.arcsin(np.clip(np.sum(_unit(upper) * fwd_h, -1), -1, 1)))
        ch[f"arm_{side}_elbow"] = _joint_bend(w[:, s], w[:, e], w[:, wr])

    for side, h, k, a in (("L", P.L_HIP, P.L_KNEE, P.L_ANKLE), ("R", P.R_HIP, P.R_KNEE, P.R_ANKLE)):
        thigh = w[:, k] - w[:, h]
        # Forward elevation of the thigh: 0 standing, positive when the knee comes up in front.
        ch[f"hip_{side}_flex"] = np.degrees(np.arctan2(np.sum(thigh * fwd_h, -1), np.sum(thigh * -up, -1)))
        ch[f"knee_{side}_bend"] = _joint_bend(w[:, h], w[:, k], w[:, a])

    # Root translation and foot lift from the image, in torso lengths.
    im = image[:, :, :2]
    hip_im = (im[:, P.L_HIP] + im[:, P.R_HIP]) / 2
    sh_im = (im[:, P.L_SHOULDER] + im[:, P.R_SHOULDER]) / 2
    torso_len = float(np.median(np.linalg.norm(sh_im - hip_im, axis=-1))) + 1e-6
    drift = _odd(2.0 * fps)
    ch["bounce"] = -(hip_im[:, 1] - _rolling_mean(hip_im[:, 1], drift)) / torso_len
    ch["sway"] = (hip_im[:, 0] - _rolling_mean(hip_im[:, 0], drift)) / torso_len
    ground = _odd(1.0 * fps)
    for side, a in (("L", P.L_ANKLE), ("R", P.R_ANKLE)):
        y = im[:, a, 1]
        lowest = np.array([y[max(0, i - ground // 2): i + ground // 2 + 1].max() for i in range(len(y))])
        ch[f"foot_{side}_lift"] = (lowest - y) / torso_len
    return ch


def compute_energy(image: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Mean landmark speed per body part, in torso lengths per second (scale invariant)."""
    im = image[:, :, :2]
    hip_im = (im[:, P.L_HIP] + im[:, P.R_HIP]) / 2
    sh_im = (im[:, P.L_SHOULDER] + im[:, P.R_SHOULDER]) / 2
    torso_len = float(np.median(np.linalg.norm(sh_im - hip_im, axis=-1))) + 1e-6
    vel = np.gradient(im, axis=0) * fps / torso_len
    speed = np.linalg.norm(vel, axis=-1)  # (T, 33)
    energy = {part: speed[:, idx].mean(axis=1) for part, idx in PART_LANDMARKS.items()}
    energy["step_L"] = speed[:, [P.L_ANKLE, P.L_FOOT]].mean(axis=1)
    energy["step_R"] = speed[:, [P.R_ANKLE, P.R_FOOT]].mean(axis=1)
    energy["arm_L"] = speed[:, [P.L_ELBOW, P.L_WRIST]].mean(axis=1)
    energy["arm_R"] = speed[:, [P.R_ELBOW, P.R_WRIST]].mean(axis=1)
    total_w = sum(PART_WEIGHTS.values())
    energy["total"] = sum(PART_WEIGHTS[p] * energy[p] for p in PART_WEIGHTS) / total_w
    return energy


def detect_accents(
    speed: np.ndarray,
    fps: float,
    min_gap_sec: float = 0.2,
    lookback_sec: float = 0.35,
    rel_threshold: float = 0.25,
    abs_threshold: float = 0.15,
) -> Accents:
    """Kinematic accents: local minima of speed that follow a burst of motion (a hit or a hold).

    A minimum counts when the speed dropped into it by at least
    max(abs_threshold, rel_threshold * 95th-percentile speed).
    """
    s = smooth(speed, fps, 0.15)
    minima, _ = find_peaks(-s, distance=max(1, int(min_gap_sec * fps)))
    lookback = max(1, int(lookback_sec * fps))
    threshold = max(abs_threshold, rel_threshold * float(np.percentile(s, 95)))
    times, strengths = [], []
    for m in minima:
        drop = float(s[max(0, m - lookback): m + 1].max() - s[m])
        if drop >= threshold:
            times.append(m / fps)
            strengths.append(drop)
    return Accents(np.array(times), np.array(strengths))


def segment_dance(energy: dict[str, np.ndarray], fps: float, min_len_sec: float = 1.0) -> list[Segment]:
    """Split the dance into stretches dominated by one body part."""
    # A part dominates when it moves more than usual for that part, so each part is compared to its own mean.
    parts = ["arms", "legs", "torso", "head"]
    norm = np.stack([smooth(energy[p], fps, 0.5) / (energy[p].mean() + 1e-6) for p in parts], axis=1)
    names = parts + ["still"]
    total = smooth(energy["total"], fps, 0.5)
    labels = norm.argmax(axis=1)
    labels[total < 0.25 * np.median(total)] = len(parts)

    # Majority vote over a sliding window so labels do not flicker frame to frame.
    half = max(1, int(min_len_sec * fps / 2))
    voted = np.array([np.bincount(labels[max(0, i - half): i + half + 1], minlength=len(names)).argmax() for i in range(len(labels))])

    runs: list[list[int]] = []  # [start_frame, end_frame, label]
    for i, lab in enumerate(voted):
        if runs and runs[-1][2] == lab:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, int(lab)])
    # Absorb runs shorter than min_len into the previous run (or the next one at the start).
    min_frames = int(min_len_sec * fps)
    merged: list[list[int]] = []
    for run in runs:
        if merged and (run[1] - run[0] < min_frames or merged[-1][2] == run[2]):
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_frames:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [Segment(s / fps, e / fps, names[lab]) for s, e, lab in merged]


def extract_features(seq: PoseSequence) -> DanceFeatures:
    world, image = clean(seq)
    channels = compute_channels(world, image, seq.fps)
    energy = compute_energy(image, seq.fps)
    accents = detect_accents(energy["total"], seq.fps)
    part_accents = {name: detect_accents(energy[name], seq.fps) for name in ("step_L", "step_R", "arm_L", "arm_R")}
    part_accents["arm_hits"] = detect_accents(energy["arms"], seq.fps)
    return DanceFeatures(
        fps=seq.fps,
        num_frames=seq.num_frames,
        channels=channels,
        energy=energy,
        accents=accents,
        part_accents=part_accents,
        segments=segment_dance(energy, seq.fps),
    )
