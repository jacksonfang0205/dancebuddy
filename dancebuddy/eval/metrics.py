"""Code-based metrics. All of them work on channel values, so every method that drives the same rig is scored the
same way. (Pixel-only baselines such as diffusion models need a flow-based speed signal instead; see TODO in README.)
"""

from __future__ import annotations

import numpy as np

from ..characters.spec import CharacterSpec
from ..motion.plan import CharacterMotion
from ..perception.features import Accents, DanceFeatures, detect_accents, smooth

DEFAULT_TOLERANCE_SEC = 0.1


def character_speed(motion: CharacterMotion, spec: CharacterSpec) -> np.ndarray:
    """Sum over channels of |velocity| / channel range: how much the character visibly moves per second."""
    total = np.zeros(motion.num_frames)
    for name, values in motion.channels.items():
        ch = spec.channels[name]
        span = ch.max - ch.min
        total += np.abs(np.gradient(values)) * motion.fps / span
    return total


def character_accents(motion: CharacterMotion, spec: CharacterSpec) -> Accents:
    return detect_accents(character_speed(motion, spec), motion.fps, abs_threshold=0.05)


def match_events(reference: np.ndarray, predicted: np.ndarray, tolerance: float) -> list[tuple[float, float]]:
    """Greedy one-to-one matching of event times within a tolerance, closest pairs first."""
    pairs = sorted(
        ((abs(r - p), i, j) for i, r in enumerate(reference) for j, p in enumerate(predicted) if abs(r - p) <= tolerance)
    )
    used_r, used_p, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in used_r and j not in used_p:
            used_r.add(i)
            used_p.add(j)
            matched.append((float(reference[i]), float(predicted[j])))
    return matched


def accent_scores(reference: np.ndarray, predicted: np.ndarray, tolerance: float = DEFAULT_TOLERANCE_SEC) -> dict:
    matched = match_events(reference, predicted, tolerance)
    coverage = len(matched) / len(reference) if len(reference) else float("nan")
    precision = len(matched) / len(predicted) if len(predicted) else 0.0
    f1 = 2 * coverage * precision / (coverage + precision) if (coverage + precision) > 0 else 0.0
    sync_ms = float(np.median([abs(r - p) for r, p in matched]) * 1000) if matched else float("nan")
    return {"coverage": coverage, "precision": precision, "f1": f1, "sync_error_ms": sync_ms, "matched": len(matched)}


def evaluate(features: DanceFeatures, motion: CharacterMotion, spec: CharacterSpec, tolerance: float = DEFAULT_TOLERANCE_SEC) -> dict:
    fps = motion.fps
    char_speed = character_speed(motion, spec)
    char_acc = character_accents(motion, spec)
    dancer_energy = smooth(features.energy["total"], fps, 0.5)
    char_env = smooth(char_speed, fps, 0.5)

    scores = accent_scores(features.accents.times, char_acc.times, tolerance)
    activity = float(np.corrcoef(dancer_energy, char_env)[0, 1]) if char_env.std() > 1e-9 else 0.0
    dancer_active = dancer_energy > np.median(dancer_energy)
    char_still = char_env < 0.1 * (np.percentile(char_env, 95) + 1e-9)
    idle = float(np.mean(char_still[dancer_active])) if dancer_active.any() else 0.0

    jerks = []
    for name, values in motion.channels.items():
        ch = spec.channels[name]
        x = (values - ch.min) / (ch.max - ch.min)
        if len(x) > 3:
            jerks.append(np.mean(np.abs(np.diff(x, 3))) * fps**3)
    return {
        "accent_coverage": scores["coverage"],
        "accent_precision": scores["precision"],
        "accent_f1": scores["f1"],
        "sync_error_ms": scores["sync_error_ms"],
        "dancer_accents": int(len(features.accents)),
        "character_accents": int(len(char_acc)),
        "activity_correlation": activity,
        "idle_while_dancer_moves": idle,
        "jitter": float(np.mean(jerks)) if jerks else 0.0,
        "clip_rate": float(np.mean(list(motion.clip_fraction.values()))) if motion.clip_fraction else 0.0,
    }
