"""Perception: dance video -> keypoints -> dance score.

Pose: MediaPipe Pose Landmarker (33 landmarks per frame, both image
coordinates and hip-centred world coordinates in metres).

Dance score (dance_score.json) is the shared input to the planner, the
baseline and the metrics:
  fps, n_frames, beats, accents        frames of kinematic hits
  units                                move units: span, accents, which body
                                       regions move most, simple features
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from dancebuddy.perception.beats import beats_and_accents, smooth

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "pose_landmarker_full.task"

# MediaPipe landmark indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
REGIONS = {
    "arms": [L_EL, R_EL, L_WR, R_WR],
    "legs": [L_KN, R_KN, L_AN, R_AN],
    "torso": [L_SH, R_SH, L_HIP, R_HIP],
    "head": [NOSE],
}


# ---------------------------------------------------------------- keypoints

def _ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading pose model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def extract_keypoints(video: str | Path, out_npz: str | Path, max_seconds: float | None = None) -> dict:
    """Run MediaPipe on every frame; save world (N,33,3), image (N,33,3), visibility (N,33)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    video = _readable_video(Path(video), Path(out_npz).parent)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    world, image, vis = [], [], []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_seconds and i >= max_seconds * fps):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
            if res.pose_landmarks:
                image.append([[p.x, p.y, p.z] for p in res.pose_landmarks[0]])
                world.append([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
                vis.append([p.visibility or 0.0 for p in res.pose_landmarks[0]])
            else:
                image.append(np.full((33, 3), np.nan).tolist())
                world.append(np.full((33, 3), np.nan).tolist())
                vis.append([0.0] * 33)
            i += 1
    cap.release()
    world, image, vis = (np.array(a, float) for a in (world, image, vis))
    world, image = _fill_missing(world), _fill_missing(image)
    np.savez_compressed(out_npz, world=world, image=image, visibility=vis, fps=fps)
    detected = float(np.mean(vis.max(axis=1) > 0))
    print(f"pose: {len(world)} frames at {fps:.1f} fps, person found in {detected:.0%} of frames")
    return {"world": world, "image": image, "visibility": vis, "fps": fps}


def _readable_video(video: Path, work_dir: Path) -> Path:
    """Return a path OpenCV can decode. Phone and TikTok videos are often HEVC
    (or .mov/.webm), which OpenCV builds may not read, so convert to H.264."""
    import shutil
    import subprocess

    import cv2

    video = video.expanduser()
    if not video.exists():
        raise FileNotFoundError(f"no file at {video}. Check the path (ls -la '{video}').")
    cap = cv2.VideoCapture(str(video))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if ok:
        return video
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"OpenCV cannot decode {video} and ffmpeg is not installed (brew install ffmpeg)")
    work_dir.mkdir(parents=True, exist_ok=True)
    conv = work_dir / "input_h264.mp4"
    print(f"OpenCV cannot decode {video.name}; converting to H.264 -> {conv}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(conv)], check=True)
    return conv


def _fill_missing(a: np.ndarray) -> np.ndarray:
    """Linear interpolation over frames where pose was not found."""
    a = a.copy()
    n = len(a)
    ok = ~np.isnan(a[:, 0, 0])
    if ok.sum() == 0:
        raise RuntimeError("no person detected in any frame")
    idx = np.arange(n)
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            a[:, j, k] = np.interp(idx, idx[ok], a[ok, j, k])
    return a


def load_keypoints(npz: str | Path) -> dict:
    d = np.load(npz)
    return {"world": d["world"], "image": d["image"], "visibility": d["visibility"], "fps": float(d["fps"])}


# ---------------------------------------------------------------- dance score

def body_speed(kp: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-frame speed of the whole body (m/s) and of each region."""
    fps = kp["fps"]
    w = kp["world"]
    img = kp["image"]
    vel = np.linalg.norm(np.gradient(w, axis=0), axis=2) * fps  # (N,33)
    # world landmarks are hip-centred, so add whole-body vertical motion from the image
    torso_len = np.median(np.abs(img[:, [L_HIP, R_HIP], 1].mean(1) - img[:, [L_SH, R_SH], 1].mean(1))) + 1e-6
    hip_y = img[:, [L_HIP, R_HIP], 1].mean(1) / torso_len * 0.5  # metres, roughly (torso ~0.5 m)
    bob = np.abs(np.gradient(hip_y)) * fps
    regions = {r: vel[:, idx].mean(1) for r, idx in REGIONS.items()}
    regions["bounce"] = bob
    joints = REGIONS["arms"] + REGIONS["legs"] + REGIONS["torso"]
    total = vel[:, joints].mean(1) + bob
    return total, regions


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians((x + 180) % 360 - 180)))


def _yaw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    return np.degrees(np.arctan2(v[:, 2], v[:, 0]))


def build_dance_score(kp: dict, unit_seconds: float = 2.0, source: str = "") -> dict:
    fps = kp["fps"]
    n = len(kp["world"])
    speed, regions = body_speed(kp)
    beats, accents = beats_and_accents(speed, fps)
    if len(accents) < 2:  # fall back to the strongest beats so the plan has anchors
        accents = beats

    # cut into move units of about unit_seconds, at the beat closest to each cut
    cuts = [0]
    target = int(unit_seconds * fps)
    while cuts[-1] + target < n - target // 2:
        want = cuts[-1] + target
        near = [b for b in beats if abs(b - want) < target // 2]
        cut = min(near, key=lambda b: abs(b - want)) if near else want
        # cut a little before the beat so the beat belongs to the next unit
        cut = max(cuts[-1] + 1, cut - 2)
        cuts.append(cut)
    cuts.append(n)

    w = kp["world"]
    sh_yaw = _yaw(w[:, L_SH], w[:, R_SH])
    hip_yaw = _yaw(w[:, L_HIP], w[:, R_HIP])
    twist = smooth(wrap_deg(sh_yaw - hip_yaw), 2)
    sh_mid, hip_mid = w[:, [L_SH, R_SH]].mean(1), w[:, [L_HIP, R_HIP]].mean(1)
    roll = smooth(np.degrees(np.arctan2(sh_mid[:, 0] - hip_mid[:, 0], -(sh_mid[:, 1] - hip_mid[:, 1]))), 2)

    units = []
    for i, (s, e) in enumerate(zip(cuts[:-1], cuts[1:])):
        e = e - 1
        energy = {r: float(v[s:e + 1].mean()) for r, v in regions.items()}
        tot = sum(energy[r] for r in ("arms", "legs", "torso", "head")) + 1e-9
        share = {r: round(energy[r] / tot, 2) for r in ("arms", "legs", "torso", "head")}
        dominant = [r for r, v in sorted(share.items(), key=lambda kv: -kv[1]) if v >= 0.25]
        label = "full-body"
        if share["arms"] >= 0.5:
            label = "arm-focused"
        elif share["legs"] >= 0.4:
            label = "footwork"
        units.append({
            "id": i,
            "start": int(s), "end": int(e),
            "start_s": round(s / fps, 2), "end_s": round(e / fps, 2),
            "accents": [a for a in accents if s <= a <= e],
            "beats": [b for b in beats if s <= b <= e],
            "label": label,
            "dominant_regions": dominant,
            "region_share": share,
            "features": {
                "mean_speed_mps": round(float(speed[s:e + 1].mean()), 3),
                "vertical_bounce_mps": round(energy["bounce"], 3),
                "torso_twist_range_deg": round(float(np.ptp(twist[s:e + 1])), 1),
                "torso_side_lean_range_deg": round(float(np.ptp(roll[s:e + 1])), 1),
            },
        })

    from dancebuddy.perception.signals import compute_signals, signal_stats

    signals = compute_signals(kp)
    return {
        "source": source,
        "fps": fps,
        "n_frames": n,
        "duration_s": round(n / fps, 2),
        "beats": beats,
        "accents": accents,
        "units": units,
        # per-frame data for the `follow` primitive and the get_dance_signals tool (not shown to the planner raw)
        "signals": {k: [round(float(x), 3) for x in v] for k, v in signals.items()},
        "signal_stats": signal_stats(signals),
        "motion_energy": [round(float(x), 4) for x in speed],
    }


def perceive(video: str | Path, out_dir: str | Path, max_seconds: float | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "keypoints.npz"
    kp = load_keypoints(npz) if npz.exists() else extract_keypoints(video, npz, max_seconds)
    score = build_dance_score(kp, source=str(video))
    (out / "dance_score.json").write_text(json.dumps(score, indent=1))
    print(f"dance score: {len(score['beats'])} beats, {len(score['accents'])} accents, {len(score['units'])} units")
    return score
