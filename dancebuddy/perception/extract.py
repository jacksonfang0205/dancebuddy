"""Extract body landmarks from a single-dancer video with MediaPipe Pose Landmarker."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pose_sequence import NUM_LANDMARKS, PoseSequence

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_full.task"


def extract_pose(video_path: str | Path, model_path: str | Path = DEFAULT_MODEL) -> PoseSequence:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    import mediapipe as mp

    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} not found; run scripts/download_models.sh")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    world, image, vis = [], [], []
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(frame_idx * 1000 / fps)
            )
            if result.pose_landmarks:
                lm, wl = result.pose_landmarks[0], result.pose_world_landmarks[0]
                image.append([[p.x, p.y, p.z] for p in lm])
                world.append([[p.x, p.y, p.z] for p in wl])
                vis.append([p.visibility if p.visibility is not None else 1.0 for p in lm])
            else:  # no detection: mark the frame invisible so it is interpolated later
                image.append(np.zeros((NUM_LANDMARKS, 3)))
                world.append(np.zeros((NUM_LANDMARKS, 3)))
                vis.append(np.zeros(NUM_LANDMARKS))
            frame_idx += 1
    cap.release()
    if frame_idx == 0:
        raise ValueError(f"could not read any frames from {video_path}")
    return PoseSequence(
        world=np.asarray(world, dtype=np.float32),
        image=np.asarray(image, dtype=np.float32),
        visibility=np.asarray(vis, dtype=np.float32),
        fps=float(fps),
        source=str(video_path),
    )
