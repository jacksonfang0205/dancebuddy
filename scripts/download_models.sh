#!/usr/bin/env bash
# Downloads the MediaPipe pose landmarker model used by dancebuddy/perception/extract.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
echo "saved models/pose_landmarker_full.task"
