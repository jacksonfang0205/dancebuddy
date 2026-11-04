"""A standard test motion for a character (not a dance): stand, hop, quick turns that stop, a side tilt, stand.

Used to check a character's physical properties (e.g. how its hair swings and settles) without fitting them to any
particular dance.

  python tools/test_motion.py runs/_test_motion --char assets/plush_soft
  python -m dancebuddy render runs/_test_motion --systems test --char assets/plush_soft --engine workbench
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dancebuddy.retarget import _local  # noqa: E402

FPS = 30


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--char", default="assets/plush_soft")
    a = p.parse_args()
    rig = json.loads(Path(a.char + "_rig.json").read_text())
    M = {b["name"]: np.array(b["matrix"]) for b in rig["bones"]}
    h = rig["height_m"]
    n = 130
    t = np.arange(n)
    lift = 0.15 * h * np.where((t >= 15) & (t <= 27), np.sin(np.pi * (t - 15) / 12), 0)
    squash = 0.12 * np.exp(-0.5 * ((t - 12) / 2) ** 2) + 0.14 * np.exp(-0.5 * ((t - 30) / 3) ** 2)
    turn = 40 * ease((t - 45) / 6) - 40 * ease((t - 70) / 6)
    tilt = 18 * ease((t - 95) / 5) - 18 * ease((t - 105) / 5)
    ch = {"root.quat": [], "root.loc": [], "root.squash": [round(float(x), 4) for x in squash]}
    for i in range(n):
        r = Rot.from_euler("z", turn[i], degrees=True) * Rot.from_euler("y", tilt[i], degrees=True)
        ch["root.quat"].append(_local(M["root"], Rot.identity(), r))
        ch["root.loc"].append([0.0, round(float(lift[i]), 4), 0.0])
    out = Path(a.run) / "test"
    out.mkdir(parents=True, exist_ok=True)
    (out / "curves.json").write_text(json.dumps({"fps": FPS, "n_frames": n, "character": rig["name"], "channels": ch}))
    print(f"test motion: {n} frames -> {out}")


main()
