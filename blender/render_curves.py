"""Apply curves.json to a rigged character and render the frames.

  blender -b -P blender/render_curves.py -- --blend assets/capsule.blend --curves runs/x/curves.json --frames runs/x/frames
  python blender/render_curves.py --blend ... (if the bpy module is installed)

Channel -> pose bone mapping (bones point up, so bone-local Y is the bone axis):
  lean  -> rotation X      twist -> rotation Y      sway -> rotation Z
  root.lift -> location Y (along the root bone = world up)
  root.squash -> scale (Y shrinks, X/Z grow to keep volume)
  <bone>.stretch -> length factor along the bone (volume kept)
  <bone>.quat -> rotation quaternion [w, x, y, z] in the bone's rest frame (overrides the Euler channels)
"""

import argparse
import json
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_utils import setup_render  # noqa: E402

AXIS = {"lean": 0, "twist": 1, "sway": 2}


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--curves", required=True)
    p.add_argument("--frames", required=True, help="output directory for PNG frames")
    p.add_argument("--engine", default="auto")
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--save-blend", default="", help="optionally save the animated .blend here")
    return p.parse_args(argv)


def main():
    a = parse()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    with open(a.curves) as f:
        curves = json.load(f)
    scene = bpy.context.scene
    n = int(curves["n_frames"])
    if a.max_frames:
        n = min(n, a.max_frames)
    fps = float(curves["fps"])
    scene.render.fps = int(round(fps))
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start, scene.frame_end = 1, n
    rig = bpy.data.objects["Rig"]
    pbones = rig.pose.bones
    for pb in pbones:
        pb.rotation_mode = "XYZ"

    for f in range(n):
        for pb in pbones:
            pb.rotation_euler = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
            pb.scale = (1, 1, 1)
        for name, vals in curves["channels"].items():
            bone, ch = name.split(".", 1)
            if bone not in pbones:
                continue
            pb = pbones[bone]
            v = vals[f]
            if ch == "quat":  # full rotation (w, x, y, z) in the bone's rest frame, e.g. from retarget.py
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = v
            elif ch in AXIS:
                r = list(pb.rotation_euler)
                r[AXIS[ch]] = math.radians(v)
                pb.rotation_euler = r
            elif ch == "lift":
                pb.location = (0, v, 0)
            elif ch == "stretch":  # lengthen a bone along its axis, keeping its volume (soft limbs)
                st = max(0.3, v)
                pb.scale = (1 / math.sqrt(st), st, 1 / math.sqrt(st))
            elif ch == "squash":
                s = max(0.3, 1 - v)
                pb.scale = (1 / math.sqrt(s), s, 1 / math.sqrt(s))
        for pb in pbones:
            pb.keyframe_insert("rotation_quaternion" if pb.rotation_mode == "QUATERNION" else "rotation_euler",
                               frame=f + 1)
            pb.keyframe_insert("location", frame=f + 1)
            pb.keyframe_insert("scale", frame=f + 1)

    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend))
    setup_render(a.engine, a.width, a.height)
    os.makedirs(a.frames, exist_ok=True)
    scene.render.filepath = os.path.join(os.path.abspath(a.frames), "frame_")
    bpy.ops.render.render(animation=True)
    print(f"rendered {n} frames to {a.frames}")


main()
