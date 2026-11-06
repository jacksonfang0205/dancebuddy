"""Apply curves.json to a rigged character and render the frames.

  blender -b -P dancebuddy/blender/render_curves.py -- --blend assets/plush_soft.blend --curves runs/x/curves.json --frames runs/x/frames
  python dancebuddy/blender/render_curves.py --blend ... (if the bpy module is installed)

The channels are applied by pose_utils.pose_frame (see there for what each one does).
"""

import argparse
import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_utils import BBONE_PROPS, pose_frame  # noqa: E402
from scene_utils import setup_render  # noqa: E402



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

    keys = bpy.data.objects["Body"].data.shape_keys if "Body" in bpy.data.objects else None
    body = bpy.data.objects.get("Body")
    for f in range(n):
        pose_frame(rig, body, curves["channels"], f)
        if keys is not None:
            for kb in keys.key_blocks[1:]:
                kb.keyframe_insert("value", frame=f + 1)
        for pb in pbones:
            pb.keyframe_insert("rotation_quaternion" if pb.rotation_mode == "QUATERNION" else "rotation_euler",
                               frame=f + 1)
            pb.keyframe_insert("location", frame=f + 1)
            pb.keyframe_insert("scale", frame=f + 1)
            if pb.bone.bbone_segments > 1:
                for prop in BBONE_PROPS:
                    pb.keyframe_insert(prop, frame=f + 1)

    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend))
    setup_render(a.engine, a.width, a.height)
    os.makedirs(a.frames, exist_ok=True)
    scene.render.filepath = os.path.join(os.path.abspath(a.frames), "frame_")
    bpy.ops.render.render(animation=True)
    print(f"rendered {n} frames to {a.frames}")


main()
