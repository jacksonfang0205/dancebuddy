"""Measure a character's motion from a curves.json without rendering, for the evaluation metrics.

For every frame (or every --every-th) it poses the rig and writes:
  energy      how much the character moves: mean speed of its surface points (body heights per second)
  hands       each hand's centre in the picture (0..1, x right, y down), if the character has arms
  shoulders   the middle between its shoulders in the picture
  height_px   its height in the picture (0..1), to normalise positions
  in_frame    whether each hand is inside the picture

  python dancebuddy/blender/measure_motion.py -- --blend assets/plush_soft.blend --curves runs/x/agent/curves.json --out m.json
"""

import argparse
import json
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_utils import pose_frame  # noqa: E402


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--curves", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--height", type=int, default=960)
    return p.parse_args(argv)


def main():
    a = parse()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    curves = json.load(open(a.curves))
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = a.width, a.height  # the picture the renders use
    cam = sc.camera
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    body = bpy.data.objects.get("Body") or next(o for o in bpy.data.objects if o.type == "MESH" and
                                                 any(m.type == "ARMATURE" for m in o.modifiers))
    rest = np.array([v.co[:] for v in body.data.vertices])
    height = float(rest[:, 2].max() - rest[:, 2].min()) or 1.0
    groups = {g.index: g.name for g in body.vertex_groups}
    hands, shoulders = {}, {}
    for s in "LR":
        chain = sorted(b.name for b in rig.data.bones if b.name.startswith(f"arm_{s}"))
        if not chain:
            continue
        last = rig.data.bones[chain[-1]]
        h, t = np.array(last.head_local), np.array(last.tail_local)
        idx = np.array([v.index for v in body.data.vertices
                        if sum(g.weight for g in v.groups if groups[g.group] in chain) > 0.5])
        if not len(idx):
            continue
        along = (rest[idx] - h) @ (t - h) / max(float(np.dot(t - h, t - h)), 1e-9)
        hands[s] = idx[along > 0.8] if (along > 0.8).sum() > 5 else idx[np.argsort(-along)[:20]]
        shoulders[s] = rig.data.bones[chain[0]].name
    n = int(curves["n_frames"])
    frames = list(range(0, n, max(1, a.every)))
    sample = np.linspace(0, len(rest) - 1, min(len(rest), 4000)).astype(int)  # the speed of a spread of points
    mw = body.matrix_world

    def px_world(p):
        q = world_to_camera_view(sc, cam, Vector(p))
        return [float(q.x), float(1 - q.y)]

    def px(p):  # a point of the posed mesh (object space)
        return px_world(mw @ Vector(p))

    rows, prev = [], None
    fps = float(curves["fps"])
    for f in frames:
        pose_frame(rig, body, curves["channels"], f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg).to_mesh()
        P = np.array([v.co[:] for v in ev.vertices])
        body.evaluated_get(dg).to_mesh_clear()
        pts = P[sample]
        energy = 0.0 if prev is None else float(np.linalg.norm(pts - prev, axis=1).mean() / height * fps / a.every)
        prev = pts
        row = {"frame": f, "energy": energy}
        top, bottom = px(P[np.argmax(P[:, 2])]), px(P[np.argmin(P[:, 2])])
        row["height_px"] = abs(bottom[1] - top[1])
        if hands:
            row["hands"] = {s: px(P[idx].mean(0)) for s, idx in hands.items()}
            row["in_frame"] = {s: bool(0 <= v[0] <= 1 and 0 <= v[1] <= 1) for s, v in row["hands"].items()}
            sh = [rig.matrix_world @ rig.pose.bones[b].head for b in shoulders.values()]
            row["shoulders"] = px_world(np.mean([list(x) for x in sh], 0))
        rows.append(row)
    if len(rows) > 1:
        rows[0]["energy"] = rows[1]["energy"]
    json.dump({"fps": fps / a.every, "every": a.every, "rows": rows}, open(a.out, "w"))
    print(f"measured {len(rows)} frames -> {a.out}")


main()
