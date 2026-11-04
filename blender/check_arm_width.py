"""Check that the arms keep their width (motion spec: an arm bends but its width and length never change).

For sampled frames of a curves.json, pose the rig, and measure each arm's thickness along its length: the arm's
points are sliced along the rest arm, and each slice's spread (RMS distance to the slice centre) is compared
with the rest pose. Prints the worst change per frame and an overall verdict.

  python blender/check_arm_width.py -- --blend assets/plush_soft.blend --curves runs/x/copy/curves.json
"""

import argparse
import json
import math
import os
import sys

import bpy
import numpy as np

BBONE = ("bbone_curveinx", "bbone_curveinz", "bbone_curveoutx", "bbone_curveoutz", "bbone_rollin", "bbone_rollout")


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--curves", required=True)
    p.add_argument("--frames", type=int, default=24, help="how many frames to sample")
    p.add_argument("--tolerance", type=float, default=0.10, help="allowed width change (fraction)")
    return p.parse_args(argv)


def pose(rig, curves, f):
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion, pb.location, pb.scale = (1, 0, 0, 0), (0, 0, 0), (1, 1, 1)
        for prop in BBONE:
            setattr(pb, prop, 0.0)
    for name, vals in curves["channels"].items():
        bone, ch = name.split(".", 1)
        if bone not in rig.pose.bones:
            continue
        pb, v = rig.pose.bones[bone], vals[f]
        if ch == "quat":
            pb.rotation_quaternion = v
        elif ch == "loc":
            pb.location = v
        elif ch == "bbone":
            for prop, x in zip(BBONE, v):
                setattr(pb, prop, x)
        elif ch == "squash":
            s = max(0.3, 1 - v)
            pb.scale = (1 / math.sqrt(s), s, 1 / math.sqrt(s))


def main():
    a = parse()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    with open(a.curves) as fh:
        curves = json.load(fh)
    body, rig = bpy.data.objects["Body"], bpy.data.objects["Rig"]
    groups = {g.index: g.name for g in body.vertex_groups}
    rest = np.array([v.co[:] for v in body.data.vertices])
    arms = {}
    for side in "LR":
        chain = sorted(b.name for b in rig.data.bones if b.name.startswith(f"arm_{side}_") and not b.name.endswith("_0"))
        if not chain:
            continue
        # the arm proper: points that belong to the arm bones only (not the seam, which blends with the shoulder)
        idx = np.array([v.index for v in body.data.vertices
                        if sum(g.weight for g in v.groups if groups[g.group] in chain) > 0.99])
        h, t = np.array(rig.data.bones[chain[0]].head_local), np.array(rig.data.bones[chain[-1]].tail_local)
        s = (rest[idx] - h) @ (t - h) / np.dot(t - h, t - h)
        bins = [idx[(s >= lo) & (s < lo + 0.1)] for lo in np.arange(0.15, 0.95, 0.1)]
        bins = [x for x in bins if len(x) > 20]
        spread0 = [np.sqrt(((rest[x] - rest[x].mean(0)) ** 2).sum(1).mean()) for x in bins]
        arms[side] = (bins, spread0)
    n = int(curves["n_frames"])
    worst_all, every = 0.0, []
    for f in np.linspace(0, n - 1, min(a.frames, n)).astype(int):
        pose(rig, curves, int(f))
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg).to_mesh()
        P = np.array([v.co[:] for v in ev.vertices])
        body.evaluated_get(dg).to_mesh_clear()
        row = []
        for side, (bins, spread0) in arms.items():
            ch = [np.sqrt(((P[x] - P[x].mean(0)) ** 2).sum(1).mean()) / s0 - 1 for x, s0 in zip(bins, spread0)]
            w = max(ch, key=abs)
            worst_all = max(worst_all, abs(w))
            every.append(abs(w))
            row.append(f"arm_{side} worst {w:+.1%}")
        print(f"frame {int(f):4d}: " + ", ".join(row))
    ok = worst_all <= a.tolerance
    print(f"arm width: worst change {worst_all:.1%}, median {np.median(every):.1%}, 95th percentile "
          f"{np.percentile(every, 95):.1%} -> {'OK' if ok else 'FAIL'} (tolerance {a.tolerance:.0%})")


main()
