"""Check that the arms stay round and outside the body (motion spec: an arm only bends, as a smooth curve, and
passes in front of the body).

For sampled frames of a curves.json, pose the rig and measure per arm:
  fold    the tightest bend along the arm, as (radius of the bend) / (half the arm's thickness): below 1 the inside
          of the bend folds over itself (the arm looks broken), around 1-2 it creases
  kink    the largest turn between two neighbouring B-bone segments (deg): a smooth curve spreads its turn
  inside  how deep arm points sink into the body (cm), and how many points do

  python blender/check_arm_shape.py -- --blend assets/plush_soft.blend --curves runs/x/copy/curves.json
"""

import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_utils import pose_frame  # noqa: E402



def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--curves", required=True)
    p.add_argument("--frames", type=int, default=60, help="how many frames to sample (0 = all)")
    p.add_argument("--out", help="write per-frame numbers here (json)")
    return p.parse_args(argv)


def pose(rig, body, curves, f):
    pose_frame(rig, body, curves["channels"], f)

def main():
    a = parse()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    with open(a.curves) as fh:
        curves = json.load(fh)
    body, rig = bpy.data.objects["Body"], bpy.data.objects["Rig"]
    info = json.load(open(os.path.abspath(a.blend)[:-6] + "_rig.json"))
    height = float(info["height_m"])
    groups = {g.index: g.name for g in body.vertex_groups}
    rest = np.array([v.co[:] for v in body.data.vertices])
    arm_w = {}
    for v in body.data.vertices:
        for g in v.groups:
            nm = groups[g.group]
            if nm.startswith("arm_") and g.weight > 0.01:
                arm_w[v.index] = arm_w.get(v.index, 0) + g.weight
    any_arm = np.array(sorted(arm_w))
    arm_pts = {s: np.array([v.index for v in body.data.vertices
                            if sum(g.weight for g in v.groups if groups[g.group] == f"arm_{s}_1") > 0.99]) for s in "LR"}
    # the body: points that do not move with the arms and are not hair (hair hangs behind, arms pass in front of it)
    hair_like = rest[:, 1] > np.percentile(rest[:, 1], 70)  # rough: the back of the character
    is_body = np.ones(len(rest), bool)
    is_body[any_arm] = False
    is_body &= ~hair_like
    is_body &= rest[:, 2] > 0.02 * height
    # half thickness of each arm (its thinnest direction across the arm, at rest), for the fold ratio
    half = {}
    for s in "LR":
        b = rig.data.bones[f"arm_{s}_1"]
        h, t = np.array(b.head_local), np.array(b.tail_local)
        ax = (t - h) / np.linalg.norm(t - h)
        q = rest[arm_pts[s]] - h
        sv = q @ ax / np.linalg.norm(t - h)
        q = q[(sv > 0.2) & (sv < 0.65)]
        q = q - np.outer(q @ ax, ax)
        q -= q.mean(0)
        ev = np.linalg.eigvalsh(np.cov(q.T))
        half[s] = (float(np.sqrt(ev[1] * 3)), float(np.sqrt(ev[2] * 3)))  # thin, wide half-extents (uniform spread)
    n = int(curves["n_frames"])
    frames = range(n) if a.frames <= 0 else np.unique(np.linspace(0, n - 1, min(a.frames, n)).astype(int))
    from scipy.spatial import cKDTree
    rows = []
    for f in frames:
        f = int(f)
        pose(rig, body, curves, f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg).to_mesh()
        P = np.array([v.co[:] for v in ev.vertices])
        N = np.array([v.normal[:] for v in ev.vertices])
        body.evaluated_get(dg).to_mesh_clear()
        tree = cKDTree(P[is_body])
        bP, bN = P[is_body], N[is_body]
        row = {"frame": f}
        for s in "LR":
            pb = rig.pose.bones[f"arm_{s}_1"]
            segs = pb.bone.bbone_segments
            mats = [pb.bbone_segment_matrix(i, rest=False) for i in range(segs + 1)]
            ys = np.array([(m.to_3x3() @ Vector((0, 1, 0))).normalized()[:] for m in mats])
            seg_len = pb.bone.length / segs
            turn = np.degrees(np.arccos(np.clip((ys[1:] * ys[:-1]).sum(1), -1, 1)))
            # radius of the bend over each segment; the inside folds when it is smaller than the half thickness
            # in the bending plane (between thin and wide: take the thin one, the arm bends flat-wise mostly)
            r = seg_len / np.maximum(np.radians(turn), 1e-6)
            fold = float(r.min() / half[s][0])
            # arm points inside the body: behind the nearest body point's surface
            ap = P[arm_pts[s]]
            d, k = tree.query(ap)
            depth = -((ap - bP[k]) * bN[k]).sum(1)
            near = d < 0.08 * height
            sunk = depth[near & (depth > 0.005 * height)]
            row[s] = {"fold": round(fold, 2), "kink_deg": round(float(turn.max()), 1),
                      "bend_deg": round(float(np.degrees(np.arccos(np.clip(ys[0] @ ys[-1], -1, 1)))), 1),
                      "inside_pts": int(len(sunk)), "inside_cm": round(float(sunk.max() * 100) if len(sunk) else 0.0, 1)}
        rows.append(row)
        print(f"frame {f:4d}: " + "  ".join(
            f"{s}: bend {row[s]['bend_deg']:5.1f} fold {row[s]['fold']:5.2f} kink {row[s]['kink_deg']:4.1f} "
            f"inside {row[s]['inside_pts']:4d} pts {row[s]['inside_cm']:4.1f} cm" for s in "LR"))
    folds = np.array([r[s]["fold"] for r in rows for s in "LR"])
    ins = np.array([r[s]["inside_pts"] for r in rows for s in "LR"])
    print(f"arm shape: tightest bend {folds.min():.2f} x half thickness (folded below 1) on "
          f"{int((folds < 1).sum())} of {len(folds)} arm-frames; arm points inside the body on "
          f"{int((ins > 20).sum())} of {len(ins)} arm-frames (worst {ins.max()} points)")
    if a.out:
        json.dump({"half_thickness": half, "rows": rows}, open(a.out, "w"))


main()
