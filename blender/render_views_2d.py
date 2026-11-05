"""Render the whole 3D model from several sides, as pictures for the 2D puppet (tools/parts_from_image.py cuts them
into parts the same way as the character's own front picture).

Nothing is cut out of the model, so the renders are clean: the whole character as modelled, with the same
camera and light as the 3D copy, turned to each yaw. For each yaw it also writes where each arm shows (the arm
alone, with the rest of the character hiding what it covers), and where the shoulders, the base and the top of
the body are, so the parts line up with the picture.

  python blender/render_views_2d.py -- --blend assets/plush_soft.blend --out assets/plush_views_2d
"""

import argparse
import json
import math
import os
import sys

import bpy  # noqa: I001  (bpy first: it provides bmesh)
import bmesh
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scene_utils import setup_render  # noqa: E402


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--yaws", default="0,45,90,135,180")
    p.add_argument("--px", type=int, default=1200)
    p.add_argument("--samples", type=int, default=96)
    return p.parse_args(argv)


def part(src, keep, name):
    ob = src.copy()
    ob.data = src.data.copy()
    ob.name = ob.data.name = name
    ob.modifiers.clear()
    ob.parent = None
    bpy.context.collection.objects.link(ob)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep[v.index]], context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def main():
    a = parse()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    info = json.load(open(os.path.abspath(a.blend)[:-6] + "_rig.json"))
    height = float(info["height_m"])
    sc = bpy.context.scene
    body = bpy.data.objects["Body"]
    body.modifiers.clear()  # the model as it is (rest pose)
    body.parent = None
    names = {g.index: g.name for g in body.vertex_groups}
    n = len(body.data.vertices)
    arm_w = {s: np.zeros(n) for s in "LR"}
    for v in body.data.vertices:
        for g in v.groups:
            nm = names[g.group]
            if nm.startswith("arm_"):
                arm_w[nm[4]][v.index] += g.weight
    is_arm = {s: arm_w[s] > 0.5 for s in "LR"}
    rest = part(body, ~(is_arm["L"] | is_arm["R"]), "V_rest")
    arms = {s: part(body, is_arm[s], f"V_arm_{s}") for s in "LR"}
    co = np.array([v.co[:] for v in body.data.vertices])
    low = ~(is_arm["L"] | is_arm["R"]) & (co[:, 2] < 0.3 * height)
    cx, cy = float(co[low, 0].mean()), float(co[low, 1].mean())

    setup_render("cycles", a.px, a.px)
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    for o in bpy.data.objects:  # no floor or walls
        if o.type == "MESH" and o not in (body, rest, *arms.values()):
            o.hide_render = True
    cam = sc.camera  # the 3D copy's camera (rig_soft.py), squared up
    sc.render.resolution_x = sc.render.resolution_y = a.px

    def to_px(p):
        q = world_to_camera_view(sc, cam, Vector(p))
        return [round(q.x * a.px, 1), round((1 - q.y) * a.px, 1)]

    def shoot(path, visible, holdout=(), samples=None):
        for o in (body, rest, *arms.values()):
            o.hide_render = o not in visible and o not in holdout
            o.is_holdout = o in holdout
        sc.cycles.samples = samples or a.samples
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)

    meta = {"px": a.px, "views": {}}
    pivot = Matrix.Translation((cx, cy, 0))
    top_body = float(co[~(is_arm["L"] | is_arm["R"]), 2].max())
    for yaw in [int(y) for y in a.yaws.split(",")]:
        turn = pivot @ Matrix.Rotation(math.radians(yaw), 4, "Z") @ pivot.inverted()
        for o in (body, rest, *arms.values()):
            o.matrix_world = turn
        bpy.context.view_layer.update()
        shoot(f"{out}/full_{yaw}.png", [body])
        for s in "LR":  # where this arm shows: the rest of the character hides what it covers
            shoot(f"{out}/armvis_{s}_{yaw}.png", [arms[s]], holdout=[rest, arms["R" if s == "L" else "L"]], samples=8)
        v = {}
        for s in "LR":
            hd = turn @ Vector(next(b["head"] for b in info["bones"] if b["name"] == f"arm_{s}_1"))
            v[f"shoulder_{s}"] = to_px(hd)
            v[f"shoulder_{s}_depth"] = round(float(hd.y - cy), 4)
        v["base"] = to_px((cx, cy, 0.0))
        v["top"] = to_px((cx, cy, top_body))
        v["px_per_m"] = round((v["base"][1] - v["top"][1]) / top_body, 3)
        meta["views"][str(yaw)] = v
    with open(os.path.join(out, "views.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", out)


main()
