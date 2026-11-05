"""Render a rigged character as layered 2D parts, for the 2D puppet (dancebuddy/render2d.py).

The 3D model is only the source of the art: each part is rendered on its own, with the same orthographic camera,
from several views around the character, so the 2D puppet keeps the model's look and soft shading:
  body_<yaw>.png        the body without arms and hair (the face is on it), holes where the arms were filled in
  hair_back_<yaw>.png   all of the hair (drawn behind the body)
  hair_front_<yaw>.png  the hair that lies over the body in that view (drawn in front of it)
  arm_<L|R>.png, arm_<L|R>_fist.png
                        each arm straight out to the side, flat face to the camera (the puppet bends it along an arc)
and parts.json with the camera scale and, per view, where the shoulders, the base and the top of the body are.

  python blender/render_parts_2d.py -- --blend assets/plush_soft.blend --out assets/plush_2d
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
sys.path.insert(0, os.path.dirname(HERE))
exec(open(os.path.join(HERE, "rig_soft.py")).read().split("def main():")[0])  # mesh_arrays, vertex_hsv

VIEWS = (0, 45, 90, 135, 180)  # yaw of the character, deg; the other side is the mirror image


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--px", type=int, default=1100, help="image size (square)")
    p.add_argument("--samples", type=int, default=48)
    return p.parse_args(argv)


def part_object(src, keep: np.ndarray, name: str):
    """A copy of the mesh with only the points in `keep` (and the faces between them)."""
    ob = src.copy()
    ob.data = src.data.copy()
    ob.name = ob.data.name = name
    bpy.context.collection.objects.link(ob)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep[v.index]], context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def show_only(objs):
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_render = o not in objs


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    info = json.load(open(os.path.abspath(a.blend)[:-6] + "_rig.json"))
    height = float(info["height_m"])
    sc = bpy.context.scene
    body, rig = bpy.data.objects["Body"], bpy.data.objects["Rig"]
    for o in list(bpy.data.objects):  # nothing but the character (and lights)
        if o.type == "MESH" and o not in (body,):
            o.hide_render = True

    # ---- parts, by the texture colour (as rig_soft.py) and the arm bones
    co, vuv = mesh_arrays(body.data)
    H, S, V = vertex_hsv(body, vuv)
    names = {g.index: g.name for g in body.vertex_groups}
    arm_w = {s: np.zeros(len(co)) for s in "LR"}
    for v in body.data.vertices:
        for g in v.groups:
            n = names[g.group]
            if n.startswith("arm_"):
                arm_w[n[4]][v.index] += g.weight
    is_arm = {s: arm_w[s] > 0.5 for s in "LR"}
    any_arm = is_arm["L"] | is_arm["R"]
    # hair: dark brown points, and points whose neighbours mostly are (the curls' light highlights would count as
    # skin by their own colour alone; a skin point in shadow is surrounded by skin)
    from scipy.spatial import cKDTree
    dark = (H < 40) & (S > 0.45) & (V < 0.45)
    nb = cKDTree(co).query(co, k=24)[1]
    is_hair = (dark | (dark[nb].mean(1) > 0.4)) & (co[:, 2] > 0.15 * height) & ~any_arm
    is_body = ~any_arm & ~is_hair
    parts = {"body": part_object(body, is_body, "P_body"), "hair": part_object(body, is_hair, "P_hair"),
             "arm_L": part_object(body, is_arm["L"], "P_arm_L"), "arm_R": part_object(body, is_arm["R"], "P_arm_R")}
    body.hide_render = True

    # ---- camera: orthographic, looking along +Y at the character
    cam = sc.camera
    cam.data.type = "ORTHO"
    span = 2.4 * height / 1.436
    cam.data.ortho_scale = span
    centre_z = 0.55 * height
    cam.location = Vector((0.0, -10.0, centre_z))
    cam.rotation_euler = (math.pi / 2, 0, 0)
    sys.path.insert(0, HERE)
    from scene_utils import setup_render
    setup_render("cycles", a.px, a.px)
    sc.cycles.samples = a.samples
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    for o in bpy.data.objects:  # no floor or walls in the parts
        if o.type == "MESH" and not o.name.startswith("P_"):
            o.hide_render = True

    def to_px(p):
        q = world_to_camera_view(sc, cam, Vector(p))
        return [round(q.x * a.px, 1), round((1 - q.y) * a.px, 1)]

    cx, cy = float(np.mean(co[is_body & (co[:, 2] < 0.3 * height), 0])), float(np.mean(co[is_body & (co[:, 2] < 0.3 * height), 1]))
    meta = {"px": a.px, "metres_per_px": span / a.px, "height_m": height, "views": {}, "arms": {}}
    pivot = Matrix.Translation((cx, cy, 0))
    for yaw in VIEWS:
        turn = pivot @ Matrix.Rotation(math.radians(yaw), 4, "Z") @ pivot.inverted()
        rig.matrix_world = turn
        for o in parts.values():
            o.matrix_world = Matrix.Identity(4)
        bpy.context.view_layer.update()
        # the parts are children of nothing: turn their points with the rig through the armature modifier
        for o in parts.values():
            o.matrix_world = turn
        bpy.context.view_layer.update()
        show_only([parts["body"]])
        render(f"{os.path.abspath(a.out)}/body_{yaw}.png")
        show_only([parts["hair"]])
        render(f"{os.path.abspath(a.out)}/hair_back_{yaw}.png")
        # hair in front of the body: the hair with the body as a holdout (the body hides what is behind it)
        show_only([parts["hair"], parts["body"]])
        parts["body"].is_holdout = True
        render(f"{os.path.abspath(a.out)}/hair_vis_{yaw}.png")
        parts["body"].is_holdout = False
        anchors = {}
        for s in "LR":
            hd = turn @ Vector(next(b["head"] for b in info["bones"] if b["name"] == f"arm_{s}_1"))
            anchors[f"shoulder_{s}"] = to_px(hd)
            anchors[f"shoulder_{s}_depth"] = round(float(hd.y - cy), 4)  # + = behind the body's centre
        anchors["base"] = to_px((cx, cy, 0.0))
        anchors["top"] = to_px((cx, cy, float(co[is_body, 2].max())))
        meta["views"][str(yaw)] = anchors
    rig.matrix_world = Matrix.Identity(4)
    for o in parts.values():
        o.matrix_world = Matrix.Identity(4)

    # ---- arms: straight out to the side, flat face to the camera, open and as a fist
    from scipy.spatial.transform import Rotation as Rot
    from dancebuddy.retarget import _local, arm_pose, socket_offset, socket_rot
    M = {b["name"]: np.array(b["matrix"]) for b in info["bones"]}
    M.update({b["name"] + "__head": b["head"] for b in info["bones"]})
    M.update({b["name"] + "__tail": b["tail"] for b in info["bones"]})
    props = ("bbone_curveinx", "bbone_curveinz", "bbone_curveoutx", "bbone_curveoutz", "bbone_rollin",
             "bbone_rollout", "bbone_easein", "bbone_easeout")
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    for s, sign in (("L", 1), ("R", -1)):
        u = np.array([sign, 0.0, 0.0])
        c0 = socket_rot(M, [f"arm_{s}_1"], u, u, sign)
        q, curve, chord = arm_pose(M, f"arm_{s}_1", u, u, sign, parent=c0, face=info["hand_face"][s])
        p0, p1 = rig.pose.bones[f"arm_{s}_0"], rig.pose.bones[f"arm_{s}_1"]
        p0.rotation_quaternion = _local(M[f"arm_{s}_0"], Rot.identity(), c0)
        p0.location = M[f"arm_{s}_0"].T @ socket_offset(u, sign, height)
        p1.rotation_quaternion = q
        for prop, x in zip(props, curve):
            setattr(p1, prop, x)
    bpy.context.view_layer.update()
    keys = parts[f"arm_L"].data.shape_keys
    for s in "LR":
        ob = parts[f"arm_{s}"]
        pb = rig.pose.bones[f"arm_{s}_1"]
        root, tip = rig.matrix_world @ pb.head, rig.matrix_world @ pb.tail
        show_only([ob])
        for fist in (0.0, 1.0):
            k = ob.data.shape_keys
            if k and f"fist_{s}" in k.key_blocks:
                k.key_blocks[f"fist_{s}"].value = fist
            bpy.context.view_layer.update()
            render(f"{os.path.abspath(a.out)}/arm_{s}{'_fist' if fist else ''}.png")
        meta["arms"][s] = {"root": to_px(root), "tip": to_px(tip)}
    with open(os.path.join(a.out, "parts.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", a.out)


main()
