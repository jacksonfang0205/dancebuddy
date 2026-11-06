"""Build the "capsule" test character: a pill-shaped body with eyes, no arms,
no legs, no neck, rigged with a 4-bone chain (root -> lower -> upper -> top).

Run with either:
  blender -b -P blender/make_capsule.py -- --out assets/capsule
  python blender/make_capsule.py --out assets/capsule        (if the bpy module is installed)

Writes <out>.blend, <out>_rig.json (bones for the character card) and
<out>_views/{front,side,three_quarter,back}.png for the VLM.
"""

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_utils import reset_scene, setup_render, setup_stage  # noqa: E402

BONES = [  # name, head z, tail z
    ("root", 0.0, 0.35),
    ("lower", 0.35, 0.85),
    ("upper", 0.85, 1.35),
    ("top", 1.35, 1.9),
]
HEIGHT = 1.9


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets/capsule")
    p.add_argument("--engine", default="auto")
    p.add_argument("--color", default="0.95,0.55,0.25")
    return p.parse_args(argv)


def material(name, rgb, rough=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = rough
    m.diffuse_color = (*rgb, 1)  # used by the workbench engine
    return m


def build(color):
    # body: a stretched UV sphere, a pill with no neck or waist
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, radius=0.5, location=(0, 0, HEIGHT / 2))
    body = bpy.context.active_object
    body.name = "Body"
    body.scale = (1.0, 0.9, HEIGHT)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
    body.data.materials.append(material("Skin", color))
    # eyes and a mouth, joined into the body so they deform with it
    parts = []
    for x in (-0.15, 0.15):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.075, location=(x, -0.405, 1.42))
        eye = bpy.context.active_object
        eye.data.materials.append(material("Eye", (0.03, 0.03, 0.03), 0.2))
        parts.append(eye)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.025, location=(x + 0.02, -0.47, 1.45))
        glint = bpy.context.active_object
        glint.data.materials.append(material("Glint", (1, 1, 1), 0.1))
        parts.append(glint)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.08, minor_radius=0.018, location=(0, -0.43, 1.25),
                                     rotation=(math.radians(90), 0, 0))
    mouth = bpy.context.active_object
    mouth.data.materials.append(material("Mouth", (0.25, 0.05, 0.05), 0.4))
    parts.append(mouth)
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()

    # armature
    arm_data = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", arm_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    prev = None
    for name, z0, z1 in BONES:
        b = arm_data.edit_bones.new(name)
        b.head, b.tail = Vector((0, 0, z0)), Vector((0, 0, z1))
        b.roll = 0
        if prev is not None:
            b.parent = prev
            b.use_connect = True
        prev = b
    bpy.ops.object.mode_set(mode="OBJECT")

    # skin weights by height: smooth tent functions centred on each bone
    centres = [(z0 + z1) / 2 for _, z0, z1 in BONES]
    groups = {name: body.vertex_groups.new(name=name) for name, _, _ in BONES}
    for v in body.data.vertices:
        z = (body.matrix_world @ v.co).z
        ws = []
        for i, c in enumerate(centres):
            half = (centres[min(i + 1, len(centres) - 1)] - centres[max(i - 1, 0)]) / 2 or 0.5
            w = max(0.0, 1 - abs(z - c) / half)
            if (i == 0 and z < c) or (i == len(centres) - 1 and z > c):
                w = 1.0
            ws.append(w)
        s = sum(ws) or 1.0
        for (name, _, _), w in zip(BONES, ws):
            if w > 0:
                groups[name].add([v.index], w / s, "REPLACE")
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    body.parent = rig
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
    return body, rig


def main():
    a = parse()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    reset_scene()
    color = tuple(float(c) for c in a.color.split(","))
    build(color)
    setup_stage(target_z=HEIGHT / 2, height=HEIGHT)
    setup_render(a.engine, 512, 512)
    bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")

    rig_info = {
        "name": "capsule",
        "height_m": HEIGHT,
        "root_bone": "root",
        "bones": [{"name": n, "parent": BONES[i - 1][0] if i else None,
                   "head_z": z0, "tail_z": z1, "length": round(z1 - z0, 3)} for i, (n, z0, z1) in enumerate(BONES)],
        "channels_per_bone": ["lean", "twist", "sway"],
        "root_only_channels": ["lift", "squash"],
        "limits": {"lean": 35, "twist": 45, "sway": 35, "lift": 0.35, "squash": 0.35},
        "views": {},
    }
    views_dir = out + "_views"
    os.makedirs(views_dir, exist_ok=True)
    scene = bpy.context.scene
    cam = scene.camera
    dist = (cam.location - Vector((0, 0, HEIGHT / 2))).length
    for view, ang in [("front", 0), ("three_quarter", 40), ("side", 90), ("back", 180)]:
        r = math.radians(ang)
        cam.location = Vector((dist * math.sin(r), -dist * math.cos(r), cam.location.z))
        direction = Vector((0, 0, HEIGHT / 2)) - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        path = os.path.join(views_dir, f"{view}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        rig_info["views"][view] = os.path.relpath(path, os.path.dirname(out))
    with open(out + "_rig.json", "w") as f:
        json.dump(rig_info, f, indent=1)
    print(f"wrote {out}.blend, {out}_rig.json and {len(rig_info['views'])} views")


main()
