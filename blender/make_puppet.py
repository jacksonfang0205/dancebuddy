"""Turn a character picture into a rigged 2D puppet (a flat cut-out mesh with a
4-bone spine), so the same plans and renderer work on any drawn character.

Run `python -m dancebuddy puppet PICTURE assets/<name>` rather than this file
directly: that first cuts the character out (dancebuddy/puppet.py), then runs:
  blender -b -P blender/make_puppet.py -- --out assets/<name>

Writes <out>.blend, <out>_rig.json and <out>_views/front.png.
"""

import argparse
import json
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_utils import reset_scene, setup_render, setup_stage  # noqa: E402

HEIGHT = 1.9
BONE_NAMES = ["root", "lower", "upper", "top"]
BONE_SPLITS = [0.0, 0.18, 0.45, 0.72, 1.0]  # fraction of the height where each bone starts/ends


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--engine", default="auto")
    return p.parse_args(argv)


def build(info, sprite_path):
    rows, cols, grid = info["rows"], info["cols"], info["grid"]
    cell = HEIGHT / rows
    width = cell * cols
    occupied = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c]]
    centre_x = (sum(c for _, c in occupied) / len(occupied) + 0.5) * cell - width / 2

    verts, faces, uvs, index = [], [], [], {}

    def vid(r, c):  # r, c are grid-corner indices; r = 0 is the top edge
        if (r, c) not in index:
            index[(r, c)] = len(verts)
            verts.append((c * cell - width / 2 - centre_x, 0.0, HEIGHT - r * cell))
            uvs.append((c / cols, 1 - r / rows))
        return index[(r, c)]

    for r, c in occupied:
        faces.append((vid(r + 1, c), vid(r + 1, c + 1), vid(r, c + 1), vid(r, c)))
    mesh = bpy.data.meshes.new("Puppet")
    mesh.from_pydata(verts, [], faces)
    uv = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            uv.data[li].uv = uvs[mesh.loops[li].vertex_index]
    body = bpy.data.objects.new("Body", mesh)
    bpy.context.collection.objects.link(body)

    # material: the picture, with its transparency
    mat = bpy.data.materials.new("Sprite")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(sprite_path)
    tex.image.alpha_mode = "STRAIGHT"
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.9
    for attr, val in (("surface_render_method", "DITHERED"), ("blend_method", "HASHED")):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass
    body.data.materials.append(mat)

    # spine rig
    arm_data = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", arm_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    prev = None
    bones = []
    for name, f0, f1 in zip(BONE_NAMES, BONE_SPLITS[:-1], BONE_SPLITS[1:]):
        b = arm_data.edit_bones.new(name)
        b.head, b.tail = Vector((0, 0, f0 * HEIGHT)), Vector((0, 0, f1 * HEIGHT))
        if prev is not None:
            b.parent, b.use_connect = prev, True
        prev = b
        bones.append((name, f0 * HEIGHT, f1 * HEIGHT))
    bpy.ops.object.mode_set(mode="OBJECT")

    centres = [(z0 + z1) / 2 for _, z0, z1 in bones]
    groups = {n: body.vertex_groups.new(name=n) for n, _, _ in bones}
    for v in body.data.vertices:
        z = v.co.z
        ws = []
        for i, c in enumerate(centres):
            half = (centres[min(i + 1, len(centres) - 1)] - centres[max(i - 1, 0)]) / 2 or 0.5
            w = max(0.0, 1 - abs(z - c) / half)
            if (i == 0 and z < c) or (i == len(centres) - 1 and z > c):
                w = 1.0
            ws.append(w)
        s = sum(ws) or 1.0
        for (n, _, _), w in zip(bones, ws):
            if w > 0:
                groups[n].add([v.index], w / s, "REPLACE")
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    body.parent = rig
    return bones, width


def main():
    global HEIGHT
    a = parse()
    out = os.path.abspath(a.out)
    info = json.load(open(out + "_grid.json"))
    # keep wide characters inside the portrait frame: at most 1.6 m wide
    HEIGHT = min(HEIGHT, 1.6 * info["height_px"] / info["width_px"])
    sprite = os.path.join(os.path.dirname(out), info["sprite"])
    reset_scene()
    bones, width = build(info, sprite)
    setup_stage(target_z=HEIGHT / 2, height=HEIGHT)
    setup_render(a.engine, 512, 512)
    bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")
    views_dir = out + "_views"
    os.makedirs(views_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = os.path.join(views_dir, "front.png")
    bpy.ops.render.render(write_still=True)
    name = os.path.basename(out)
    rig_info = {
        "name": name,
        "kind": "2D cut-out puppet made from a picture (flat, seen from the front)",
        "height_m": HEIGHT,
        "width_m": round(width, 3),
        "root_bone": "root",
        "bones": [{"name": n, "parent": bones[i - 1][0] if i else None, "head_z": round(z0, 3), "tail_z": round(z1, 3),
                   "length": round(z1 - z0, 3)} for i, (n, z0, z1) in enumerate(bones)],
        "channels_per_bone": ["lean", "twist", "sway"],
        "root_only_channels": ["lift", "squash"],
        # flat puppets look broken when turned or leaned far out of the picture plane
        "limits": {"sway": 35, "lean": 20, "twist": 25, "lift": 0.35, "squash": 0.35},
        "notes": "sway (side to side, in the picture plane) reads best on a flat puppet; lean and twist move it out of the plane, keep them small",
        "views": {"original_picture": info["sprite"], "front": os.path.relpath(os.path.join(views_dir, "front.png"), os.path.dirname(out))},
    }
    with open(out + "_rig.json", "w") as f:
        json.dump(rig_info, f, indent=1)
    print(f"wrote {out}.blend and {out}_rig.json")


main()
