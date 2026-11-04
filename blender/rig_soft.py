"""Rig a textured character the way it moves, without changing its shape.

The rig follows the character's motion spec (assets/<name>_motion_spec.json), which was learned by watching it dance:
  * body      : root at the base (tilt, turn, hop, squash) -> body_lower -> body_upper (barely bends)
  * shoulders : arm_X_0, hidden in the body and not deforming; turning it slides the arm's root up the body side
                (a raised arm starts beside the face)
  * arms      : arm_X_1..3, B-bones, so each arm bends as one smooth curve (no elbow)
  * hair      : physically simulated: a coarse closed shell around the hair is a cloth (pinned at the scalp, with
                gravity, inertia, inner pressure and collisions with the body) and the real hair follows it
                (Surface Deform). Its physical properties come from the motion spec, not from any dance.

The mesh is never edited. Parts are found from the base-colour texture (arms = blue, hair = dark brown). Weights
come from Blender's heat diffusion ("automatic weights") computed on a closed voxel copy of the body (generated
meshes are open and in many pieces, which heat weighting cannot handle) and copied back to every vertex by
position, so the original surface deforms smoothly and seams stay closed.

  python blender/rig_soft.py -- --blend assets/plush.blend --out assets/plush_soft
"""

import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_utils import setup_render  # noqa: E402

FORWARD = Vector((0, -1, 0))  # the character faces the camera, which looks along +Y


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True, help="scene with a textured mesh called Body (any old rig is removed)")
    p.add_argument("--out", required=True, help="output prefix: writes <out>.blend and <out>_rig.json")
    p.add_argument("--arm-hue", type=float, nargs=2, default=[190, 260], help="hue range (deg) of the arms")
    p.add_argument("--hair-max-value", type=float, default=0.45, help="hair is darker than this (HSV value)")
    p.add_argument("--voxel", type=float, default=0.012, help="voxel size of the weighting copy, x height")
    p.add_argument("--spec", default="", help="motion spec with the hair's physical properties "
                   "(default: <out dir>/<name>_motion_spec.json, name = out name before '_')")
    return p.parse_args(argv)


# ------------------------------------------------------------------ mesh data

def mesh_arrays(me):
    nv = len(me.vertices)
    co = np.zeros(nv * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nl = len(me.loops)
    lv = np.zeros(nl, int)
    me.loops.foreach_get("vertex_index", lv)
    uv = np.zeros(nl * 2)
    me.uv_layers.active.data.foreach_get("uv", uv)
    vuv = np.zeros((nv, 2))
    vuv[lv] = uv.reshape(-1, 2)
    return co, vuv


def base_colour_image(body):
    for mat in body.data.materials:
        if mat and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    links = node.inputs["Base Color"].links
                    if links and links[0].from_node.type == "TEX_IMAGE":
                        return links[0].from_node.image
    raise SystemExit("no base-colour texture found on Body")


def vertex_hsv(body, vuv, size=1024):
    img = base_colour_image(body).copy()
    img.scale(size, size)
    px = np.zeros(size * size * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(size, size, 4)
    bpy.data.images.remove(img)
    x = np.clip((vuv[:, 0] % 1) * (size - 1), 0, size - 1).astype(int)
    y = np.clip((vuv[:, 1] % 1) * (size - 1), 0, size - 1).astype(int)
    rgb = np.clip(px[y, x, :3], 0, 1)
    mx, mn = rgb.max(1), rgb.min(1)
    d = mx - mn + 1e-9
    r, g, b = rgb.T
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, d / (mx + 1e-9), 0)
    return h, s, mx


def centre_line(p, start, n_pts=4):
    """Points along a limb from `start` to its far end (centroids of slices)."""
    tip = p[np.argmax(np.linalg.norm(p - start, axis=1))]
    axis = (tip - start) / np.linalg.norm(tip - start)
    s = (p - start) @ axis / np.linalg.norm(tip - start)
    pts = [start]
    for t in np.linspace(0, 1, n_pts)[1:]:
        t = min(t, 0.97)
        sel = np.abs(s - t) < 0.06
        pts.append(p[sel].mean(0) if sel.sum() > 5 else start + t * (tip - start))
    return pts


def chain_weights(s, joints, blend=0.12):
    """Weights for a bone chain from a parameter s along it. joints = [s0, s1, ..., sK] (bone k spans s_k..s_k+1)."""
    k = len(joints) - 1
    W = np.zeros((len(s), k))
    for i in range(k):
        lo, hi = joints[i], joints[i + 1]
        w = np.ones(len(s))
        if i > 0:
            w *= np.clip((s - (lo - blend / 2)) / blend, 0, 1)
        if i < k - 1:
            w *= np.clip(((hi + blend / 2) - s) / blend, 0, 1)
        W[:, i] = w
    return W / np.maximum(W.sum(1, keepdims=True), 1e-9)


HAIR_PHYSICS = {  # defaults; the motion spec's physics.hair overrides them (properties of the character)
    "cage_voxel": 0.045, "collider_voxel": 0.035,  # x height
    "pin_top": 0.12, "pin_free": 0.38,            # below the crown, x height: pinned above pin_top, loosest below pin_free
    "pin_min": 0.25,                              # even the loosest hair is pulled back toward its shape (it springs back)
    "gravity": 0.2,                               # the modelled shape already hangs; gravity only adds a little sag
    "quality": 8, "mass": 0.15, "air_damping": 1.0, "tension": 30.0, "compression": 30.0, "shear": 15.0,
    "bending": 20.0, "pin_stiffness": 3.0, "pressure": 5.0, "collision_distance": 0.01,
}


def _part_copy(body, keep, name, voxel, one_piece=False):
    """A closed coarse copy of part of the body (voxel remesh of the kept vertices), not rendered."""
    import bmesh

    obj = body.copy()
    obj.data = body.data.copy()
    obj.name = name
    obj.modifiers.clear()
    obj.vertex_groups.clear()
    obj.parent = None
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(~keep)[0]], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    rm = obj.modifiers.new("Remesh", "REMESH")
    rm.mode, rm.voxel_size = "VOXEL", voxel
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=rm.name)
    # clean, all-triangle surface (Surface Deform rejects concave or degenerate polygons)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=voxel * 0.05)
    bmesh.ops.dissolve_degenerate(bm, dist=voxel * 0.05, edges=bm.edges)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bad = [e for e in bm.edges if len(e.link_faces) > 2]
    if bad:
        bmesh.ops.delete(bm, geom=list({f for e in bad for f in e.link_faces}), context="FACES")
    # one piece only: a loose piece of a simulated shell has nothing holding it and would fall away
    islands, seen = [], set()
    for v in bm.verts:
        if v in seen:
            continue
        stack, comp = [v], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(e.other_vert(x) for e in x.link_edges)
        islands.append(comp)
    islands.sort(key=len, reverse=True)
    if one_piece and len(islands) > 1:
        bmesh.ops.delete(bm, geom=[v for comp in islands[1:] for v in comp], context="VERTS")
    print(f"{name}: {sum(map(len, islands))} vertices in {len(islands)} pieces" + (", kept the main one" if one_piece else ""))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.materials.clear()
    obj.hide_render = True
    obj.display_type = "WIRE"
    return obj


def add_hair_physics(body, rig, co, is_hair, W, deform, height, phys):
    from scipy.spatial import cKDTree

    p = {**HAIR_PHYSICS, **phys}
    # the body (and arms) the hair collides with, deformed by the rig like the body
    col = _part_copy(body, ~is_hair, "BodyCollider", p["collider_voxel"] * height)
    cco = np.array([v.co[:] for v in col.data.vertices])
    _, nn = cKDTree(co[~is_hair]).query(cco, k=1)
    Wn = W[~is_hair][nn]
    for j, n in enumerate(deform):
        g = col.vertex_groups.new(name=n)
        for vi in np.nonzero(Wn[:, j] > 1e-3)[0]:
            g.add([int(vi)], float(Wn[vi, j]), "REPLACE")
    am = col.modifiers.new("Armature", "ARMATURE")
    am.object = rig
    col.modifiers.new("Collision", "COLLISION")
    col.collision.thickness_outer = p["collision_distance"]
    col.parent = rig

    # the hair shell: follows the head, pinned at the scalp, free toward the ends
    cage = _part_copy(body, is_hair, "HairCage", p["cage_voxel"] * height, one_piece=True)
    gco = np.array([v.co[:] for v in cage.data.vertices])
    crown = float(co[is_hair, 2].max())
    z_pin, z_free = crown - p["pin_top"] * height, crown - p["pin_free"] * height
    pin = np.clip((gco[:, 2] - z_free) / (z_pin - z_free), 0, 1)
    pin = p["pin_min"] + (1 - p["pin_min"]) * pin * pin * (3 - 2 * pin)
    gu = cage.vertex_groups.new(name="body_upper")
    gu.add(list(range(len(gco))), 1.0, "REPLACE")
    gp = cage.vertex_groups.new(name="pin")
    for vi in range(len(gco)):
        gp.add([vi], float(pin[vi]), "REPLACE")
    am = cage.modifiers.new("Armature", "ARMATURE")
    am.object = rig
    cl = cage.modifiers.new("Cloth", "CLOTH")
    st = cl.settings
    st.quality, st.mass, st.air_damping = int(p["quality"]), p["mass"], p["air_damping"]
    st.tension_stiffness, st.compression_stiffness = p["tension"], p["compression"]
    st.shear_stiffness, st.bending_stiffness = p["shear"], p["bending"]
    st.vertex_group_mass, st.pin_stiffness = "pin", p["pin_stiffness"]
    st.use_pressure, st.uniform_pressure_force = True, p["pressure"]
    st.effector_weights.gravity = p["gravity"]
    cl.collision_settings.use_collision = True
    cl.collision_settings.distance_min = p["collision_distance"]
    cl.point_cache.frame_start, cl.point_cache.frame_end = 1, 100000
    cage.parent = rig
    print(f"hair physics: shell {len(gco)} vertices ({int((pin > 0.99).sum())} fully pinned), collider {len(cco)}")

    # the real hair follows the shell
    # only points inside the hair mass (dark brown details on the face, like eyelashes, stay on the face)
    near, _ = cKDTree(gco).query(co, k=1)
    in_mass = is_hair & (near < 1.5 * p["cage_voxel"] * height)
    hg = body.vertex_groups.new(name="hair_sim")
    for vi in np.nonzero(in_mass)[0]:
        hg.add([int(vi)], 1.0, "REPLACE")
    print(f"hair points following the simulation: {int(in_mass.sum())} of {int(is_hair.sum())}")
    sd = body.modifiers.new("HairSim", "SURFACE_DEFORM")
    sd.target = cage
    sd.vertex_group = "hair_sim"
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.surfacedeform_bind(modifier=sd.name)
    print(f"hair bound to its shell: {sd.is_bound}")


# ------------------------------------------------------------------ build

def main():
    a = parse()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    body = bpy.data.objects["Body"]
    for o in list(bpy.data.objects):
        if o.type == "ARMATURE":
            bpy.data.objects.remove(o, do_unlink=True)
    for mod in list(body.modifiers):
        if mod.type == "ARMATURE":
            body.modifiers.remove(mod)
    body.parent = None
    body.vertex_groups.clear()
    me = body.data

    co, vuv = mesh_arrays(me)
    H, S, V = vertex_hsv(body, vuv)
    height = float(co[:, 2].max())
    xmax = float(np.abs(co[:, 0]).max())
    is_arm = (H > a.arm_hue[0]) & (H < a.arm_hue[1]) & (S > 0.3) & (np.abs(co[:, 0]) > 0.25 * xmax)
    is_hair = (H < 40) & (S > 0.45) & (V < a.hair_max_value) & (co[:, 2] > 0.15 * height) & ~is_arm
    is_body = ~is_arm & ~is_hair
    low = is_body & (co[:, 2] < 0.3 * height)
    cx, cy = float(co[low, 0].mean()), float(co[low, 1].mean())
    print(f"vertices: arms {is_arm.sum()}, hair {is_hair.sum()}, body {is_body.sum()}; body centre ({cx:.3f}, {cy:.3f})")

    # ---- skeleton geometry
    z_low, z_mid = 0.2 * height, 0.55 * height
    body_top = float(co[is_body, 2].max())
    spine = [("root", (cx, cy, 0), (cx, cy, z_low), None),
             ("body_lower", (cx, cy, z_low), (cx, cy, z_mid), "root"),
             ("body_upper", (cx, cy, z_mid), (cx, cy, body_top), "body_lower")]
    arms = {}
    for side, sign in (("L", 1), ("R", -1)):
        p = co[is_arm & (np.sign(co[:, 0] - cx) == sign)]
        inner = p[np.argsort(sign * (p[:, 0] - cx))[: max(10, len(p) // 12)]]
        root_pt = inner.mean(0)                           # where the arm meets the body
        pivot = np.array([cx + sign * 0.15 * abs(root_pt[0] - cx), root_pt[1], root_pt[2]])  # inside the body
        arms[side] = {"pivot": pivot, "pts": centre_line(p, root_pt)}
        print(f"arm_{side}: root {np.round(root_pt, 3)}, tip {np.round(arms[side]['pts'][-1], 3)}")
    # ---- armature
    arm_data = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", arm_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    for name, h, t, parent in spine:
        b = eb.new(name)
        b.head, b.tail, b.roll = Vector(h), Vector(t), 0
        if parent:
            b.parent, b.use_connect = eb[parent], True
    for side, d in arms.items():
        sh = eb.new(f"arm_{side}_0")
        sh.head, sh.tail = Vector(d["pivot"]), Vector(d["pts"][0])
        sh.align_roll(FORWARD)
        sh.parent = eb["body_lower"]
        prev = sh
        for i in range(3):
            b = eb.new(f"arm_{side}_{i + 1}")
            b.head, b.tail = Vector(d["pts"][i]), Vector(d["pts"][i + 1])
            b.align_roll(FORWARD)
            b.bbone_segments = 6
            b.parent, b.use_connect = prev, True
            prev = b
    bpy.ops.object.mode_set(mode="OBJECT")

    # ---- weights
    # Body and hair: heat diffusion on a closed voxel copy, copied back by position. The arms do not take part:
    # in the copy they are fused to the body, so heat would spread them over half of it. Their weights are set
    # below the way a sewn-on plush arm moves.
    arm_bones = [b.name for b in arm_data.bones if b.name.startswith("arm_")]
    for n in arm_bones:
        arm_data.bones[n].use_deform = False
    proxy = body.copy()
    proxy.data = body.data.copy()
    proxy.name = "WeightProxy"
    proxy.modifiers.clear()
    bpy.context.collection.objects.link(proxy)
    rm = proxy.modifiers.new("Remesh", "REMESH")
    rm.mode, rm.voxel_size = "VOXEL", a.voxel * height
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = proxy
    proxy.select_set(True)
    bpy.ops.object.modifier_apply(modifier=rm.name)
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(proxy.data)
    islands, seen = [], set()
    for v in bm.verts:  # keep the main shell only: loose bits make heat weighting fail
        if v.index in seen:
            continue
        stack, comp = [v], []
        while stack:
            x = stack.pop()
            if x.index in seen:
                continue
            seen.add(x.index)
            comp.append(x)
            stack.extend(e.other_vert(x) for e in x.link_edges)
        islands.append(comp)
    islands.sort(key=len, reverse=True)
    bmesh.ops.delete(bm, geom=[v for comp in islands[1:] for v in comp], context="VERTS")
    bm.to_mesh(proxy.data)
    bm.free()
    print(f"weighting copy: {len(proxy.data.vertices)} vertices in its main shell ({len(islands)} shells before)")
    # heat weighting is unreliable at small scales: compute it on a 10x copy of the scene
    proxy.scale = rig.scale = (10, 10, 10)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    rig.scale = (1, 1, 1)

    deform = [b.name for b in arm_data.bones if b.use_deform]
    pco = np.zeros(len(proxy.data.vertices) * 3)
    proxy.data.vertices.foreach_get("co", pco)
    pco = pco.reshape(-1, 3)
    PW = np.zeros((len(pco), len(deform)))
    gidx = {g.index: deform.index(g.name) for g in proxy.vertex_groups if g.name in deform}
    for v in proxy.data.vertices:
        for g in v.groups:
            if g.group in gidx:
                PW[v.index, gidx[g.group]] = g.weight
    empty = PW.sum(1) < 1e-6
    print(f"heat weights: {int((~empty).sum())} / {len(pco)} copy vertices weighted")
    from scipy.spatial import cKDTree
    tree = cKDTree(pco[~empty])
    dist, nn = tree.query(co, k=6)
    wgt = 1 / np.maximum(dist, 1e-5)
    W = (PW[~empty][nn] * wgt[..., None]).sum(1) / wgt.sum(1, keepdims=True)
    bpy.data.objects.remove(proxy, do_unlink=True)


    # Arms, like a sewn-on plush arm: the arm bends at its own root (soft tube), not the body. Along the arm,
    # the weight moves from the socket (arm_X_0) at the seam to the arm chain a quarter of the way out; the
    # socket also carries a narrow band of body around the seam, so the seam slides up the body with it.
    for n in arm_bones:
        arm_data.bones[n].use_deform = True
    deform = deform + arm_bones
    W = np.hstack([W, np.zeros((len(co), len(arm_bones)))])
    col = {n: i for i, n in enumerate(deform)}
    from scipy.spatial import cKDTree as _T
    for side, d in arms.items():
        m = is_arm & (np.sign(co[:, 0] - cx) == (1 if side == "L" else -1))
        pts = np.array([list(v) for v in d["pts"]])
        lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        seg = np.r_[0, np.cumsum(lens)]
        best, s_par = np.full(len(co), np.inf), np.zeros(len(co))
        for i in range(3):
            ab = pts[i + 1] - pts[i]
            t = np.clip((co - pts[i]) @ ab / (ab @ ab), 0, 1)
            dd = np.linalg.norm(co - (pts[i] + t[:, None] * ab), axis=1)
            take = dd < best
            best[take], s_par[take] = dd[take], (seg[i] + t[take] * lens[i]) / seg[-1]
        # distance from the seam, measured along the arm: 0 at the arm's innermost points
        s0 = np.percentile(s_par[m], 3)
        s_rel = np.clip((s_par - s0) / (1 - s0 + 1e-9), 0, 1)
        on = np.clip(s_rel / 0.35, 0, 1)
        on = on * on * (3 - 2 * on)
        Wa = chain_weights(s_rel, list((seg / seg[-1])), blend=0.4)  # wide: the arm bends as one curve
        W[m] = 0
        W[m, col[f"arm_{side}_0"]] = 1 - on[m]
        for i in range(3):
            W[m, col[f"arm_{side}_{i + 1}"]] = on[m] * Wa[m, i]
        # the body band around the seam follows the socket a little
        dist, _ = _T(co[m]).query(co, k=1)
        band = is_body & (dist < 0.05 * height)
        f = 0.6 * (1 - dist[band] / (0.05 * height))
        W[band] *= (1 - f)[:, None]
        W[band, col[f"arm_{side}_0"]] += f
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)

    groups = {n: body.vertex_groups.new(name=n) for n in deform}
    for j, n in enumerate(deform):
        for vi in np.nonzero(W[:, j] > 1e-3)[0]:
            groups[n].add([int(vi)], float(W[vi, j]), "REPLACE")
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    mod.use_deform_preserve_volume = True
    body.parent = rig
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"

    spec_path = a.spec or os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                       os.path.basename(a.out).split("_")[0] + "_motion_spec.json")
    phys = {}
    if os.path.exists(spec_path):
        with open(spec_path) as f:
            phys = json.load(f).get("physics", {}).get("hair", {})
    add_hair_physics(body, rig, co, is_hair, W, deform, height, phys)

    # frame the character with room for raised arms (portrait, like a phone dance clip)
    cam = bpy.context.scene.camera
    if cam is not None:
        target = Vector((0, 0, 0.62 * height))
        d = (cam.location - target).normalized()
        cam.data.lens = 45
        cam.location = target + d * 3.0 * height
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    setup_render("auto", 512, 512)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out) + ".blend")

    info = {
        "name": os.path.basename(a.out),
        "kind": "3D model rigged from its motion spec (rig_soft.py): base-pivot body, sliding shoulder sockets, "
                "3-bone B-bone arms, cloth-simulated hair; mesh unchanged, heat weights",
        "height_m": round(height, 3),
        "root_bone": "root",
        "bones": [{"name": b.name, "parent": b.parent.name if b.parent else None, "deform": b.use_deform,
                   "head": [round(c, 3) for c in b.head_local], "tail": [round(c, 3) for c in b.tail_local],
                   "length": round(b.length, 3),
                   # rest orientation in armature space (columns = the bone's local x, y, z axes)
                   "matrix": [[round(b.matrix_local[i][j], 5) for j in range(3)] for i in range(3)]}
                  for b in arm_data.bones],
        "parts": {"arms": int(is_arm.sum()), "hair": int(is_hair.sum()), "body": int(is_body.sum())},
    }
    with open(os.path.abspath(a.out) + "_rig.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {a.out}.blend and {a.out}_rig.json")


main()
