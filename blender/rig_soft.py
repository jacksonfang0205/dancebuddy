"""Rig a textured character the way it moves, without changing its shape.

The rig follows the character's motion spec (assets/<name>_motion_spec.json), which was learned by watching it dance:
  * body      : root at the base (tilt, turn, hop, squash) -> body_lower -> body_upper (barely bends)
  * shoulders : arm_X_0, hidden in the body and not deforming; turning it slides the arm's root up the body side
                (a raised arm starts beside the face)
  * arms      : arm_X_1..3, B-bones, so each arm bends as one smooth curve (no elbow)
  * hair      : no bones of its own; it moves with the head (not animated separately)

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
    p.add_argument("--seam-band", type=float, default=0.10, help="body around an arm seam that follows the shoulder, x height")
    p.add_argument("--seam-follow", type=float, default=0.6, help="how much that body follows the shoulder (0-1)")
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
        # the shoulder turns around a point just inside the seam, so when it takes part of the arm's turn the
        # arm's root turns in place (a deep pivot would swing the root around, e.g. back into the hair)
        pivot = root_pt - np.array([sign * 0.06 * height, 0.0, 0.0])
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
        band = is_body & (dist < a.seam_band * height)
        f = a.seam_follow * (1 - dist[band] / (a.seam_band * height))
        W[band] *= (1 - f)[:, None]
        W[band, col[f"arm_{side}_0"]] += f
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)

    # which way each hand's flat face points at rest (the thinnest direction of the mitten, across the arm):
    # when driving, the arm keeps this face toward the camera so it shows its full width
    faces = {}
    for side, d in arms.items():
        m = is_arm & (np.sign(co[:, 0] - cx) == (1 if side == "L" else -1))
        pts = np.array([list(v) for v in d["pts"]])
        ax = (pts[-1] - pts[0]) / np.linalg.norm(pts[-1] - pts[0])
        rel = co[m] - pts[0]
        s_h = rel @ ax / np.linalg.norm(pts[-1] - pts[0])
        q = rel[s_h > 0.8]
        q = q - np.outer(q @ ax, ax)
        q = q - q.mean(0)
        _, vec = np.linalg.eigh(np.cov(q.T))
        n = vec[:, 1] if abs(vec[:, 0] @ ax) > 0.9 else vec[:, 0]  # thinnest, but not the arm axis itself
        faces[side] = [round(float(x), 4) for x in (n if n[1] < 0 else -n)]  # toward the camera (-Y) at rest
        print(f"arm_{side} hand face (rest): {faces[side]}")

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
                "3-bone B-bone arms; hair moves with the head; mesh unchanged, heat weights",
        "height_m": round(height, 3),
        "root_bone": "root",
        "bones": [{"name": b.name, "parent": b.parent.name if b.parent else None, "deform": b.use_deform,
                   "head": [round(c, 3) for c in b.head_local], "tail": [round(c, 3) for c in b.tail_local],
                   "length": round(b.length, 3),
                   # rest orientation in armature space (columns = the bone's local x, y, z axes)
                   "matrix": [[round(b.matrix_local[i][j], 5) for j in range(3)] for i in range(3)]}
                  for b in arm_data.bones],
        "parts": {"arms": int(is_arm.sum()), "hair": int(is_hair.sum()), "body": int(is_body.sum())},
        "hand_face": faces,
    }
    with open(os.path.abspath(a.out) + "_rig.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {a.out}.blend and {a.out}_rig.json")


main()
