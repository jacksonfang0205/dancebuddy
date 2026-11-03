"""Re-rig a textured character with soft arms, a base-pivot body and hair spring bones.

Follows the rig requirements in assets/<name>_motion_spec.json:
  * body : root (pivot at the base: tilt, turn, squash, lift) -> body_lower -> body_upper
  * arms : 3-bone chains with B-bone segments, so an arm bends smoothly along its length
  * hair : 3-bone chain down the back, to be driven as a lagged copy of the body (secondary motion)

Parts are found from the base-colour texture (arms = blue, hair = dark brown, the rest = body), so the weights
follow the character's real parts instead of height bands.

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
        if mat and mat.use_nodes:
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


def merged_neighbours(co, me, eps=1e-4):
    """Edge list, with vertices at the same position (UV seams) joined."""
    from scipy.spatial import cKDTree

    ne = len(me.edges)
    ed = np.zeros(ne * 2, int)
    me.edges.foreach_get("vertices", ed)
    pairs = np.array(sorted(cKDTree(co).query_pairs(eps)), int).reshape(-1, 2)
    return np.vstack([ed.reshape(-1, 2), pairs])


def smooth_weights(W, edges, mask, iters=10):
    """Laplacian smoothing of the weight matrix W (verts x bones), only on vertices in mask."""
    from scipy.sparse import coo_matrix, diags

    n = W.shape[0]
    A = coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                   shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    P = diags(1 / np.maximum(deg, 1)) @ A
    for _ in range(iters):
        Wn = P @ W
        W[mask] = 0.5 * W[mask] + 0.5 * Wn[mask]
    return W


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
    # Generated meshes are split along UV seams. Join vertices at the same position, so each point of the
    # surface gets one part label and one set of weights (otherwise seams crack open when a limb moves).
    # UVs live on face corners, so the texture is unaffected.
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.to_mesh(me)
    bm.free()
    me.update()
    print(f"merged seam vertices: {before} -> {len(me.vertices)}")

    co, vuv = mesh_arrays(me)
    H, S, V = vertex_hsv(body, vuv)
    height = float(co[:, 2].max())
    is_arm = (H > a.arm_hue[0]) & (H < a.arm_hue[1]) & (S > 0.3) & (np.abs(co[:, 0]) > 0.25 * np.abs(co[:, 0]).max())
    is_hair = (H < 40) & (S > 0.45) & (V < a.hair_max_value) & (co[:, 2] > 0.15 * height) & ~is_arm
    is_body = ~is_arm & ~is_hair
    low = is_body & (co[:, 2] < 0.3 * height)
    cx, cy = float(co[low, 0].mean()), float(co[low, 1].mean())
    print(f"vertices: arms {is_arm.sum()}, hair {is_hair.sum()}, body {is_body.sum()}; body centre ({cx:.3f}, {cy:.3f})")

    # ---- skeleton geometry
    z_low, z_mid = 0.2 * height, 0.55 * height
    body_top = float(co[is_body, 2].max())
    spine = [("root", Vector((cx, cy, 0)), Vector((cx, cy, z_low)), None),
             ("body_lower", Vector((cx, cy, z_low)), Vector((cx, cy, z_mid)), "root"),
             ("body_upper", Vector((cx, cy, z_mid)), Vector((cx, cy, body_top)), "body_lower")]

    arms = {}
    for side, sign in (("L", 1), ("R", -1)):
        m = is_arm & (np.sign(co[:, 0] - cx) == sign)
        p = co[m]
        inner = p[np.argsort(sign * (p[:, 0] - cx))[: max(10, len(p) // 12)]]  # innermost ~8%: where it joins the body
        shoulder = inner.mean(0)
        shoulder[0] -= sign * 0.04 * height  # joint a little inside the body surface
        tip = p[np.argmax(np.linalg.norm(p - shoulder, axis=1))]
        axis = (tip - shoulder) / np.linalg.norm(tip - shoulder)
        s = (p - shoulder) @ axis / np.linalg.norm(tip - shoulder)
        pts = [shoulder]
        for t in (1 / 3, 2 / 3, 0.97):
            sel = np.abs(s - t) < 0.06
            pts.append(p[sel].mean(0) if sel.sum() > 5 else shoulder + t * (tip - shoulder))
        arms[side] = {"mask": m, "pts": [Vector(x) for x in pts], "shoulder": shoulder, "axis": axis,
                      "length": float(np.linalg.norm(tip - shoulder))}
        print(f"arm_{side}: {m.sum()} verts, shoulder {np.round(shoulder, 3)}, tip {np.round(tip, 3)}")

    hp = co[is_hair]
    crown_z = float(hp[:, 2].max())
    bottom_z = float(np.percentile(hp[:, 2], 2))
    back = hp[(np.abs(hp[:, 0] - cx) < 0.2 * height) & (hp[:, 1] > cy)]
    hair_pts = []
    for z in np.linspace(crown_z - 0.12 * height, bottom_z, 4):
        sel = np.abs(back[:, 2] - z) < 0.06 * height
        y = float(np.percentile(back[sel, 1], 50)) if sel.sum() > 5 else cy + 0.2 * height
        hair_pts.append(Vector((cx, y, z)))

    # ---- armature
    arm_data = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", arm_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    for name, h, t, parent in spine:
        b = eb.new(name)
        b.head, b.tail = h, t
        b.roll = 0
        if parent:
            b.parent, b.use_connect = eb[parent], True
    for side, d in arms.items():
        prev = None
        for i in range(3):
            b = eb.new(f"arm_{side}_{i + 1}")
            b.head, b.tail = d["pts"][i], d["pts"][i + 1]
            b.align_roll(FORWARD)
            b.bbone_segments = 6
            b.parent = prev if prev else eb["body_lower"]
            b.use_connect = prev is not None
            if prev is not None:
                b.inherit_scale = "NONE"  # each segment stretches on its own (see the `stretch` channel)
            prev = b
    prev = None
    for i in range(3):
        b = eb.new(f"hair_{i + 1}")
        b.head, b.tail = hair_pts[i], hair_pts[i + 1]
        b.align_roll(FORWARD)
        b.bbone_segments = 4
        b.parent = prev if prev else eb["body_upper"]
        b.use_connect = prev is not None
        prev = b
    bpy.ops.object.mode_set(mode="OBJECT")
    bone_names = [b.name for b in arm_data.bones]
    idx = {n: i for i, n in enumerate(bone_names)}

    # ---- weights
    n = len(co)
    W = np.zeros((n, len(bone_names)))
    # body: by height, blended between the three spine bones
    zs = co[:, 2] / height
    Wb = chain_weights(zs, [0, z_low / height, z_mid / height, 1.0], blend=0.2)
    for j, name in enumerate(("root", "body_lower", "body_upper")):
        W[:, idx[name]] = Wb[:, j]
    # arms: a capsule field around each arm's centre line, so the weight depends only on where a point is
    # (two points that are close always move together, whatever their part label or mesh piece)
    arm_w = {}
    for side, d in arms.items():
        pts = np.array([list(v) for v in d["pts"]])
        lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        seg = np.r_[0, np.cumsum(lens)]
        best_d = np.full(n, np.inf)
        s_par = np.zeros(n)
        for i in range(3):
            ab = pts[i + 1] - pts[i]
            t = np.clip((co - pts[i]) @ ab / (ab @ ab), 0, 1)
            dist = np.linalg.norm(co - (pts[i] + t[:, None] * ab), axis=1)
            take = dist < best_d
            best_d[take], s_par[take] = dist[take], (seg[i] + t[take] * lens[i]) / seg[-1]
        r = float(np.median(best_d[d["mask"]]))
        s_raw = (co - pts[0]) @ ((pts[1] - pts[0]) / lens[0])          # distance past the shoulder joint
        ramp = np.clip(s_raw / (0.12 * seg[-1]), 0, 1)
        radial = 1 - np.clip((best_d - 1.15 * r) / (0.55 * r), 0, 1)
        w = ramp * radial
        w[d["mask"]] = np.maximum(w[d["mask"]], ramp[d["mask"]])
        w[is_hair] = 0
        arm_w[side] = (w, chain_weights(s_par, list(seg / seg[-1]), blend=0.15))
        print(f"arm_{side}: radius {r:.3f}, {int((w > 0.5).sum())} points follow it")
    total = sum(w for w, _ in arm_w.values())
    scale = np.where(total > 1, 1 / np.maximum(total, 1e-9), 1)
    for name in ("root", "body_lower", "body_upper"):
        W[:, idx[name]] *= 1 - total * scale
    for side, (w, Wa) in arm_w.items():
        for i in range(3):
            W[:, idx[f"arm_{side}_{i + 1}"]] = w * scale * Wa[:, i]
    # hair: the scalp stays on the head, the rest hangs on the hair chain (by height from the crown down)
    hz = np.array([v.z for v in hair_pts])
    s_h = (hz[0] + 0.12 * height - co[:, 2]) / (hz[0] + 0.12 * height - hz[-1])  # 0 at the crown, 1 at the bottom
    Wh = chain_weights(np.clip(s_h, 0, 1), [0, 0.12 / (hz[0] + 0.12 * height - hz[-1]) * height,
                                              *(((hz[0] + 0.12 * height) - hz[1:]) / (hz[0] + 0.12 * height - hz[-1]))],
                       blend=0.15)
    hm = is_hair
    W[hm] = 0
    W[hm, idx["body_upper"]] = Wh[hm, 0]
    for i in range(3):
        W[hm, idx[f"hair_{i + 1}"]] = Wh[hm, i + 1]
    # smooth in space (not along mesh edges: the mesh is in many pieces), then keep hair off the arms
    from scipy.spatial import cKDTree
    tree = cKDTree(co)
    nbrs = tree.query_ball_point(co, r=0.025 * height)
    for _ in range(2):
        W = np.array([W[nb].mean(0) for nb in nbrs])
    arm_cols = [idx[nm] for nm in bone_names if nm.startswith("arm_")]
    W[np.ix_(is_hair, arm_cols)] = 0
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)

    groups = {nm: body.vertex_groups.new(name=nm) for nm in bone_names}
    for j, nm in enumerate(bone_names):
        nz = np.nonzero(W[:, j] > 1e-3)[0]
        for vi in nz:
            groups[nm].add([int(vi)], float(W[vi, j]), "REPLACE")
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    mod.use_deform_preserve_volume = True
    if len(body.modifiers) > 1:
        print("note: other modifiers on Body:", [m.name for m in body.modifiers])
    body.parent = rig
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
    # frame the character with room for raised arms and hops (portrait video, like a phone dance clip)
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
        "kind": "3D model, soft rig (rig_soft.py): base-pivot body, 3-bone B-bone arms, hair spring chain",
        "height_m": round(height, 3),
        "root_bone": "root",
        "bones": [{"name": b.name, "parent": b.parent.name if b.parent else None,
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
