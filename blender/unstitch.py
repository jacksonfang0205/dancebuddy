"""Free the arms from the body, the way a plush arm is only sewn on at the shoulder (motion spec: the arm touches
the body, it does not drag the body's skin along).

Generated models fuse each arm to the body along its whole inner side, so a raised arm pulls a sheet of skin up
with it. Here the arm faces are cut loose from the body faces along that line (the rest shape does not change:
the cut points stay where they were), and a skin patch is laid under each arm along the body's own shape, in a
skin colour from the texture, so lifting the arm shows body, not the hollow inside of the model. At rest the arm
covers the patch.
"""

import bmesh
import numpy as np
from scipy.spatial import cKDTree


def unstitch_arms(me, is_arm: np.ndarray, is_hair: np.ndarray, centre: tuple, grid: int = 14):
    """Cut the arms loose and add the skin patches. Returns (orig, arm, hair) for the new vertices: the original
    vertex each one comes from (-1 for patch points) and whether it is arm / hair."""
    nv = len(me.vertices)
    cx, cy = centre
    bm = bmesh.new()
    bm.from_mesh(me)
    uv = bm.loops.layers.uv.active
    side = bm.faces.layers.int.new("arm_side")
    bm.verts.ensure_lookup_table()
    for f in bm.faces:  # a face with any arm point goes with the arm, so no blue is left on the body
        f[side] = int(any(is_arm[v.index] for v in f.verts))
    seam = [e for e in bm.edges if len(e.link_faces) == 2 and e.link_faces[0][side] != e.link_faces[1][side]]
    co0 = np.array([v.co[:] for v in bm.verts])
    bmesh.ops.split_edges(bm, edges=seam)
    # points where arm and body faces still meet only at a corner: give each side its own copy
    for v in list(bm.verts):
        kinds = {f[side] for f in v.link_faces}
        if len(kinds) > 1:
            arm_edges = [e for e in v.link_edges if all(f[side] for f in e.link_faces) and e.link_faces]
            if arm_edges:
                bmesh.utils.vert_separate(v, arm_edges)
    bm.verts.ensure_lookup_table()
    n1 = len(bm.verts)
    co1 = np.array([v.co[:] for v in bm.verts])
    orig = np.arange(n1)
    if n1 > nv:  # new copies sit exactly on their original
        orig[nv:] = cKDTree(co0).query(co1[nv:])[1]
    arm = np.zeros(n1, bool)
    for v in bm.verts:
        fs = list(v.link_faces)
        arm[v.index] = bool(fs) and sum(f[side] for f in fs) * 2 > len(fs)
    hair = is_hair[orig] & ~arm

    # skin patches
    vuv = np.zeros((n1, 2))
    for f in bm.faces:
        for lp in f.loops:
            vuv[lp.vert.index] = lp[uv].uv[:]
    plain = ~is_arm[orig] & ~is_hair[orig] & ~arm
    # texture for the patch only from skin well inside the skin: every face around it is skin, so its texture
    # coordinates are not on the edge of an arm or hair patch of the texture
    inner = plain.copy()
    for v in bm.verts:
        if inner[v.index] and (not v.link_faces or any(not plain[w.index] for f in v.link_faces for w in f.verts)):
            inner[v.index] = False
    skin = np.nonzero(plain)[0]
    inner_skin = np.nonzero(inner)[0]
    sco = co1[skin]
    sang = np.arctan2(sco[:, 1] - cy, sco[:, 0] - cx)
    srad = np.hypot(sco[:, 0] - cx, sco[:, 1] - cy)
    by_place = cKDTree(np.c_[sco[:, 2], 0.5 * sang])
    by_pos = cKDTree(sco)
    ico = co1[inner_skin]
    by_place_inner = cKDTree(np.c_[ico[:, 2], 0.5 * np.arctan2(ico[:, 1] - cy, ico[:, 0] - cx)])
    patch = []
    for sign in (1, -1):
        m = np.nonzero(arm & (np.sign(co1[:, 0] - cx) == sign))[0]
        if not len(m):
            continue
        d, _ = by_pos.query(co1[m])
        near = co1[m][d < 0.05]  # the part of the arm lying against the body
        if len(near) < 10:
            continue
        ang = np.arctan2(near[:, 1] - cy, near[:, 0] - cx)
        a0, a1 = np.percentile(ang, 1) - 0.12, np.percentile(ang, 99) + 0.12
        z0, z1 = near[:, 2].min() - 0.05, near[:, 2].max() + 0.08
        A, Z = np.linspace(a0, a1, grid), np.linspace(z0, z1, grid)
        mid_a, mid_z = (a0 + a1) / 2, (z0 + z1) / 2
        k = by_place_inner.query([mid_z, 0.5 * mid_a])[1]
        uv0 = vuv[inner_skin[k]]  # one spot of the skin texture, from the body right there
        rows = []
        for iz, z in enumerate(Z):
            row = []
            for ia, a in enumerate(A):
                r = float(np.median(srad[by_place.query([z, 0.5 * a], k=10)[1]])) * 0.96  # just under the skin
                v = bm.verts.new((cx + r * np.cos(a), cy + r * np.sin(a), z))
                row.append((v, uv0 + 0.0004 * np.array([ia - grid / 2, iz - grid / 2])))
            rows.append(row)
        for iz in range(grid - 1):
            for ia in range(grid - 1):
                q = [rows[iz][ia], rows[iz][ia + 1], rows[iz + 1][ia + 1], rows[iz + 1][ia]]
                f = bm.faces.new([x[0] for x in q])
                f.smooth = True
                for lp, (_, u) in zip(f.loops, q):
                    lp[uv].uv = u
                patch.append(f)
    bm.normal_update()
    for f in patch:  # facing out of the body
        c = f.calc_center_median()
        if f.normal.x * (c.x - cx) + f.normal.y * (c.y - cy) < 0:
            f.normal_flip()
    bm.faces.layers.int.remove(side)
    bm.to_mesh(me)
    bm.free()
    n2 = len(me.vertices)
    pad = n2 - n1
    return (np.r_[orig, -np.ones(pad, int)], np.r_[arm, np.zeros(pad, bool)], np.r_[hair, np.zeros(pad, bool)])
