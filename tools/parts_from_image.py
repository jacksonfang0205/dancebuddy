"""Layered 2D parts for the puppet (dancebuddy/render2d.py) from the character's own picture (front view).

The picture is the cleanest look of the character. Its parts are told apart by colour (arms, hair, body), the
arms are cut out and straightened along their centre line, and what they covered is filled in: the body is a
convex egg, so its outline there is the convex hull of the skin; the hair behind the arm continues the hair
around it. Writes figure_0.png, arm_L.png, arm_R.png (and the same as *_fist.png until a fist is drawn) and
parts.json, in the same layout as blender/render_parts_2d.py.

  python tools/parts_from_image.py assets/plush_ref/front.png assets/plush_2d_img --rig assets/plush_soft_rig.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def masks(im: np.ndarray, arm_hue=(95, 128), alpha: np.ndarray | None = None,
          arm_masks: dict | None = None) -> dict:
    """Parts by colour. A render brings its own outline (alpha) and where each arm shows (arm_masks)."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    fg = ((S > 0.15) | (V < 0.5)).astype(np.uint8)  # the background is a light, colourless grey
    if alpha is not None:
        fg = (alpha > 128).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    fg = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ff = fg.copy()  # holes inside the character (the whites of the eyes are as pale as the background)
    cv2.floodFill(ff, np.zeros((fg.shape[0] + 2, fg.shape[1] + 2), np.uint8), (0, 0), 2)
    fg = ff != 2
    blue = (H >= arm_hue[0]) & (H <= arm_hue[1]) & (S > 0.25) & fg
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue.astype(np.uint8))
    w = im.shape[1]
    arms = {}
    if arm_masks is not None:
        arms = {k: v & fg for k, v in arm_masks.items() if v.any()}
        n = 0
    for i in range(1, n):  # the arms: big blue pieces at the sides (the eyes are blue too)
        cx = st[i, 0] + st[i, 2] / 2
        if st[i, 4] > 0.002 * im.shape[0] * w and abs(cx - w / 2) > 0.2 * w:
            side = "L" if cx > w / 2 else "R"  # the character's left is the picture's right
            arms[side] = arms.get(side, np.zeros_like(blue)) | (lab == i)
    for side in arms:  # the arm with its pale fuzzy edge: bluish points next to it
        near = cv2.dilate(arms[side].astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        arms[side] |= near & fg & (H >= arm_hue[0] - 10) & (H <= arm_hue[1] + 10) & (S > 0.08)
    any_arm = arms.get("L", np.zeros_like(blue)) | arms.get("R", np.zeros_like(blue))
    hair = fg & ~any_arm & (H <= 11) & (S > 0.55)
    skin = fg & ~any_arm & ~hair
    return {"fg": fg, "arms": arms, "hair": hair, "skin": skin}


def _detail(im: np.ndarray, source: np.ndarray, shape) -> np.ndarray:
    """Fine texture (fuzz, curls) of a surface: its biggest clean square, high-pass, tiled to the image size."""
    d = cv2.distanceTransform(source.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    r = max(8, int(d[y, x] * 0.7))
    patch = im[y - r:y + r, x - r:x + r].astype(np.float32)
    hp = patch - cv2.GaussianBlur(patch, (0, 0), 5)
    reps = (shape[0] // hp.shape[0] + 1, shape[1] // hp.shape[1] + 1, 1)
    return np.tile(hp, reps)[:shape[0], :shape[1]]


def fill_behind_arms(im: np.ndarray, m: dict) -> tuple[np.ndarray, np.ndarray]:
    """The picture without its arms: RGB and alpha of body and hair, with what the arms covered filled in."""
    any_arm = np.zeros_like(m["fg"])
    for a in m["arms"].values():
        any_arm |= a
    cover = cv2.dilate(any_arm.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0  # the arm and its fuzzy edge
    # the body is a convex egg: its outline behind the arms is the convex hull of the skin
    n, lab, st, _ = cv2.connectedComponentsWithStats(m["skin"].astype(np.uint8))
    skin = lab == 1 + np.argmax(st[1:, 4])
    hull = np.zeros(skin.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.argwhere(skin)[:, ::-1].astype(np.int32)), 1)
    hull = hull > 0
    # the hair behind the arm: the hair around it closed over the gap the arm leaves
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    hair_full = cv2.morphologyEx((m["hair"] | hull).astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    keep = m["fg"] & ~cover
    alpha = keep | (cover & (hull | hair_full))
    out = im.copy()
    body_fill = cover & hull
    hair_fill = cover & hair_full & ~hull
    # colour from the surface around (smooth), fuzz from a real piece of that surface (its fine detail)
    for region, source in ((body_fill, skin & keep), (hair_fill, m["hair"] & keep)):
        if not region.any():
            continue
        # everything the arm covered is unknown (also the part that stays empty), so no blue leaks in
        hidden = (cover | region | ~(source | (m["fg"] & ~cover))).astype(np.uint8) * 255
        low = cv2.inpaint(out, hidden, 15, cv2.INPAINT_TELEA).astype(np.float32)
        low = cv2.GaussianBlur(low, (0, 0), 5)
        out[region] = np.clip(low[region] + _detail(im, source, out.shape)[region], 0, 255).astype(np.uint8)
    return out, alpha


def straighten(im: np.ndarray, arm: np.ndarray, body_centre: np.ndarray, pointing: int) -> tuple[np.ndarray, dict]:
    """Cut an arm out and lay it straight along x (pointing +1 right / -1 left), root at its body end.
    Returns RGBA and the root and tip in it."""
    soft = cv2.GaussianBlur(cv2.dilate(arm.astype(np.uint8) * 255, np.ones((5, 5), np.uint8)), (0, 0), 1.2)
    pts = np.argwhere(arm)[:, ::-1].astype(float)
    mean = pts.mean(0)
    _, vec = np.linalg.eigh(np.cov((pts - mean).T))
    ax = vec[:, 1]
    t = (pts - mean) @ ax
    if np.linalg.norm(mean + ax * t.min() - body_centre) > np.linalg.norm(mean + ax * t.max() - body_centre):
        ax, t = -ax, -t  # run from the body end (root) to the hand
    nrm = np.array([-ax[1], ax[0]])
    bins = np.linspace(t.min(), t.max(), 40)
    centre = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        sel = (t >= b0) & (t < b1)
        if sel.sum() > 5:
            centre.append([(b0 + b1) / 2, float(np.median((pts[sel] - mean) @ nrm))])
    centre = np.array(centre)
    # a smooth centre line: offset across the axis as a quadratic of the position along it
    co = np.polyfit(centre[:, 0], centre[:, 1], 2)
    width = float(np.percentile(np.abs((pts - mean) @ nrm - np.polyval(co, t)), 99.5)) * 1.25
    ts = np.linspace(t.min() - 4, t.max() + 4, int(t.max() - t.min() + 9))
    line = mean + np.outer(ts, ax) + np.outer(np.polyval(co, ts), nrm)
    tan = np.gradient(line, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nl = np.stack([-tan[:, 1], tan[:, 0]], 1)
    L = len(ts)
    Wd = int(np.ceil(width)) + 6
    vs = np.arange(-Wd, Wd + 1, dtype=float)
    mx = line[None, :, 0] + vs[:, None] * nl[None, :, 0]
    my = line[None, :, 1] + vs[:, None] * nl[None, :, 1]
    rgba = np.dstack([im, soft])
    out = cv2.remap(rgba, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if pointing < 0:  # root on the right, the hand to the left; keep the same side of the arm on top
        out = out[::-1, ::-1]
    root = [4.0 if pointing > 0 else L - 5.0, float(Wd)]
    tip = [L - 5.0 if pointing > 0 else 4.0, float(Wd)]
    return out, {"root": root, "tip": tip, "root_in_picture": line[4].tolist()}


def fill_from_render(rgb: np.ndarray, al: np.ndarray, rest: np.ndarray, m: dict):
    """Remove the arms from a render and fill what they covered. The render of the character without its arms
    shows what is behind them, except where the model has holes (where the arms were sewn on, it is open: the
    dark inside shows); those points become skin or hair, whichever is nearer among the points that show."""
    rest_alpha = rest[:, :, 3]
    any_arm = np.zeros(al.shape, bool)
    for x in m["arms"].values():
        any_arm |= x
    cover = cv2.dilate(any_arm.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    sil = ((rest_alpha > 128) | ((al > 128) & ~cover)).astype(np.uint8)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    ff = sil.copy()
    cv2.floodFill(ff, np.zeros((sil.shape[0] + 2, sil.shape[1] + 2), np.uint8), (0, 0), 2)
    sil = ff != 2
    region = cover & sil
    # behind the arms, as rendered without them (not where the model is open and its dark inside shows)
    seen = region & (rest_alpha > 200) & (rest[:, :, :3].max(-1) > 60)
    seen = cv2.erode(seen.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    rgb = rgb.copy()
    rgb[seen] = rest[:, :, :3][seen]
    keep = ((al > 128) & ~cover) | seen
    region = region & ~seen
    skin_src, hair_src = m["skin"] & keep, m["hair"] & keep
    d_skin = cv2.distanceTransform((~skin_src).astype(np.uint8), cv2.DIST_L2, 5)
    d_hair = cv2.distanceTransform((~hair_src).astype(np.uint8), cv2.DIST_L2, 5)
    out = rgb.copy()
    if region.any():
        # work in a box around what the arms covered (inpainting the whole picture is slow)
        ys, xs = np.nonzero(region)
        y0, y1 = max(0, ys.min() - 80), min(rgb.shape[0], ys.max() + 81)
        x0, x1 = max(0, xs.min() - 80), min(rgb.shape[1], xs.max() + 81)
        box = (slice(y0, y1), slice(x0, x1))
        sub, k_, cov_ = rgb[box], keep[box], cover[box]
        for part, src in ((region & (d_skin <= d_hair), skin_src), (region & (d_hair < d_skin), hair_src)):
            part, src_b = part[box], src[box]
            if part.any() and src_b.any():
                # colour only from this kind of surface: everything else is hidden from the inpainting
                unknown = (cov_ | ~src_b).astype(np.uint8) * 255
                low = cv2.GaussianBlur(cv2.inpaint(sub, unknown, 9, cv2.INPAINT_TELEA).astype(np.float32), (0, 0), 5)
                det = _detail(rgb, src, rgb.shape)[box]
                o = out[box]
                o[part] = np.clip(low[part] + det[part], 0, 255).astype(np.uint8)
    a8 = np.where(cover, (region | seen).astype(np.uint8) * 255, al).astype(np.uint8)
    return out, a8


def _resize(img: np.ndarray, f: float) -> np.ndarray:
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def side_views(views_dir: Path, out: Path, meta: dict, pic_height_px: float) -> None:
    """The other sides from renders of the whole 3D model (blender/render_views_2d.py), cut the same way and
    scaled to the picture."""
    vmeta = json.loads((views_dir / "views.json").read_text())
    full0 = cv2.imread(str(views_dir / "full_0.png"), cv2.IMREAD_UNCHANGED)
    v0 = vmeta["views"]["0"]
    f = pic_height_px / (v0["base"][1] - float(np.nonzero(full0[:, :, 3] > 128)[0].min()))
    for yaw, v in vmeta["views"].items():
        if yaw == "0":
            continue
        full = cv2.imread(str(views_dir / f"full_{yaw}.png"), cv2.IMREAD_UNCHANGED)
        arm_m = {s: cv2.imread(str(views_dir / f"armvis_{s}_{yaw}.png"), cv2.IMREAD_UNCHANGED)[:, :, 3] > 100
                 for s in "LR"}
        rgb, al = full[:, :, :3], full[:, :, 3]
        m = masks(rgb, alpha=al, arm_masks=arm_m)
        rest = cv2.imread(str(views_dir / f"rest_{yaw}.png"), cv2.IMREAD_UNCHANGED)
        fig, a8 = fill_from_render(rgb, al, rest, m)
        figure = _resize(np.dstack([fig, a8]), f)
        cv2.imwrite(str(out / f"figure_{yaw}.png"), figure)
        if int(yaw) >= 135:  # back turned: the hair is drawn over the arms
            hair = _resize(np.dstack([fig, np.where(m["hair"], a8, 0).astype(np.uint8)]), f)
            cv2.imwrite(str(out / f"hair_vis_{yaw}.png"), hair)
        meta["views"][yaw] = {k: ([round(c * f, 1) for c in val] if isinstance(val, list) else val)
                              for k, val in v.items() if k != "px_per_m"}


def fist_sprites(out: Path, meta: dict, fist_dir: Path) -> None:
    """Fists from the 3D model's straight arm, scaled and placed onto the picture's arm (root to tip)."""
    fmeta = json.loads((fist_dir / "parts.json").read_text())
    for s in "LR":
        open_ = cv2.imread(str(out / f"arm_{s}.png"), cv2.IMREAD_UNCHANGED)
        fz = cv2.imread(str(fist_dir / f"arm_{s}_fist.png"), cv2.IMREAD_UNCHANGED)
        r0, t0 = (np.array(fmeta["arms"][s][k], float) for k in ("root", "tip"))
        r1, t1 = (np.array(meta["arms"][s][k], float) for k in ("root", "tip"))
        k = np.linalg.norm(t1 - r1) / np.linalg.norm(t0 - r0)
        A = np.array([[k, 0, r1[0] - k * r0[0]], [0, k, r1[1] - k * r0[1]]], np.float32)
        pad = 60
        big = np.zeros((open_.shape[0] + 2 * pad, open_.shape[1], 4), np.uint8)
        big[pad:pad + open_.shape[0]] = open_
        A[1, 2] += pad
        dark = fz[:, :, :3].max(-1) < 90  # bits of the model's dark inside and shadowed flaps at the cut end
        fz[dark, 3] = 0
        solid = cv2.morphologyEx((fz[:, :, 3] > 100).astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
        fz[:, :, 3] = np.minimum(fz[:, :, 3], cv2.dilate(solid, np.ones((3, 3), np.uint8)) * 255)  # no thin flaps
        warped = cv2.warpAffine(fz, A, (big.shape[1], big.shape[0]), flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        cv2.imwrite(str(out / f"arm_{s}.png"), big)
        cv2.imwrite(str(out / f"arm_{s}_fist.png"), warped)
        meta["arms"][s] = {"root": [r1[0], r1[1] + pad], "tip": [t1[0], t1[1] + pad]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("picture")
    p.add_argument("out")
    p.add_argument("--rig", required=True, help="the character's rig json (its height, for scale)")
    p.add_argument("--views", help="renders of the whole 3D model from other sides (blender/render_views_2d.py)")
    p.add_argument("--fists", help="parts rendered from the 3D model with fists (blender/render_parts_2d.py)")
    a = p.parse_args()
    im = cv2.imread(a.picture)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = masks(im)
    fig, alpha = fill_behind_arms(im, m)
    a8 = cv2.GaussianBlur(alpha.astype(np.uint8) * 255, (0, 0), 0.8)
    cv2.imwrite(str(out / "figure_0.png"), np.dstack([fig, a8]))
    ys, xs = np.nonzero(m["skin"])
    base = [float(xs[ys > np.percentile(ys, 99)].mean()), float(ys.max())]
    top_skin = float(ys.min())
    top_all = float(np.nonzero(m["fg"])[0].min())
    height_m = json.loads(Path(a.rig).read_text())["height_m"]
    meta = {"px": im.shape[1], "metres_per_px": height_m / (base[1] - top_all), "height_m": height_m,
            "source": str(a.picture), "views": {"0": {"base": base, "top": [base[0], top_skin]}}, "arms": {}}
    centre = np.array([base[0], (base[1] + top_skin) / 2])
    for side, pointing in (("L", 1), ("R", -1)):
        spr, info = straighten(im, m["arms"][side], centre, pointing)
        cv2.imwrite(str(out / f"arm_{side}.png"), spr)
        cv2.imwrite(str(out / f"arm_{side}_fist.png"), spr)  # no fist in the picture yet
        meta["arms"][side] = {"root": info["root"], "tip": info["tip"]}
        meta["views"]["0"][f"shoulder_{side}"] = info["root_in_picture"]
        meta["views"]["0"][f"shoulder_{side}_depth"] = -0.3
    if a.fists:
        fist_sprites(out, meta, Path(a.fists))
    if a.views:
        side_views(Path(a.views), out, meta, base[1] - top_all)
    (out / "parts.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, "arm lengths px", {s: round(meta["arms"][s]["tip"][0] - meta["arms"][s]["root"][0]) for s in "LR"})


if __name__ == "__main__":
    main()
