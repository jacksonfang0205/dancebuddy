"""2D puppet: the character as layered 2D parts, moved by the same motion as the 3D copy (retarget.py `motion`).

Parts (blender/render_parts_2d.py, rendered from the 3D model so the puppet keeps its look):
  figure_<yaw>   body and hair, one layer per view (0, 45, 90, 135, 180; the other side is the mirror image)
  hair_vis_<yaw> the hair over the body in that view (drawn over the arms when the back is turned)
  arm_<s>, arm_<s>_fist   each arm straight, flat face to the camera

Per frame:
  * the view nearest the body's turn is picked (mirrored for turns the other way)
  * the figure squashes / bulges, bends as a soft C or S from its base, tilts on its base, hops and slides
  * each arm is its sprite laid along a circular arc from its shoulder: the arc keeps the arm's width and length
    (shorter only when the arm points toward the camera), so it bends round and never folds; arms are drawn in
    front of the body unless their shoulder is behind it in that view
  * soft shadows: the arms on the body, the character on the floor
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

VIEWS = (0, 45, 90, 135, 180)
STYLE = {"canvas": (540, 960), "body_height": 0.46, "base_y": 0.80, "horizon": 0.60,
         "wall": (196, 196, 200), "floor": (236, 236, 238), "arm_shadow": 0.30, "floor_shadow": 0.28,
         "bend_split": 0.55}


# ---------------------------------------------------------------- parts

def _fill_holes(img: np.ndarray, close_px: int = 31) -> np.ndarray:
    """Close notches and holes in a part's outline (where the 3D model had the arms) with nearby colour."""
    a = img[:, :, 3]
    m = (a > 128).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    ff = closed.copy()
    h, w = ff.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 2)
    closed = np.where(ff == 2, 0, 1).astype(np.uint8)  # everything not reachable from outside
    new = (closed > 0) & (m == 0)
    if not new.any():
        return img
    rgb = cv2.inpaint(img[:, :, :3], new.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    out = img.copy()
    out[new, :3] = rgb[new]
    out[new, 3] = 255
    return out


def _fill_black(img: np.ndarray, level: int = 30) -> np.ndarray:
    """The model's hollow inside shows black through gaps (under the hair, at the arm's cut end): paint those
    pixels over with the colours around them."""
    m = ((img[:, :, 3] > 100) & (img[:, :, :3].max(-1) < level)).astype(np.uint8)
    if not m.any():
        return img
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    out = img.copy()
    out[:, :, :3] = cv2.inpaint(img[:, :, :3], m * 255, 7, cv2.INPAINT_TELEA)
    return out


def _crop(img: np.ndarray, pts: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the visible part; return the image and the offset to subtract from points."""
    ys, xs = np.nonzero(img[:, :, 3] > 2)
    x0, y0 = max(0, xs.min() - 4), max(0, ys.min() - 4)
    x1, y1 = xs.max() + 5, ys.max() + 5
    return img[y0:y1, x0:x1].copy(), np.array([x0, y0], float)


class Parts:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.meta = json.loads((folder / "parts.json").read_text())
        self.m_per_px = float(self.meta["metres_per_px"])
        self.fig, self.hair, self.off = {}, {}, {}
        for y in VIEWS:
            f = _fill_black(_fill_holes(cv2.imread(str(folder / f"figure_{y}.png"), cv2.IMREAD_UNCHANGED)))
            self.fig[y] = f
            h = cv2.imread(str(folder / f"hair_vis_{y}.png"), cv2.IMREAD_UNCHANGED)
            body = cv2.imread(str(folder / f"body_{y}.png"), cv2.IMREAD_UNCHANGED)
            near = cv2.dilate((body[:, :, 3] > 128).astype(np.uint8), np.ones((15, 15), np.uint8))
            h[:, :, 3] = h[:, :, 3] * near  # only the hair lying over the body
            self.hair[y] = h
        self.arm = {}
        for s in "LR":
            root, tip = (np.array(self.meta["arms"][s][k], float) for k in ("root", "tip"))
            o = _fill_black(cv2.imread(str(folder / f"arm_{s}.png"), cv2.IMREAD_UNCHANGED))
            fz = _fill_black(cv2.imread(str(folder / f"arm_{s}_fist.png"), cv2.IMREAD_UNCHANGED))
            # the arm comes out of the body: its cut end at the seam fades in over the first part of its length
            yy, xx = np.mgrid[0:o.shape[0], 0:o.shape[1]]
            along = ((xx - root[0]) * (tip - root)[0] + (yy - root[1]) * (tip - root)[1]) / np.sum((tip - root) ** 2)
            fade = np.clip((along - 0.04) / 0.14, 0, 1)
            for im in (o, fz):
                im[:, :, 3] = (im[:, :, 3] * fade).astype(np.uint8)
            both = np.maximum(o[:, :, 3], fz[:, :, 3])
            ys, xs = np.nonzero(both > 2)
            x0, y0, x1, y1 = max(0, xs.min() - 4), max(0, ys.min() - 4), xs.max() + 5, ys.max() + 5
            sh = np.array([x0, y0], float)
            self.arm[s] = {"open": o[y0:y1, x0:x1].astype(np.float32), "fist": fz[y0:y1, x0:x1].astype(np.float32),
                           "root": root - sh, "tip": tip - sh}


# ---------------------------------------------------------------- geometry

def _bend_tables(H: float, lower: float, upper: float, split: float, n: int = 256):
    """Centre line of the bent body: per height y (0..H, px, up), the angle a(y) (rad, + toward screen right)
    and the centre point c(y)."""
    ys = np.linspace(0, H * 1.3, n)
    hs = split * H
    a = np.radians(lower) * np.clip(ys / hs, 0, 1) + np.radians(upper) * np.clip((ys - hs) / (H - hs), 0, 1)
    dy = np.diff(ys, prepend=0)
    cx = np.cumsum(np.sin(a) * dy)
    cy = np.cumsum(np.cos(a) * dy)
    return ys, a, cx, cy


def _forward(p: np.ndarray, st: dict) -> np.ndarray:
    """Figure point (px, base origin, y up) -> deformed point (same frame, before placing on the canvas)."""
    x, y = p[..., 0], p[..., 1]
    sy = 1.0 - st["squash"]
    sx = 1.0 / np.sqrt(sy)
    low = np.clip(1 - y / (st["H"] * 0.6), 0, 1)
    x1 = x * sx * (1 + (st["bulge"] - 1) * low)
    y1 = y * sy
    ys, a, cx, cy = st["bend"]
    ai = np.interp(y1, ys, a)
    px = np.interp(y1, ys, cx) + x1 * np.cos(ai)
    py = np.interp(y1, ys, cy) - x1 * np.sin(ai)
    t = np.radians(st["tilt"])
    return np.stack([px * np.cos(t) + py * np.sin(t), -px * np.sin(t) + py * np.cos(t)], -1)


def _inverse(q: np.ndarray, st: dict, iters: int = 6) -> np.ndarray:
    p = q.copy()
    for _ in range(iters):  # the deformation is mild around a rigid tilt: fixed-point refinement
        p = p + (q - _forward(p, st))
    return p


# ---------------------------------------------------------------- drawing

def _over(dst: np.ndarray, src: np.ndarray) -> None:
    """Alpha-composite float RGBA src (0..255, straight alpha) over float RGB dst, in place."""
    a = src[:, :, 3:4] / 255.0
    dst[:] = src[:, :, :3] * a + dst * (1 - a)


def _warp_figure(img: np.ndarray, anchor_base: np.ndarray, mirror: bool, st: dict, place) -> np.ndarray:
    """Canvas-sized RGBA of a figure-like sprite (same frame as the figure) under the body deformation."""
    W, Hc = STYLE["canvas"]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(Hc, dtype=np.float32))
    q = np.stack([(gx - place["x"]) / place["k"], (place["y"] - gy) / place["k"]], -1)
    p = _inverse(q, st)
    sxp = p[..., 0] * (-1 if mirror else 1) + anchor_base[0]
    syp = anchor_base[1] - p[..., 1]
    return cv2.remap(img, sxp.astype(np.float32), syp.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)).astype(np.float32)


def _arm_layer(spr: dict, fist: float, root_c: np.ndarray, dir_c: np.ndarray, theta: float, length_c: float,
               k: float) -> np.ndarray:
    """Canvas RGBA of an arm sprite laid along a circular arc: from root_c (canvas px), starting along the unit
    direction dir_c turned back by theta/2 (so the chord runs along dir_c), bending by theta (rad, + = toward
    the left normal), with arc length length_c (canvas px)."""
    W, Hc = STYLE["canvas"]
    img = spr["fist"] if fist > 0.5 else spr["open"]  # a hand balls up at once, like a drawn hand
    root_s, tip_s = spr["root"], spr["tip"]
    L_s = float(np.linalg.norm(tip_s - root_s))
    e = (tip_s - root_s) / L_s
    e_n = np.array([-e[1], e[0]])
    reach = int(length_c + 0.6 * L_s * k + 10)
    x0, x1 = int(max(0, root_c[0] - reach)), int(min(W, root_c[0] + reach))
    y0, y1 = int(max(0, root_c[1] - reach)), int(min(Hc, root_c[1] + reach))
    out = np.zeros((Hc, W, 4), np.float32)
    if x1 <= x0 or y1 <= y0:
        return out
    gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    d = np.stack([gx - root_c[0], gy - root_c[1]], -1)
    t0 = _rot2(dir_c, -theta / 2)                       # start tangent
    n0 = np.array([-t0[1], t0[0]])                      # left normal (canvas, y down)
    along = length_c / max(L_s, 1e-6)                   # canvas px of arc per sprite px of arm
    if abs(theta) < 1e-3:
        s = d @ t0
        v = d @ n0
    else:
        # arc: P(s) = C + R (t0 sin(s/R) - n0 cos(s/R)), C = root + R n0, R = length / theta (signed); a point
        # off the arc by v along the normal n(s) = -(P(s) - C) / R sits at C + (P(s) - C)(1 - v / R)
        R = length_c / theta
        r = d - n0 * R
        sg = np.sign(R)
        phi = np.arctan2(sg * (r @ t0), sg * (r @ -n0))
        s = R * phi
        v = R - sg * np.linalg.norm(r, axis=-1)
    # sprite coordinates: along the arm by arc length, across by the true width
    sp = root_s[None, None, :] + (s[..., None] / along) * e + (v[..., None] / k) * e_n
    patch = cv2.remap(img, sp[..., 0].astype(np.float32), sp[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out[y0:y1, x0:x1] = patch
    return out


def _rot2(v: np.ndarray, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def render(motion: dict, parts: Parts, out_dir: str | Path, step: int = 1) -> int:
    """Render every step-th frame of a motion (retarget.py) into out_dir/frame_0001.png ..."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, Hc = STYLE["canvas"]
    m_px = parts.m_per_px
    v0 = parts.meta["views"]["0"]
    H_fig = v0["base"][1] - v0["top"][1]                       # body height in sprite px
    k = STYLE["body_height"] * Hc / H_fig                       # sprite px -> canvas px
    bg = np.zeros((Hc, W, 3), np.float32)
    hz = int(STYLE["horizon"] * Hc)
    bg[:hz] = STYLE["wall"]
    bg[hz:] = STYLE["floor"]
    frames = range(0, int(motion["n"]), step)
    for idx, i in enumerate(frames):
        turn = float(motion["turn"][i])
        t_wrapped = (turn + 180) % 360 - 180
        yaw = min(VIEWS, key=lambda v: abs(v - abs(t_wrapped)))
        mirror = t_wrapped < 0
        view = parts.meta["views"][str(yaw)]
        base_s = np.array(view["base"], float)
        st = {"H": H_fig, "squash": float(motion["squash"][i]), "bulge": float(motion["bulge"][i]),
              "tilt": float(motion["tilt"][i]),
              "bend": _bend_tables(H_fig, float(motion["lower_bend"][i]), float(motion["upper_bend"][i]),
                                   STYLE["bend_split"])}
        place = {"k": k, "x": W / 2 + float(motion["shift_m"][i]) / m_px * k,
                 "y": STYLE["base_y"] * Hc - float(motion["lift_m"][i]) / m_px * k}
        canvas = bg.copy()
        # floor shadow
        sh = np.zeros((Hc, W), np.float32)
        lift = float(motion["lift_m"][i]) / m_px * k
        cv2.ellipse(sh, (int(place["x"]), int(STYLE["base_y"] * Hc)), (int(0.32 * H_fig * k / (1 + lift / 80)),
                    int(0.05 * H_fig * k)), 0, 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 9) * STYLE["floor_shadow"] / (1 + lift / 60)
        canvas *= (1 - sh[..., None])
        fig = _warp_figure(parts.fig[yaw], base_s, mirror, st, place)
        hair = _warp_figure(parts.hair[yaw], base_s, mirror, st, place) if yaw >= 135 else None

        def to_canvas(p_fig_up: np.ndarray) -> np.ndarray:
            q = _forward(p_fig_up[None], st)[0]
            return np.array([place["x"] + q[0] * k, place["y"] - q[1] * k])

        # arms: shoulder on the deformed figure, plus the shoulder's own slide; direction from the 3D arc, seen
        # through the body turn, then tilted with the body
        front, back = [], []
        rz = Rot.from_euler("z", turn, degrees=True)
        for s in "LR":
            arc = motion[f"arm_{s}"][i]
            src = ("R" if s == "L" else "L") if mirror else s
            sp = np.array(view[f"shoulder_{src}"], float)
            sp_up = np.array([(sp[0] - base_s[0]) * (-1 if mirror else 1), base_s[1] - sp[1]])
            shift3 = rz.apply(arc["head_shift"])
            sp_up = sp_up + np.array([shift3[0], shift3[2]]) / m_px
            root_c = to_canvas(sp_up)
            ch3 = rz.apply(arc["chord"])
            b3 = rz.apply(arc["bulge"])
            c2 = np.array([ch3[0], -ch3[2]])                     # canvas: y down
            fore = float(np.linalg.norm(c2))
            if fore < 1e-4:
                continue
            c2 = c2 / fore
            # the arc bulges toward b: seen on screen it turns away from that side, by as much of the bend as
            # lies across the line of sight
            b2 = np.array([b3[0], -b3[2]])
            theta = -float(arc["theta"]) * float(c2[0] * b2[1] - c2[1] * b2[0])
            # body tilt and bend at the shoulder turn the arm with the body
            ys, a, _, _ = st["bend"]
            ang = np.radians(st["tilt"]) + float(np.interp(sp_up[1] * (1 - st["squash"]), ys, a))
            c2 = _rot2(c2, ang)
            th3 = max(float(arc["theta"]), 1e-6)
            chord_len = arc["length"] * np.sin(th3 / 2) / (th3 / 2) * fore / m_px * k
            arc_len = chord_len * ((abs(theta) / 2) / np.sin(abs(theta) / 2) if abs(theta) > 1e-3 else 1.0)
            layer = _arm_layer(parts.arm[s], float(motion[f"fist_{s}"][i]), root_c, c2, theta, arc_len, k)
            depth = float(view[f"shoulder_{src}_depth"])
            (back if depth > 0.08 else front).append(layer)
        for layer in back:
            _over(canvas, layer)
        # the arms' soft shadow on the body
        if front:
            am = np.maximum.reduce([layer[:, :, 3] for layer in front]) / 255.0
            shd = cv2.GaussianBlur(np.roll(np.roll(am, 6, 0), 4, 1), (0, 0), 7) * STYLE["arm_shadow"]
            fig[:, :, :3] *= (1 - shd[..., None])
        _over(canvas, fig)
        if hair is not None:  # back turned: the arms go under the hair
            for layer in front:
                _over(canvas, layer)
            _over(canvas, hair)
        else:
            for layer in front:
                _over(canvas, layer)
        cv2.imwrite(str(out_dir / f"frame_{idx + 1:04d}.png"), np.clip(canvas, 0, 255).astype(np.uint8))
    return len(frames)
