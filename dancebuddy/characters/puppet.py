"""Character picture -> cut-out sprite + occupancy grid, for dancebuddy/blender/make_puppet.py.

Accepts a PNG with transparency, or any image on a plain background (the
background colour is estimated from the image border and keyed out).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def cut_out(image_path: str | Path, out_prefix: str | Path, grid_rows: int = 64, bg_tolerance: float = 40.0) -> dict:
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image {image_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgr = img[..., :3].astype(float)
    alpha = img[..., 3]
    if alpha.min() > 250:  # no real transparency: key out the border colour
        border = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]])
        bg = np.median(border, axis=0)
        mask = (np.linalg.norm(bgr - bg, axis=2) > bg_tolerance).astype(np.uint8)
        # keep the largest blob and fill its holes, so white eyes etc. stay part of the character
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == biggest).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
        alpha = (mask * 255).astype(np.uint8)
        img[..., 3] = alpha
    ys, xs = np.where(alpha > 127)
    if len(xs) == 0:
        raise RuntimeError("could not find the character in the picture; try a PNG with a transparent background")
    pad = 4
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    crop = img[y0:y1, x0:x1]
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    sprite = Path(str(out) + "_sprite.png")
    cv2.imwrite(str(sprite), crop)
    h, w = crop.shape[:2]
    cols = max(4, int(round(grid_rows * w / h)))
    small = cv2.resize((crop[..., 3] > 127).astype(np.float32), (cols, grid_rows), interpolation=cv2.INTER_AREA)
    grid = (small > 0.3).astype(int).tolist()  # row 0 = top of the picture
    info = {"sprite": sprite.name, "width_px": w, "height_px": h, "rows": grid_rows, "cols": cols, "grid": grid}
    Path(str(out) + "_grid.json").write_text(json.dumps(info))
    print(f"cut out {w}x{h} px character -> {sprite}")
    return info
