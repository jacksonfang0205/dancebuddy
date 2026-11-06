"""A mitten balled into a fist, as a shape (shape key) the hand can blend into; the mesh itself is not changed.

The fingers (the part of the mitten past the knuckles) roll up toward the back of the hand around an axis across
the hand, like a sock rolled over a finger: the inside of the roll is pressed together (plush stuffing gives),
the outside keeps its length, and the fist is a little narrower than the open hand.
"""

import numpy as np


def fist_shape(rel: np.ndarray, s: np.ndarray, length: float, axis: np.ndarray, face: np.ndarray,
               knuckle: float = 0.82, turn_deg: float = 160.0, press: float = 0.55, narrow: float = 0.9) -> np.ndarray:
    """New positions (relative to the arm's start) of hand points.

    rel   points relative to the arm's start; s their position along the arm (0 seam .. 1 tip of the mitten)
    axis  unit vector along the arm; face  unit vector of the hand's flat face (the fingers roll away from it)
    """
    c = -np.asarray(face, float)                 # the side the fingers roll toward
    c = c - (c @ axis) * axis
    c /= np.linalg.norm(c)
    w = np.cross(axis, c)
    x, y, z = rel @ w, rel @ c, rel @ axis
    out = rel.copy()
    d = (s - knuckle) * length                   # distance past the knuckles
    fing = d > 0
    if not fing.any():
        return out
    reach = float(d[fing].max())
    y_in = np.where(y > 0, y * press, y)         # the inside of the roll is pressed together
    inner = float(np.percentile(y_in[fing], 98))
    r = max(reach / np.radians(turn_deg), 1.1 * inner)   # radius of the roll's middle line
    phi = d[fing] / r
    z0 = knuckle * length + 0.0 * z[fing]
    rad = r - y_in[fing]
    new_z = z0 + rad * np.sin(phi)
    new_y = r - rad * np.cos(phi)
    # the palm narrows a little toward the fist, and the rolled part with it
    k = np.clip((s - (knuckle - 0.15)) / 0.15, 0, 1)
    sx = 1 - (1 - narrow) * k
    out = np.outer(x * sx, w) + np.outer(y, c) + np.outer(z, axis)
    out[fing] = np.outer(x[fing] * sx[fing], w) + np.outer(new_y, c) + np.outer(new_z, axis)
    return out


def smooth_shape(rest: np.ndarray, pos: np.ndarray, movable: np.ndarray, radius: float, iters: int = 3) -> np.ndarray:
    """Round off the rolled fingers so the fist reads as one soft ball instead of folded finger tips: each
    movable point moves toward the mean of the points near it. Neighbours are found in the open hand (by
    position, not by mesh edges, so points the mesh duplicates along its seams stay together)."""
    from scipy.spatial import cKDTree
    pos = pos.copy()
    nb = cKDTree(rest).query_ball_point(rest[movable], radius)
    idx = np.nonzero(movable)[0]
    for _ in range(iters):
        pos[idx] = np.array([pos[n].mean(0) for n in nb])
    return pos
