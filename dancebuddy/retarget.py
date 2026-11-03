"""Copy a dancer onto a character, following the character's motion spec.

What the body can do is copied frame by frame; what it cannot is translated (assets/<name>_motion_spec.json):
  * arms   : the dancer's upper-arm direction drives the first arm bone, the forearm direction bends the soft arm
             (bones 2 and 3); arms are kept in front of the hair and the body
  * body   : the dancer's torso turn / side tilt / forward lean drive the whole body around its base, scaled down
  * legs   : no legs -> hip bounce becomes lift (up) and squash (down); a kick tilts the body away from the leg
  * hair   : never driven directly; it keeps the body's orientation from a moment ago (lag), i.e. it trails

Output is a curves.json with quaternion channels (`<bone>.quat`, [w, x, y, z] in the bone's rest frame) plus
`root.lift` and `root.squash`, rendered by blender/render_curves.py.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .beats import smooth
from .perception import L_EL, L_HIP, L_SH, L_WR, R_EL, R_HIP, R_SH, R_WR

# amplitudes from the motion spec: body follows the dancer, scaled down and within the character's range
BODY = {"turn_gain": 0.7, "turn_max": 35.0, "side_gain": 0.7, "side_max": 14.0, "lean_gain": 0.4, "lean_max": 10.0,
        "kick_tilt_deg_per_torso": 20.0, "kick_tilt_max": 10.0,
        "lift_per_torso": 0.25, "lift_max": 0.12, "squash_per_torso": 0.2, "squash_max": 0.06}
HAIR = {"lag_s": 0.24, "follow": 0.6, "max_deg": 12.0, "share": (0.2, 0.5, 0.3)}


def _norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def to_blender(p: np.ndarray) -> np.ndarray:
    """MediaPipe world (x screen-right, y down, z negative toward the camera) -> Blender (x right, y away, z up)."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def torso_frames(w: np.ndarray) -> np.ndarray:
    """(N,3,3) torso orientation; columns = dancer's left, back, up (the character's rest axes X, Y, Z)."""
    sh, hip = (w[:, L_SH] + w[:, R_SH]) / 2, (w[:, L_HIP] + w[:, R_HIP]) / 2
    z = _norm(sh - hip)
    x = w[:, L_SH] - w[:, R_SH] + 0.5 * (w[:, L_HIP] - w[:, R_HIP])
    x = _norm(x - (x * z).sum(1, keepdims=True) * z)
    y = np.cross(z, x)
    return np.stack([x, y, z], -1)


def _align(target: np.ndarray, current: np.ndarray) -> Rot:
    """Smallest rotation taking direction `current` onto `target`."""
    a, b = _norm(current), _norm(target)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-8:
        return Rot.identity() if c > 0 else Rot.from_rotvec(np.pi * _norm(np.cross(a, [0.0, 0.0, 1.0])))
    return Rot.from_rotvec(axis / s * np.arctan2(s, c))


def _local(M: np.ndarray, parent_c: Rot, c: Rot) -> list[float]:
    """Pose rotation of a bone with rest matrix M whose parent chain is rotated by parent_c (armature space),
    so that the bone ends up rotated by c (armature space): L = M^T parent_c^T c M."""
    m = Rot.from_matrix(M)
    q = (m.inv() * parent_c.inv() * c * m).as_quat()  # x, y, z, w
    return [round(float(q[3]), 5), round(float(q[0]), 5), round(float(q[1]), 5), round(float(q[2]), 5)]


def keep_clear(u: np.ndarray, sign: int) -> np.ndarray:
    """Keep an arm direction (character rest axes: x left, y back, z up) in front of the hair and the body."""
    u = u.copy()
    u[1] = min(u[1], 0.25)                       # never far behind the body
    if u[2] > 0:
        u[1] = min(u[1], -0.8 * u[2])            # raised arms pass in front of the hair
        u[0] += sign * 0.35 * u[2]               # and open outward so they read beside the head
    if sign * u[0] < 0.2:
        u[1] = min(u[1], -0.5)                   # arms crossing the body go in front of it
    return _norm(u)


def retarget(kp: dict, rig: dict, signals: dict | None = None) -> dict:
    fps = float(kp["fps"])
    w = to_blender(kp["world"])
    w = np.stack([np.stack([smooth(w[:, j, k], 1.5) for k in range(3)], -1) for j in range(w.shape[1])], 1)
    n = len(w)
    T = torso_frames(w)
    bones = {b["name"]: b for b in rig["bones"]}
    M = {k: np.array(b["matrix"], float) for k, b in bones.items()}
    rest_dir = {k: M[k][:, 1] for k in M}  # bone axis = local Y

    if signals is None:
        from .signals import compute_signals
        signals = compute_signals(kp)
    bounce = np.asarray(signals["hip_bounce"], float)
    kick = np.asarray(signals["foot_R_lift"], float) - np.asarray(signals["foot_L_lift"], float)

    # ---- body: turn, side tilt, forward lean of the dancer's torso, scaled to the character
    X, Z = T[:, :, 0], T[:, :, 2]
    turn = np.degrees(np.arctan2(X[:, 1], X[:, 0]))          # + = dancer's left side turns away from the camera
    turn = smooth(np.degrees(np.unwrap(np.radians(turn))), 2.0)
    side, lean = [], []
    for i in range(n):
        z = Rot.from_euler("z", -turn[i], degrees=True).apply(Z[i])
        side.append(np.degrees(np.arctan2(z[0], z[2])))       # + = tilt toward the character's left (+X)
        lean.append(np.degrees(np.arctan2(-z[1], z[2])))      # + = forward (toward the camera, -Y)
    side = smooth(np.array(side), 2.0) * BODY["side_gain"]
    side += np.clip(kick * BODY["kick_tilt_deg_per_torso"], -BODY["kick_tilt_max"], BODY["kick_tilt_max"])
    side = np.clip(side, -BODY["side_max"], BODY["side_max"])
    lean = np.clip(smooth(np.array(lean), 2.0) * BODY["lean_gain"], -BODY["lean_max"], BODY["lean_max"])
    turn = np.clip(turn * BODY["turn_gain"], -BODY["turn_max"], BODY["turn_max"])
    body = [Rot.from_euler("z", turn[i], degrees=True) * Rot.from_euler("y", side[i], degrees=True)
            * Rot.from_euler("x", lean[i], degrees=True) for i in range(n)]

    lift = np.clip(bounce * BODY["lift_per_torso"], 0, BODY["lift_max"])
    squash = np.clip(-bounce * BODY["squash_per_torso"], 0, BODY["squash_max"])

    ch: dict[str, list] = {k: [] for k in ["root.quat", "root.lift", "root.squash"]}
    ident = Rot.identity()
    for i in range(n):
        ch["root.quat"].append(_local(M["root"], ident, body[i]))
    ch["root.lift"] = [round(float(x), 4) for x in lift]
    ch["root.squash"] = [round(float(x), 4) for x in squash]

    # ---- arms: copied in the torso frame (the body bones already carry the body rotation)
    for side_name, sign, (sh, el, wr) in (("L", 1, (L_SH, L_EL, L_WR)), ("R", -1, (R_SH, R_EL, R_WR))):
        names = [f"arm_{side_name}_{k}" for k in (1, 2, 3)]
        if not all(nm in M for nm in names):
            continue
        for nm in names:
            ch[f"{nm}.quat"] = []
        for i in range(n):
            Ti = T[i]
            u = keep_clear(_norm(Ti.T @ (w[i, el] - w[i, sh])), sign)
            f = _norm(Ti.T @ (w[i, wr] - w[i, el]))
            if f[1] > 0.3:
                f = _norm(f - np.array([0, f[1] - 0.3, 0]))
            c1 = _align(u, rest_dir[names[0]])
            mid = _norm(u + f)
            c2 = _align(mid, c1.apply(rest_dir[names[1]])) * c1
            c3 = _align(f, c2.apply(rest_dir[names[2]])) * c2
            ch[f"{names[0]}.quat"].append(_local(M[names[0]], ident, c1))
            ch[f"{names[1]}.quat"].append(_local(M[names[1]], c1, c2))
            ch[f"{names[2]}.quat"].append(_local(M[names[2]], c2, c3))

    # ---- hair: trails the body (keeps where the body was a moment ago)
    lag = max(1, int(round(HAIR["lag_s"] * fps)))
    hair = [nm for nm in ("hair_1", "hair_2", "hair_3") if nm in M]
    if hair:
        rots = Rot.concatenate(body)
        for nm in hair:
            ch[f"{nm}.quat"] = []
        for i in range(n):
            # how far the body has turned since a moment ago; the hair keeps a part of that (capped),
            # spread over the chain (little at the scalp so it does not tear, most in the middle)
            past = rots[max(0, i - lag)]
            delta = (past * body[i].inv()).as_rotvec() * HAIR["follow"]
            ang = np.linalg.norm(delta)
            if ang > np.radians(HAIR["max_deg"]):
                delta *= np.radians(HAIR["max_deg"]) / ang
            parent = body[i]
            for k, nm in enumerate(hair):
                target = Rot.from_rotvec(delta * HAIR["share"][k]) * parent
                ch[f"{nm}.quat"].append(_local(M[nm], parent, target))
                parent = target

    return {"fps": fps, "n_frames": n, "character": rig.get("name", "character"), "channels": ch,
            "retarget": {"body": BODY, "hair": HAIR}}
