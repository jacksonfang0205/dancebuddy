"""Copy a dancer onto a character, following the character's motion spec.

What the body can do is copied frame by frame; what it cannot is translated (assets/<name>_motion_spec.json):
  * arms   : the dancer's upper-arm direction drives the first arm bone, the forearm direction bends the soft arm
             (bones 2 and 3); arms are kept in front of the hair and the body
  * body   : the dancer's torso turn / side tilt / forward lean drive the whole body around its base, at full size
             and smoothly compressed near the character's limits (never cut flat, so peaks keep their timing)
  * legs   : no legs -> every jump, kick, step, squat and move across is translated (legs.py) into hops,
             squash and stretch, leans and slides of the body
  * hair   : never driven directly; it keeps the body's orientation from a moment ago (lag), i.e. it trails

Output is a curves.json with quaternion channels (`<bone>.quat`, [w, x, y, z] in the bone's rest frame) plus
`root.loc` (slide across, hop) and `root.squash`, rendered by blender/render_curves.py.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .beats import smooth
from .perception import L_EL, L_HIP, L_SH, L_WR, R_EL, R_HIP, R_SH, R_WR

# the character's range (motion spec "can_do"): the dancer is copied at full size and compressed softly near these
BODY = {"turn_max": 60.0, "side_max": 25.0, "lean_max": 20.0}


def soft_limit(x: np.ndarray, limit: float) -> np.ndarray:
    """Full size for small values, smoothly compressed toward the limit (peaks keep their timing)."""
    return limit * np.tanh(np.asarray(x, float) / limit)
# arm directions: in-plane from the image (reliable), depth from the 3D estimate (monocular depth is not)
ARM_DEPTH_GAIN = 0.4
HAIR = {"lag_s": 0.24, "follow": 0.6, "max_deg": 12.0, "share": (0.2, 0.5, 0.3)}
# style from the spec's "tends_to": arms lead and the body follows, soft arms, squash low / stretch high
STYLE = {"straight_above_deg": 120,         # raised arms straighten out (reference: arms up are straight)
         "bend_share": 0.8,                 # how much of the dancer's elbow bend the soft arm takes, as one arc
         # the shoulder is a socket that slides up the side of the body as the arm rises (reference: a raised
         # arm starts beside the face, a hanging arm beside the chest), so raised hands get above the head
         "socket_slide_deg": 40.0,          # shoulder-bone turn that slides the socket up, for a vertical arm
         "root_share": 0.0,                 # part of the arm's turn taken by the shoulder (0.4 pushed raised
                                            # arms back into the hair, so it is off)
         "socket_slide_from_deg": 80,
         "forearm_lag_s": 0.07,             # the end of the arm trails the upper arm a little
         "follow_tilt_deg": 6.0,            # the body leans toward the higher arm...
         "follow_delay_s": 0.4,             # ...about this long after the arms move
         }


def _norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def to_blender(p: np.ndarray) -> np.ndarray:
    """MediaPipe world (x screen-right, y down, z negative toward the camera) -> Blender (x right, y away, z up)."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def image_aspect(kp: dict) -> float:
    """Width / height of the video, estimated from the body (keypoints.npz has no frame size): the shoulder
    width / torso length ratio must be the same in the 3D estimate and in the image."""
    w, im = kp["world"], kp["image"]
    sw_w = np.linalg.norm(w[:, L_SH, :2] - w[:, R_SH, :2], axis=1)
    th_w = np.abs(((w[:, L_HIP] + w[:, R_HIP]) / 2 - (w[:, L_SH] + w[:, R_SH]) / 2)[:, 1])
    sw_i = np.abs(im[:, L_SH, 0] - im[:, R_SH, 0])
    th_i = np.abs(((im[:, L_HIP] + im[:, R_HIP]) / 2 - (im[:, L_SH] + im[:, R_SH]) / 2)[:, 1])
    facing = sw_i >= np.percentile(sw_i, 60)  # frames where the dancer faces the camera
    r = (sw_w / np.maximum(th_w, 1e-6) * th_i / np.maximum(sw_i, 1e-6))[facing]
    r = r[np.isfinite(r)]
    return float(np.median(r)) if len(r) else 1.0


def limb_dirs(kp: dict, w: np.ndarray, a: int, b: int, aspect: float) -> np.ndarray:
    """(N,3) direction a->b in Blender axes: x and z from the image, y (depth) from the 3D estimate, damped."""
    im = kp["image"]
    dx = (im[:, b, 0] - im[:, a, 0]) * aspect
    dz = -(im[:, b, 1] - im[:, a, 1])
    v3 = w[:, b] - w[:, a]
    scale = np.linalg.norm(np.c_[dx, dz], axis=1) / np.maximum(np.linalg.norm(v3[:, [0, 2]], axis=1), 1e-6)
    dy = v3[:, 1] * scale * ARM_DEPTH_GAIN
    d = np.c_[dx, dy, dz]
    d = np.stack([smooth(d[:, k], 1.5) for k in range(3)], -1)
    return _norm(d)


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
        u[1] = min(u[1], -0.35 * u[2])           # raised arms pass just in front of the hair (not foreshortened)
        u[0] += sign * 0.15 * u[2]               # and open slightly outward so they read beside the head
    if sign * u[0] < 0.2:
        u[1] = min(u[1], -0.5)                   # arms crossing the body go in front of it
    return _norm(u)


def raise_deg(u: np.ndarray) -> float:
    """0 = hanging down, 90 = horizontal, 180 = straight up (direction in character rest axes)."""
    return float(np.degrees(np.arccos(np.clip(-u[2], -1, 1))))


def socket_turn(u: np.ndarray, sign: int) -> Rot:
    """Turn of the hidden shoulder bone (armature space) that slides the arm's root up the side of the body as
    the arm rises: it swings the root upward around a pivot inside the body."""
    a = raise_deg(_norm(u))
    k = float(np.clip((a - STYLE["socket_slide_from_deg"]) / (175 - STYLE["socket_slide_from_deg"]), 0, 1))
    k = k * k * (3 - 2 * k)
    return Rot.from_rotvec([0.0, -sign * np.radians(STYLE["socket_slide_deg"]) * k, 0.0])


def socket_rot(M: dict, names: list[str], u: np.ndarray, f: np.ndarray, sign: int) -> Rot:
    """The hidden shoulder bone: slides the seam up the body as the arm rises, and turns with the arm a little,
    so a big arm turn is not all taken at the arm's root (which would twist it thin)."""
    u_c = keep_clear(_norm(u), sign)
    target = keep_clear(_norm(u_c + _norm(f)), sign)
    rest = _norm(np.array(M[f"{names[-1]}__tail"]) - np.array(M[f"{names[0]}__head"]))
    share = Rot.from_rotvec(_align(target, rest).as_rotvec() * STYLE["root_share"])
    return share * socket_turn(u_c, sign)


def arm_pose(M: dict, names: list[str], u: np.ndarray, f: np.ndarray, sign: int,
             parent: Rot | None = None) -> tuple[list, float]:
    """Quaternions for a 3-bone soft arm (character rest axes), plus its length factor.

    The arm moves as one soft tube: the whole arm (keeping its own rest curve) turns to point from the shoulder
    toward the hand (upper arm + forearm), and the dancer's elbow bend is spread evenly along the arm as one arc,
    so no bone joint ever shows as a corner."""
    parent = parent if parent is not None else Rot.identity()
    u = keep_clear(_norm(u), sign)
    f = _norm(f)
    if f[1] > 0.3:
        f = _norm(f - np.array([0, f[1] - 0.3, 0]))
    k = float(np.clip((raise_deg(u) - STYLE["straight_above_deg"]) / 40.0, 0, 1))
    f = _norm((1 - k) * f + k * u)                   # raised arms are straight
    target = keep_clear(_norm(u + f), sign)          # shoulder -> hand
    heads = [np.array(M[f"{n}__head"]) for n in names]
    tip = np.array(M[f"{names[-1]}__tail"])
    rest_dir = _norm(tip - heads[0])
    whole = _align(target, rest_dir)                 # the arm with its rest curve, turned as a whole
    axis = np.cross(u, f)
    bend = float(np.arctan2(np.linalg.norm(axis), np.dot(u, f))) * STYLE["bend_share"]
    axis = _norm(axis) if np.linalg.norm(axis) > 1e-6 else np.zeros(3)
    cs = [Rot.from_rotvec(axis * bend * t) * whole for t in (-1 / 3, 0.0, 1 / 3)]  # one even arc
    quats = [_local(M[names[0]], parent, cs[0]), _local(M[names[1]], cs[0], cs[1]),
             _local(M[names[2]], cs[1], cs[2])]
    return quats, 1.0  # the arm keeps its length; the `stretch` channel is there for characters that need it


def retarget(kp: dict, rig: dict, signals: dict | None = None) -> dict:
    fps = float(kp["fps"])
    w = to_blender(kp["world"])
    w = np.stack([np.stack([smooth(w[:, j, k], 1.5) for k in range(3)], -1) for j in range(w.shape[1])], 1)
    n = len(w)
    T = torso_frames(w)
    bones = {b["name"]: b for b in rig["bones"]}
    M = {k: np.array(b["matrix"], float) for k, b in bones.items()}
    M.update({f"{k}__head": b["head"] for k, b in bones.items()})
    M.update({f"{k}__tail": b["tail"] for k, b in bones.items()})
    rest_dir = {k: M[k][:, 1] for k in M if "__" not in k}  # bone axis = local Y

    if signals is None:
        from .signals import compute_signals
        signals = compute_signals(kp)
    from .legs import translate as translate_legs
    legs = translate_legs(kp, signals)
    height = float(rig.get("height_m", 1.4))

    # ---- body: turn, side tilt, forward lean of the dancer's torso, scaled to the character
    X, Z = T[:, :, 0], T[:, :, 2]
    turn = np.degrees(np.arctan2(X[:, 1], X[:, 0]))          # + = dancer's left side turns away from the camera
    turn = smooth(np.degrees(np.unwrap(np.radians(turn))), 2.0)
    side, lean = [], []
    for i in range(n):
        z = Rot.from_euler("z", -turn[i], degrees=True).apply(Z[i])
        side.append(np.degrees(np.arctan2(z[0], z[2])))       # + = tilt toward the character's left (+X)
        lean.append(np.degrees(np.arctan2(-z[1], z[2])))      # + = forward (toward the camera, -Y)
    side = smooth(np.array(side), 1.0) + legs["tilt"]
    lean = soft_limit(smooth(np.array(lean), 1.0), BODY["lean_max"])
    turn = soft_limit(turn, BODY["turn_max"])

    # arm directions (image-based in the picture plane), in Blender axes
    aspect = image_aspect(kp)
    upper = {"L": limb_dirs(kp, w, L_SH, L_EL, aspect), "R": limb_dirs(kp, w, R_SH, R_EL, aspect)}
    fore = {"L": limb_dirs(kp, w, L_EL, L_WR, aspect), "R": limb_dirs(kp, w, R_EL, R_WR, aspect)}

    # arms lead, the body follows: lean toward the higher arm a moment later
    arm_up = {k: np.array([(T[i].T @ upper[k][i])[2] for i in range(n)]) for k in ("L", "R")}
    delay = int(round(STYLE["follow_delay_s"] * fps))
    diff = arm_up["L"] - arm_up["R"]
    follow = np.r_[np.full(delay, diff[0]), diff[:n - delay]] if delay < n else np.zeros(n)
    side = soft_limit(side + STYLE["follow_tilt_deg"] * smooth(follow, 3.0), BODY["side_max"])
    body = [Rot.from_euler("z", turn[i], degrees=True) * Rot.from_euler("y", side[i], degrees=True)
            * Rot.from_euler("x", lean[i], degrees=True) for i in range(n)]

    ch: dict[str, list] = {k: [] for k in ["root.quat", "root.loc", "root.squash"]}
    ident = Rot.identity()
    for i in range(n):
        ch["root.quat"].append(_local(M["root"], ident, body[i]))
        # root rest frame: x = across (character's left), y = up
        ch["root.loc"].append([round(float(legs["shift"][i] * height), 4), round(float(legs["lift"][i] * height), 4), 0.0])
    ch["root.squash"] = [round(float(x), 4) for x in legs["squash"]]

    # ---- arms: copied in the torso frame (the body bones already carry the body rotation)
    for side_name, sign, (sh, el, wr) in (("L", 1, (L_SH, L_EL, L_WR)), ("R", -1, (R_SH, R_EL, R_WR))):
        names = [f"arm_{side_name}_{k}" for k in (1, 2, 3)]
        if not all(nm in M for nm in names):
            continue
        for nm in names:
            ch[f"{nm}.quat"] = []
            ch[f"{nm}.stretch"] = []
        socket = f"arm_{side_name}_0" if f"arm_{side_name}_0" in M else None
        if socket:
            ch[f"{socket}.quat"] = []
        lag_f = int(round(STYLE["forearm_lag_s"] * fps))
        for i in range(n):
            Ti = T[i]
            j = max(0, i - lag_f)
            u = Ti.T @ upper[side_name][i]
            f = Ti.T @ fore[side_name][j]  # forearm from a moment ago: the soft end trails
            c0 = socket_rot(M, [f"arm_{side_name}_{k}" for k in (1, 2, 3)], u, f, sign) if socket else Rot.identity()
            if socket:
                ch[f"{socket}.quat"].append(_local(M[socket], ident, c0))
            quats, st = arm_pose(M, names, u, f, sign, parent=c0)
            for nm, q in zip(names, quats):
                ch[f"{nm}.quat"].append(q)
                ch[f"{nm}.stretch"].append(st)

    # ---- hair: trails the body (keeps where the body was a moment ago)
    lag = max(1, int(round(HAIR["lag_s"] * fps)))
    hair_chains = [[nm for nm in M if nm.startswith(pre) and nm[len(pre):].isdigit()]
                   for pre in ("hair_", "hair_L_", "hair_R_")]
    hair_chains = [sorted(c, key=lambda x: int(x.rsplit("_", 1)[1])) for c in hair_chains if c]
    hair = [nm for c in hair_chains for nm in c]
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
            for chain in hair_chains:
                parent = body[i]
                share = HAIR["share"] if len(chain) == 3 else (0.4, 0.6)
                for k, nm in enumerate(chain):
                    target = Rot.from_rotvec(delta * share[k]) * parent
                    ch[f"{nm}.quat"].append(_local(M[nm], parent, target))
                    parent = target

    return {"fps": fps, "n_frames": n, "character": rig.get("name", "character"), "channels": ch,
            "retarget": {"body": BODY, "hair": HAIR, "style": STYLE},
            "leg_events": [{k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in ev.items()}
                           for ev in legs["events"]]}
