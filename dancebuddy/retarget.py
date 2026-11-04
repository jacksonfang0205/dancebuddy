"""Copy a dancer onto a character, following the character's motion spec.

What the body can do is copied frame by frame; what it cannot is translated (assets/<name>_motion_spec.json):
  * arms   : one bendable bone per arm: it points from the shoulder toward the hand and curves from the dancer's
             upper-arm direction to her forearm direction, keeping its width; kept in front of the hair and body
  * body   : the dancer's torso turn / side tilt / forward lean drive the whole body around its base, at full size
             and smoothly compressed near the character's limits (never cut flat, so peaks keep their timing)
  * legs   : no legs -> every jump, kick, step, squat and move across is translated (legs.py) into hops,
             squash and stretch, leans and slides of the body
  * hair   : not animated: it moves with the head
  * groove : a beat-synced wiggle of the soft body (sway, twist, upper-body counter-bend, a squash pulse)

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
# the body is a soft bag that keeps its volume (motion spec body_soft): part of every sideways tilt is a bend of the
# body itself (a C curve from the base), a hip lift bends it toward the lifted hip, hips swaying under a still torso
# make an S, and the pressed bag bulges: the lower body gets wider and shorter as it bends
SOFT = {"bend_share": 0.6,          # part of the side tilt shown as a bend (the rest tilts it on its base)
        "hip_bend": 0.8,            # bend (deg) per deg of the dancer's hip tilt
        "sway_bend": 25.0,          # S-curve bend (deg) per torso length of the hips' sideways shift
        "bend_max": 22.0,           # per body bone
        "bulge_per_deg": 0.005}     # widening of the lower body per deg of bend (height shrinks to keep the volume)


def soft_limit(x: np.ndarray, limit: float) -> np.ndarray:
    """Full size for small values, smoothly compressed toward the limit (peaks keep their timing)."""
    return limit * np.tanh(np.asarray(x, float) / limit)
# arm directions: in-plane from the image (reliable), depth from the 3D estimate (monocular depth is not)
ARM_DEPTH_GAIN = 0.4
# groove: a beat-synced wiggle of the soft body (sway and twist a quarter beat apart, upper body bending the
# other way a moment later), backing off while the dancer's own body moves a lot
GROOVE = {"tilt_deg": 5.0, "twist_deg": 7.0, "bend_deg": 5.0, "bend_delay_s": 0.08, "pulse_squash": 0.03,
          "busy_deg": 15.0}
# style from the spec's "tends_to": arms lead and the body follows, soft arms, squash low / stretch high
STYLE = {"straight_above_deg": 120,         # raised arms straighten out (reference: arms up are straight)
         "bend_share": 0.8,                 # how much of the dancer's elbow bend the soft arm takes, as one arc
         # the shoulder is a socket that slides up the side of the body as the arm rises (reference: a raised
         # arm starts beside the face, a hanging arm beside the chest), so raised hands get above the head
         "socket_slide": 0.12,              # how far the seam slides up the body for a vertical arm (body heights)
         "root_share": 0.5,                 # part of the arm's turn taken at the seam, so the root turns with the
                                            # arm instead of folding (the shoulder pivots just inside the seam)
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


def hip_tilt(kp: dict, aspect: float) -> np.ndarray:
    """(N,) degrees the dancer's pelvis is tilted in the picture, + = her left hip higher (a hip lift)."""
    im = kp["image"]
    d = im[:, L_HIP, :2] - im[:, R_HIP, :2]
    ang = np.degrees(np.arctan2(-d[:, 1], np.abs(d[:, 0]) * aspect + 1e-6))
    return smooth(ang - np.median(ang), 2.0)


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
    """The hidden shoulder bone turns with the arm (a share of its turn), so a big arm turn is not all taken at
    the arm's root, where it would fold the arm."""
    u_c = keep_clear(_norm(u), sign)
    target = keep_clear(_norm(u_c + _norm(f)), sign)
    rest = _norm(np.array(M[f"{names[-1]}__tail"]) - np.array(M[f"{names[0]}__head"]))
    return Rot.from_rotvec(_align(target, rest).as_rotvec() * STYLE["root_share"])


def socket_offset(u: np.ndarray, sign: int, height: float) -> np.ndarray:
    """Where the seam slides as the arm rises (armature axes, metres): up the body side, a little inward as the
    body narrows, and a little forward so the arm stays in front of the hair."""
    a = raise_deg(keep_clear(_norm(u), sign))
    k = float(np.clip((a - STYLE["socket_slide_from_deg"]) / (175 - STYLE["socket_slide_from_deg"]), 0, 1))
    dz = STYLE["socket_slide"] * height * k * k * (3 - 2 * k)
    return np.array([-sign * 0.3 * dz, -0.15 * dz, dz])


def _hand_roll(n: np.ndarray, d: np.ndarray) -> float:
    """Turn (radians, around the hand's direction d) that brings the hand's flat face n toward the camera (-Y):
    palm or back, whichever is closer. 0 when the hand points at the camera."""
    n_p, cam = n - d * (n @ d), np.array([0.0, -1.0, 0.0]) - d * (-d[1])
    if np.linalg.norm(n_p) < 1e-6 or np.linalg.norm(cam) < 0.2:
        return 0.0
    n_p, cam = _norm(n_p), _norm(cam)
    if n_p @ cam < 0:
        cam = -cam
    return float(np.arctan2(np.cross(n_p, cam) @ d, n_p @ cam))


def arm_pose(M: dict, name: str, u: np.ndarray, f: np.ndarray, sign: int,
             parent: Rot | None = None, scale: float = 1.0, face: list | None = None) -> tuple[list, list]:
    """Pose of a one-bone soft arm (a B-bone): its rotation [w, x, y, z] and its curve [curve in x, in z,
    out x, out z, roll in, roll out].

    The arm is one bendable tube that keeps its width and length: the bone points from the shoulder toward the
    hand, the curve leans its start toward the dancer's upper arm and its end toward the forearm (the elbow
    bend as one even arc), and the hand's flat face turns toward the camera through a twist that grows along the
    arm (none at the seam)."""
    parent = parent if parent is not None else Rot.identity()
    u = keep_clear(_norm(u), sign)
    f = _norm(f)
    if f[1] > 0.3:
        f = _norm(f - np.array([0, f[1] - 0.3, 0]))
    k = float(np.clip((raise_deg(u) - STYLE["straight_above_deg"]) / 40.0, 0, 1))
    f = _norm((1 - k) * f + k * u)                   # raised arms are straight
    target = keep_clear(_norm(u + f), sign)          # shoulder -> hand
    head, tail = np.array(M[f"{name}__head"], float), np.array(M[f"{name}__tail"], float)
    length = float(np.linalg.norm(tail - head))
    c = _align(target, M[name][:, 1])                # the arm with its rest shape, turned as a whole
    if scale != 1.0:                                 # the agent's choice: exaggerate or soften the arm move
        c = Rot.from_rotvec(c.as_rotvec() * scale)
    frame = c * Rot.from_matrix(M[name])             # bone axes -> armature
    chord = frame.apply([0.0, 1.0, 0.0])
    share = STYLE["bend_share"] * scale

    def lean(vec: np.ndarray) -> tuple[np.ndarray, float]:
        """Handle offset (bone x, z) that tilts the curve's tangent toward vec, and that tilt (radians)."""
        loc = frame.inv().apply(vec)
        perp = np.array([loc[0], loc[2]])
        ang = min(float(np.arctan2(np.linalg.norm(perp), loc[1])) * share, np.radians(60))
        if np.linalg.norm(perp) < 1e-6:
            return np.zeros(2), 0.0
        return np.tan(ang) * length / 3 * perp / np.linalg.norm(perp), ang   # B-bone handle = length / 3

    cin, _ = lean(u)
    out_off, a_out = lean(f)
    cout = -out_off                                   # an end handle offset tilts the end tangent the other way
    roll = 0.0
    if face is not None:                              # the hand's direction and face at the end of the curve
        ax = np.cross(chord, f)
        bend_end = Rot.from_rotvec(_norm(ax) * a_out) if np.linalg.norm(ax) > 1e-6 else Rot.identity()
        roll = _hand_roll(bend_end.apply(c.apply(np.asarray(face, float))), bend_end.apply(chord))
    q = _local(M[name], parent, c)
    curve = [round(float(x), 5) for x in (cin[0], cin[1], cout[0], cout[1], 0.0, roll)]
    return q, curve


def beat_phase(n: int, beats: list[int]) -> np.ndarray:
    """Continuous beat count per frame (k at the k-th beat), extrapolated with the median beat period."""
    b = sorted(int(x) for x in beats)
    if len(b) < 2:
        return np.arange(n) / 15.0
    period = float(np.median(np.diff(b)))
    xs = np.r_[b[0] - period * 50, b, b[-1] + period * 50]
    ks = np.r_[-50, np.arange(len(b)), len(b) - 1 + 50]
    return np.interp(np.arange(n), xs, ks)


def groove(n: int, fps: float, beats: list[int] | None, busy: np.ndarray) -> dict:
    """Beat-synced wiggle: side sway (one side per beat), twist a quarter beat later, an upper-body counter-bend,
    and a small squash pulse on each beat; scaled down while the body is busy with the dancer's own moves."""
    if not beats:
        z = np.zeros(n)
        return {"tilt": z, "twist": z, "bend": z, "pulse": z}
    ph = beat_phase(n, beats)
    amp = 1.0 - 0.7 * np.clip(busy / GROOVE["busy_deg"], 0, 1)
    sway = np.sin(np.pi * ph)
    tilt = GROOVE["tilt_deg"] * amp * sway
    twist = GROOVE["twist_deg"] * amp * np.sin(np.pi * ph - np.pi / 2)
    d = max(1, int(round(GROOVE["bend_delay_s"] * fps)))
    bend = -GROOVE["bend_deg"] * np.r_[np.zeros(d), (amp * sway)[:-d]]
    frac = ph % 1.0
    pulse = GROOVE["pulse_squash"] * amp * np.exp(-0.5 * (np.minimum(frac, 1 - frac) / 0.12) ** 2)
    return {"tilt": tilt, "twist": twist, "bend": bend, "pulse": pulse}


def retarget(kp: dict, rig: dict, signals: dict | None = None, beats: list[int] | None = None,
             plan: dict | None = None, score: dict | None = None) -> dict:
    """plan: the agent's decisions (copy_plan.py); None = the plain copy. score: the dance score (move units)."""
    from .copy_plan import emphasis_curves, knob_curves, leg_overrides
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
    legs = translate_legs(kp, signals, leg_overrides(plan))
    height = float(rig.get("height_m", 1.4))
    knob = knob_curves(plan, score, n, fps)
    emph = emphasis_curves(plan, n, fps, height)

    # ---- body: turn, side tilt, forward lean of the dancer's torso, scaled to the character
    X, Z = T[:, :, 0], T[:, :, 2]
    turn = np.degrees(np.arctan2(X[:, 1], X[:, 0]))          # + = dancer's left side turns away from the camera
    turn = smooth(np.degrees(np.unwrap(np.radians(turn))), 2.0)
    side, lean = [], []
    for i in range(n):
        z = Rot.from_euler("z", -turn[i], degrees=True).apply(Z[i])
        side.append(np.degrees(np.arctan2(z[0], z[2])))       # + = tilt toward the character's left (+X)
        lean.append(np.degrees(np.arctan2(-z[1], z[2])))      # + = forward (toward the camera, -Y)
    side = smooth(np.array(side), 1.0) * knob["body"] + legs["tilt"] + emph["tilt"]
    lean = soft_limit(smooth(np.array(lean), 1.0) * knob["body"], BODY["lean_max"])
    turn = soft_limit(turn * knob["body"] + emph["twist"], BODY["turn_max"])

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
    busy = np.maximum.reduce([np.abs(side), np.abs(lean), np.abs(turn) / 2])
    gv = groove(n, fps, beats, busy)
    gv = {k: v * knob["groove"] for k, v in gv.items()}
    side = side + gv["tilt"]
    turn = turn + gv["twist"]
    # ---- soft body: split the side tilt into a tilt on the base and a bend of the body (C), add the hip lift and the
    # hips' sway (S), and let the bent lower body bulge
    bend_c = side * SOFT["bend_share"]
    side = side - bend_c                                       # the rigid part, on the base
    hip_t = hip_tilt(kp, aspect)                               # + = the dancer's left hip higher
    sway = smooth(np.asarray(signals.get("hip_sway", np.zeros(n)), float), 1.5)
    lower_bend = soft_limit(0.5 * bend_c - SOFT["hip_bend"] * hip_t + SOFT["sway_bend"] * sway, SOFT["bend_max"])
    upper_bend_deg = soft_limit(0.5 * bend_c + 1.5 * SOFT["hip_bend"] * hip_t - SOFT["sway_bend"] * sway,
                                SOFT["bend_max"])
    bulge = 1.0 + SOFT["bulge_per_deg"] * (np.abs(lower_bend) + 0.5 * np.abs(upper_bend_deg))
    body = [Rot.from_euler("z", turn[i], degrees=True) * Rot.from_euler("y", side[i], degrees=True)
            * Rot.from_euler("x", lean[i], degrees=True) for i in range(n)]
    # upper body: S-shaped wiggle, plus the dancer's head nods and tilts when the plan asks for them
    nod = soft_limit(smooth(np.asarray(signals.get("head_nod", np.zeros(n)), float), 1.0) * knob["head"] + emph["nod"], 20.0)
    htilt = soft_limit(smooth(np.asarray(signals.get("head_tilt", np.zeros(n)), float), 1.0) * knob["head"], 20.0)
    upper_bend = [Rot.from_euler("y", gv["bend"][i] + htilt[i] + upper_bend_deg[i], degrees=True)
                  * Rot.from_euler("x", nod[i], degrees=True) for i in range(n)]

    ch: dict[str, list] = {k: [] for k in ["root.quat", "root.loc", "root.squash"]}
    ident = Rot.identity()
    for i in range(n):
        ch["root.quat"].append(_local(M["root"], ident, body[i]))
        # root rest frame: x = across (character's left), y = up
        # tilting on the base would push one side of the bottom into the floor: rise like a roly-poly toy
        tilt_up = body[i].apply([0.0, 0.0, 1.0])
        roll = float(rig.get("base_radius_m", 0.0)) * np.sqrt(max(0.0, 1 - tilt_up[2] ** 2))
        ch["root.loc"].append([round(float(legs["shift"][i] * knob["travel"][i] * height), 4),
                               round(float((legs["lift"][i] + emph["lift"][i]) * height + roll), 4), 0.0])
    ch["root.squash"] = [round(float(x), 4) for x in np.clip(legs["squash"] + gv["pulse"] + emph["squash"], -0.1, 0.25)]
    if "body_upper" in M:
        ch["body_upper.quat"] = [_local(M["body_upper"], ident, upper_bend[i]) for i in range(n)]
    if "body_lower" in M:
        ch["body_lower.quat"] = [_local(M["body_lower"], ident, Rot.from_euler("y", lower_bend[i], degrees=True))
                                 for i in range(n)]
        # bone axes: y = up, x = across: wider and as much shorter, so the bag keeps its volume
        ch["body_lower.scale"] = [[round(float(b), 4), round(float(1 / b), 4), 1.0] for b in bulge]

    # ---- arms: copied in the torso frame (the body bones already carry the body rotation)
    for side_name, sign, (sh, el, wr) in (("L", 1, (L_SH, L_EL, L_WR)), ("R", -1, (R_SH, R_EL, R_WR))):
        name = f"arm_{side_name}_1"
        if name not in M:
            continue
        ch[f"{name}.quat"], ch[f"{name}.bbone"] = [], []
        socket = f"arm_{side_name}_0" if f"arm_{side_name}_0" in M else None
        if socket:
            ch[f"{socket}.quat"] = []
            ch[f"{socket}.loc"] = []
        lag_f = int(round(STYLE["forearm_lag_s"] * fps))
        for i in range(n):
            Ti = T[i]
            j = max(0, i - lag_f)
            u = Ti.T @ upper[side_name][i]
            f = Ti.T @ fore[side_name][j]  # forearm from a moment ago: the soft end trails
            c0 = socket_rot(M, [name], u, f, sign) if socket else Rot.identity()
            if socket:
                ch[f"{socket}.quat"].append(_local(M[socket], ident, c0))
                off = M[socket].T @ socket_offset(u, sign, height)  # armature -> the bone's rest frame
                ch[f"{socket}.loc"].append([round(float(x), 4) for x in off])
            q, curve = arm_pose(M, name, u, f, sign, parent=c0, scale=float(knob["arms"][i]),
                                face=rig.get("hand_face", {}).get(side_name))
            ch[f"{name}.quat"].append(q)
            ch[f"{name}.bbone"].append(curve)

    return {"fps": fps, "n_frames": n, "character": rig.get("name", "character"), "channels": ch,
            "retarget": {"body": BODY, "soft": SOFT, "style": STYLE, "groove": GROOVE},
            "leg_events": [{k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in ev.items()}
                           for ev in legs["events"]]}
