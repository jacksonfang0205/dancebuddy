"""Put a rigged character in one frame's pose from a curves.json (shared by rendering, the checks and the
evaluation measurements, so they all read the channels the same way).

Channel -> pose bone (bones point up, so bone-local Y is the bone axis):
  <bone>.lean / .twist / .sway  rotation X / Y / Z in degrees (Euler; the first prototype's channels)
  <bone>.quat                   rotation quaternion [w, x, y, z] in the bone's rest frame (overrides the Euler ones)
  <bone>.lift                   location Y (along the root bone = up)
  <bone>.loc                    head offset [x, y, z] in the bone's rest frame (e.g. a shoulder sliding up)
  <bone>.scale                  [x, y, z] in the bone's frame (e.g. a soft body bulging)
  <bone>.stretch                lengthen along the bone, keeping most of its thickness
  <bone>.squash                 shorter and as much wider (volume kept)
  <bone>.bbone                  B-bone curve [curve in x, in z, out x, out z, roll in, roll out, ease in, ease out]
  <arm>.fist                    shape key fist_<side> of the Body (0 open hand .. 1 fist)
"""

import math

AXIS = {"lean": 0, "twist": 1, "sway": 2}
BBONE_PROPS = ("bbone_curveinx", "bbone_curveinz", "bbone_curveoutx", "bbone_curveoutz", "bbone_rollin", "bbone_rollout",
               "bbone_easein", "bbone_easeout")


def pose_frame(rig, body, channels: dict, f: int) -> None:
    """Reset the rig and the fist shape keys, then apply frame f of every channel."""
    pbones = rig.pose.bones
    keys = body.data.shape_keys if body is not None else None
    if keys is not None:
        for kb in keys.key_blocks[1:]:
            kb.value = 0.0
    for pb in pbones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
        for prop in BBONE_PROPS:
            setattr(pb, prop, 0.0)
    for name, vals in channels.items():
        bone, ch = name.split(".", 1)
        if bone not in pbones:
            continue
        pb, v = pbones[bone], vals[f]
        if ch == "quat":
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = v
        elif ch in AXIS:
            r = list(pb.rotation_euler)
            r[AXIS[ch]] = math.radians(v)
            pb.rotation_euler = r
        elif ch == "lift":
            pb.location = (0, v, 0)
        elif ch == "bbone":
            for prop, x in zip(BBONE_PROPS, v):
                setattr(pb, prop, x)
        elif ch == "loc":
            pb.location = v
        elif ch == "scale":
            pb.scale = v
        elif ch == "stretch":
            st = max(0.3, v)
            pb.scale = (st ** -0.2, st, st ** -0.2)
        elif ch == "squash":
            s = max(0.3, 1 - v)
            pb.scale = (1 / math.sqrt(s), s, 1 / math.sqrt(s))
        elif ch == "fist" and keys is not None:
            key = f"fist_{bone.split('_')[1]}"
            if key in keys.key_blocks:
                keys.key_blocks[key].value = float(v)
