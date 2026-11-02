"""Apply character channel values to the rig nodes and keyframe a full motion."""

from __future__ import annotations

import math

import bpy

from ..characters.spec import CharacterSpec
from ..motion.plan import CharacterMotion


def _base_state(nodes: dict) -> dict:
    """Remember each node's rest transform so channel values are applied as offsets."""
    return {
        name: {"location": tuple(obj.location), "rotation": tuple(obj.rotation_euler), "scale": tuple(obj.scale)}
        for name, obj in nodes.items()
    }


def set_pose(nodes: dict, spec: CharacterSpec, values: dict[str, float], base: dict) -> None:
    for name, obj in nodes.items():
        obj.location = base[name]["location"]
        obj.rotation_euler = base[name]["rotation"]
        obj.scale = base[name]["scale"]
    for ch_name, ch in spec.channels.items():
        value = values.get(ch_name, ch.rest)
        b = ch.bind
        obj = nodes[b["node"]]
        v = value * b.get("factor", 1.0)
        if b["prop"] == "location":
            loc = list(obj.location)
            loc[b["axis"]] += v * spec.body_height_m
            obj.location = loc
        elif b["prop"] == "rotation":
            rot = list(obj.rotation_euler)
            rot[b["axis"]] += math.radians(v)
            obj.rotation_euler = rot
        elif b["prop"] == "squash":  # stretch along z, preserve volume
            sz = max(0.2, 1.0 + v)
            sxy = 1.0 / math.sqrt(sz)
            obj.scale = (obj.scale[0] * sxy, obj.scale[1] * sxy, obj.scale[2] * sz)
        else:
            raise ValueError(f"unknown binding prop {b['prop']!r}")


def rest_pose(nodes: dict, spec: CharacterSpec) -> dict:
    base = _base_state(nodes)
    set_pose(nodes, spec, {}, base)
    return base


def keyframe_motion(nodes: dict, spec: CharacterSpec, motion: CharacterMotion, base: dict | None = None) -> None:
    scene = bpy.context.scene
    base = base or _base_state(nodes)
    scene.render.fps = int(round(motion.fps))
    scene.frame_start, scene.frame_end = 1, motion.num_frames
    for f in range(motion.num_frames):
        values = {name: float(v[f]) for name, v in motion.channels.items()}
        set_pose(nodes, spec, values, base)
        for obj in nodes.values():
            obj.keyframe_insert("location", frame=f + 1)
            obj.keyframe_insert("rotation_euler", frame=f + 1)
            obj.keyframe_insert("scale", frame=f + 1)
