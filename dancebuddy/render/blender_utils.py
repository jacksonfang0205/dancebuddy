"""Small helpers for building and rendering scenes with Blender's Python module (bpy)."""

from __future__ import annotations

import math

import bpy
from mathutils import Vector


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def hex_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """sRGB hex string to linear RGBA, which is what Blender color sockets expect."""
    h = hex_color.lstrip("#")
    srgb = [int(h[i: i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, alpha)


def material(
    name: str,
    color: str,
    roughness: float = 0.6,
    sheen: float = 0.0,
    bump: float = 0.0,
    bump_scale: float = 60.0,
    emission: float = 0.0,
    subsurface: float = 0.0,
) -> bpy.types.Material:
    """Principled material with optional felt-like sheen and fine noise bump."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    rgba = hex_rgba(color)
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    if sheen:
        bsdf.inputs["Sheen Weight"].default_value = sheen
        bsdf.inputs["Sheen Roughness"].default_value = 0.6
    if subsurface:
        bsdf.inputs["Subsurface Weight"].default_value = subsurface
        bsdf.inputs["Subsurface Radius"].default_value = (0.05, 0.03, 0.02)
    if emission:
        bsdf.inputs["Emission Color"].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = emission
    if bump:
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = bump_scale
        noise.inputs["Detail"].default_value = 8.0
        bump_node = nodes.new("ShaderNodeBump")
        bump_node.inputs["Strength"].default_value = bump
        links.new(noise.outputs["Fac"], bump_node.inputs["Height"])
        links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def assign(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def empty(name: str, location=(0.0, 0.0, 0.0), parent: bpy.types.Object | None = None) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 0.1
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    if parent is not None:
        parent_keep_world(obj, parent)
    return obj


def parent_keep_world(child: bpy.types.Object, parent: bpy.types.Object) -> None:
    bpy.context.view_layer.update()  # matrix_world is stale until the depsgraph updates
    world = child.matrix_world.copy()
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    child.matrix_world = world


def shade_smooth(obj: bpy.types.Object) -> None:
    for poly in obj.data.polygons:
        poly.use_smooth = True


def align_to_normal(obj: bpy.types.Object, normal: Vector) -> None:
    """Point the object's local +Z along a surface normal."""
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = normal.normalized().to_track_quat("Z", "Y")


def setup_render(resolution=(512, 512), samples: int = 32, transparent: bool = True) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"


def setup_lights_and_world(strength: float = 1.0) -> None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.75, 0.77, 0.8, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6 * strength
    for name, loc, energy, size in (
        ("key", (-1.8, -2.4, 2.6), 350, 2.0),
        ("fill", (2.2, -1.8, 1.4), 150, 2.5),
        ("rim", (0.5, 2.5, 2.2), 220, 1.5),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy * strength
        data.size = size
        light = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(light)
        light.location = loc
        direction = Vector((0, 0, 0.55)) - Vector(loc)
        light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(azimuth_deg: float = 0.0, distance: float = 3.2, height: float = 0.75, target_z: float = 0.5, lens: float = 50) -> bpy.types.Object:
    """Camera orbiting the character; azimuth 0 looks at the character's front (it faces -Y)."""
    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = lens
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    a = math.radians(azimuth_deg)
    cam.location = (distance * math.sin(a), -distance * math.cos(a), height)
    cam.rotation_euler = (Vector((0, 0, target_z)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam
