"""Shared Blender scene setup: camera, lights, floor, render settings."""

import math
import sys

import bpy
from mathutils import Vector


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_stage(target_z=0.95, height=1.9):
    scene = bpy.context.scene
    # floor
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    m = bpy.data.materials.new("Floor")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.82, 0.84, 0.88, 1)
    m.diffuse_color = (0.82, 0.84, 0.88, 1)
    floor.data.materials.append(m)
    # world
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.9, 0.93, 0.97, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    # lights
    bpy.ops.object.light_add(type="SUN", location=(3, -4, 6))
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))
    bpy.ops.object.light_add(type="AREA", location=(-3, -3, 3))
    fill = bpy.context.active_object
    fill.data.energy = 120
    fill.data.size = 4
    fill.rotation_euler = (math.radians(60), 0, math.radians(-45))
    # camera, framed for a portrait-ish view with room to bounce
    bpy.ops.object.camera_add(location=(0, -(3.2 * height + 0.6), target_z + 0.6 * height / 1.9))
    cam = bpy.context.active_object
    direction = Vector((0, 0, target_z + 0.1)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 65
    scene.camera = cam


def setup_render(engine="auto", width=540, height=960):
    scene = bpy.context.scene
    engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    if engine == "auto":
        engine = "eevee" if sys.platform == "darwin" else "cycles"
    if engine == "eevee":
        ident = next((e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if e in engines), "CYCLES")
    elif engine == "workbench":
        ident = "BLENDER_WORKBENCH"
    else:
        ident = "CYCLES"
    scene.render.engine = ident
    if ident == "CYCLES":
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        scene.cycles.device = "CPU"
    elif ident.startswith("BLENDER_EEVEE"):
        try:
            scene.eevee.taa_render_samples = 16
        except AttributeError:
            pass
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass
    print(f"render engine: {ident} at {width}x{height}")
