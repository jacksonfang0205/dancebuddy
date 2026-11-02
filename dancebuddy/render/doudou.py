"""Procedural model of Doudou: a plush potato body with curly hair, blue fluffy arms and no legs.

The character faces -Y, stands on z = 0 and is about 1 unit tall. Animation nodes (empties) match the channel
bindings in characters/specs/doudou.json: root -> body -> arm_L / arm_R.
"""

from __future__ import annotations

import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .blender_utils import align_to_normal, assign, empty, material, parent_keep_world, shade_smooth

SKIN = "#cf9160"
ARM_BLUE = "#4f73d8"
HAIR_SHADES = ("#6b3119", "#7d3d1f", "#552511")
CENTER = Vector((0.0, 0.0, 0.5))


# ---------------------------------------------------------------- body


def _body_shape(v: Vector, rng: random.Random) -> Vector:
    """Map a unit-sphere vertex onto the potato body."""
    t = (v.z + 1) / 2  # 0 at the bottom, 1 at the top
    widen = 1 + 0.2 * (1 - t) ** 1.5 - 0.1 * t**2
    x, y, z = v.x * 0.33 * widen, v.y * 0.29 * widen, t
    if z < 0.06:  # flatten the bottom so it sits on the ground
        z = 0.06 - (0.06 - z) * 0.3
    lump = 1 + 0.018 * math.sin(3.1 * v.x + 2.3 * v.z) * math.cos(2.7 * v.y + 1.1 * v.z)
    return Vector((x * lump, y * lump, z))


def _build_body_mesh() -> tuple[bpy.types.Object, BVHTree, bmesh.types.BMesh]:
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=40, radius=1.0)
    rng = random.Random(0)
    for vert in bm.verts:
        vert.co = _body_shape(vert.co.copy(), rng)
    bm.normal_update()
    mesh = bpy.data.meshes.new("doudou_body")
    bm.to_mesh(mesh)
    obj = bpy.data.objects.new("doudou_body", mesh)
    bpy.context.scene.collection.objects.link(obj)
    shade_smooth(obj)
    obj.modifiers.new("subsurf", "SUBSURF").levels = 1
    tree = BVHTree.FromBMesh(bm)
    return obj, tree, bm


def _direction(azimuth: float, elevation: float) -> Vector:
    """Azimuth 0 is the character's front (-Y); positive azimuth turns toward its left (+X)."""
    return Vector((math.sin(azimuth) * math.cos(elevation), -math.cos(azimuth) * math.cos(elevation), math.sin(elevation)))


def _surface(tree: BVHTree, azimuth: float, elevation: float) -> tuple[Vector, Vector]:
    d = _direction(azimuth, elevation)
    hit, normal, _, _ = tree.ray_cast(CENTER + d * 3.0, -d)
    if hit is None:
        raise RuntimeError(f"no body surface at azimuth={azimuth}, elevation={elevation}")
    return hit, normal.normalized()


# ---------------------------------------------------------------- small mesh helpers


def _sphere(name: str, location: Vector, radius: float, mat, scale=(1, 1, 1), normal: Vector | None = None, segments: int = 24):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=segments // 2, radius=radius)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.scale = scale
    if normal is not None:
        align_to_normal(obj, normal)
    shade_smooth(obj)
    assign(obj, mat)
    return obj


def _curve(name: str, points: list[Vector], bevel: float, mat, closed: bool = False, kind: str = "NURBS"):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.bevel_depth = bevel
    data.bevel_resolution = 3
    spline = data.splines.new(kind)
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1.0)
    spline.use_cyclic_u = closed
    spline.use_endpoint_u = not closed
    spline.order_u = min(4, len(points))
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    data.materials.append(mat)
    return obj


# ---------------------------------------------------------------- face


def _build_face(tree: BVHTree) -> list[bpy.types.Object]:
    white = material("eye_white", "#f6f4ef", roughness=0.25)
    iris = material("iris", "#3d82d8", roughness=0.2)
    pupil = material("pupil", "#111111", roughness=0.2)
    glint = material("glint", "#ffffff", emission=4.0)
    brown = material("brow", "#5b3020", roughness=0.7)
    freckle = material("freckle", "#a35f3e", roughness=0.8)
    objs = []
    for side, az in (("L", 0.25), ("R", -0.25)):
        loc, n = _surface(tree, az, 0.3)
        objs.append(_sphere(f"eye_{side}", loc - n * 0.014, 0.062, white, scale=(1, 1.1, 0.75), normal=n))
        objs.append(_sphere(f"iris_{side}", loc + n * 0.03, 0.045, iris, scale=(1, 1.08, 0.35), normal=n))
        objs.append(_sphere(f"pupil_{side}", loc + n * 0.04, 0.021, pupil, scale=(1, 1.1, 0.35), normal=n))
        up = Vector((0, 0, 1))
        objs.append(_sphere(f"glint_{side}", loc + n * 0.047 + up * 0.016 - Vector((0.008, 0, 0)), 0.007, glint, normal=n))
        brow = []
        for k in range(5):
            a = az + (k - 2) * 0.045
            p, pn = _surface(tree, a, 0.47 + 0.012 * (1 - abs(k - 2) / 2))
            brow.append(p + pn * 0.004)
        objs.append(_curve(f"brow_{side}", brow, 0.0045, brown))
        for k, (da, de) in enumerate(((0.16, 0.18), (0.21, 0.16), (0.19, 0.12), (0.25, 0.19))):
            p, pn = _surface(tree, az + math.copysign(da, az), de)
            objs.append(_sphere(f"freckle_{side}{k}", p, 0.0075, freckle, scale=(1, 1, 0.4), normal=pn, segments=12))
    smile = []
    for k in range(7):
        a = -0.1 + k * (0.2 / 6)
        p, pn = _surface(tree, a, 0.15 + 0.028 * ((a / 0.1) ** 2))
        smile.append(p + pn * 0.003)
    objs.append(_curve("mouth", smile, 0.0055, brown))
    return objs


def _build_heart(tree: BVHTree) -> list[bpy.types.Object]:
    red = material("heart", "#c8242d", roughness=0.5)
    tape = material("bandage", "#e9bf93", roughness=0.7, bump=0.15, bump_scale=200)
    loc, n = _surface(tree, -0.3, -0.32)

    pts = []
    for k in range(40):
        t = 2 * math.pi * k / 40
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append(Vector((x * 0.0042, y * 0.0042, 0)))
    data = bpy.data.curves.new("heart", "CURVE")
    data.dimensions = "2D"
    data.fill_mode = "BOTH"
    data.extrude = 0.006
    data.bevel_depth = 0.004
    spline = data.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for p, co in zip(spline.points, pts):
        p.co = (*co, 1.0)
    spline.use_cyclic_u = True
    heart = bpy.data.objects.new("heart", data)
    bpy.context.scene.collection.objects.link(heart)
    data.materials.append(red)
    heart.location = loc + n * 0.004
    align_to_normal(heart, n)

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    mesh = bpy.data.meshes.new("bandage")
    bm.to_mesh(mesh)
    bm.free()
    band = bpy.data.objects.new("bandage", mesh)
    bpy.context.scene.collection.objects.link(band)
    band.location = loc + n * 0.012
    align_to_normal(band, n)
    band.rotation_quaternion = band.rotation_quaternion @ Matrix.Rotation(math.radians(-28), 4, "Z").to_quaternion()
    band.scale = (0.12, 0.034, 0.006)
    band.modifiers.new("bevel", "BEVEL").width = 0.3
    assign(band, tape)
    return [heart, band]


# ---------------------------------------------------------------- hair


def _in_hair(azimuth: float, elevation: float) -> bool:
    a = abs(math.atan2(math.sin(azimuth), math.cos(azimuth)))
    if a < 0.85:
        return elevation > 0.7
    if a < 1.6:
        return elevation > 0.7 - (a - 0.85) / 0.75 * 0.4  # curls frame the face lower toward the sides
    return elevation > -0.9  # long at the back, almost to the bottom


def _build_hair(body_bm: bmesh.types.BMesh, tree: BVHTree, count: int = 700) -> list[bpy.types.Object]:
    rng = random.Random(7)
    mats = [material(f"hair_{i}", c, roughness=0.38, sheen=0.2) for i, c in enumerate(HAIR_SHADES)]

    # A shell under the curls so no skin shows through.
    cap = body_bm.copy()
    remove = []
    for face in cap.faces:
        c = face.calc_center_median() - CENTER
        az = math.atan2(c.x, -c.y)
        el = math.asin(max(-1.0, min(1.0, c.normalized().z)))
        if not _in_hair(az, el):
            remove.append(face)
    bmesh.ops.delete(cap, geom=remove, context="FACES")
    cap.normal_update()
    for v in cap.verts:
        v.co += v.normal * 0.04
    mesh = bpy.data.meshes.new("hair_cap")
    cap.to_mesh(mesh)
    cap.free()
    cap_obj = bpy.data.objects.new("hair_cap", mesh)
    bpy.context.scene.collection.objects.link(cap_obj)
    shade_smooth(cap_obj)
    assign(cap_obj, material("hair_cap", HAIR_SHADES[2], roughness=0.6, bump=0.6, bump_scale=40))

    data = bpy.data.curves.new("hair_curls", "CURVE")
    data.dimensions = "3D"
    data.bevel_depth = 0.017
    data.bevel_resolution = 2
    data.resolution_u = 6
    for m in mats:
        data.materials.append(m)
    placed = 0
    while placed < count:
        az = rng.uniform(-math.pi, math.pi)
        el = math.asin(rng.uniform(-1, 1))
        if not _in_hair(az, el):
            continue
        loc, n = _surface(tree, az, el)
        front = abs(az) < 1.1
        flow = Vector((0, 1, 0.35)) if front or n.z > 0.85 else Vector((0, 0, -1))  # front curls sweep back
        down = flow - n * n.dot(flow)
        axis = down.normalized()
        side = axis.cross(n).normalized()
        length = rng.uniform(0.09, 0.15) * (0.6 if front else 1.0) + (0.08 if abs(az) > 1.6 and el < 0.3 else 0.0)
        radius = rng.uniform(0.024, 0.034)
        turns = length / rng.uniform(0.036, 0.046)
        start = loc + n * rng.uniform(0.05, 0.075)
        phase = rng.uniform(0, 2 * math.pi)
        steps = max(8, int(turns * 8))
        spline = data.splines.new("NURBS")
        spline.points.add(steps - 1)
        for k, p in enumerate(spline.points):
            s = k / (steps - 1)
            ang = phase + 2 * math.pi * turns * s
            co = start + axis * (s * length) + n * (radius * math.cos(ang)) + side * (radius * math.sin(ang))
            p.co = (*co, 1.0)
        spline.use_endpoint_u = True
        spline.order_u = 4
        spline.material_index = rng.randrange(len(mats))
        placed += 1
    curls = bpy.data.objects.new("hair_curls", data)
    bpy.context.scene.collection.objects.link(curls)
    return [cap_obj, curls]


# ---------------------------------------------------------------- arms


def _arm_mesh(name: str, length: float = 0.44) -> bpy.types.Mesh:
    """Arm in its local frame: shoulder at the origin, hanging down -Z, mitten hand with four fingers."""
    bm = bmesh.new()

    def blob(center, radii, segments=20):
        m = Matrix.Translation(center) @ Matrix.Diagonal((*radii, 1.0))
        bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=segments // 2, radius=1.0, matrix=m)

    # Upper arm as a chain of overlapping ellipsoids so it reads as a soft plush tube.
    for k in range(7):
        s = k / 6
        r = 0.088 - 0.012 * s
        blob(Vector((0, -0.01 * s, -s * length)), (r, r, r * 1.25))
    hand_z = -length - 0.05
    blob(Vector((0, -0.012, hand_z)), (0.104, 0.074, 0.09))
    for k, dx in enumerate((-0.05, -0.017, 0.017, 0.05)):
        blob(Vector((dx * 1.2, -0.03, hand_z - 0.085 + 0.01 * abs(dx) / 0.05)), (0.03, 0.032, 0.05), segments=12)
    blob(Vector((0.0, -0.065, hand_z + 0.005)), (0.022, 0.03, 0.035), segments=12)  # thumb toward the front
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _add_fuzz(obj: bpy.types.Object, count: int, length: float, children: int = 12) -> None:
    """Short particle hair for a plush look (rendered by Cycles)."""
    mod = obj.modifiers.new("fuzz", "PARTICLE_SYSTEM")
    settings = mod.particle_system.settings
    settings.type = "HAIR"
    settings.count = count
    settings.hair_length = length
    settings.child_type = "INTERPOLATED"
    settings.rendered_child_count = children
    settings.root_radius = 0.0012
    settings.tip_radius = 0.0002
    settings.use_hair_bspline = True
    settings.factor_random = 0.3
    settings.brownian_factor = 0.004


def _build_arms(tree: BVHTree, body_node: bpy.types.Object, fuzz: bool) -> dict[str, bpy.types.Object]:
    blue = material("arm_fuzz", ARM_BLUE, roughness=0.9, sheen=0.25, bump=0.5, bump_scale=120)
    nodes = {}
    for side, az in (("L", 1.32), ("R", -1.32)):
        loc, n = _surface(tree, az, 0.18)
        node = empty(f"arm_{side}", location=loc - n * 0.03, parent=body_node)
        node.rotation_mode = "XYZ"
        arm = bpy.data.objects.new(f"arm_{side}_mesh", _arm_mesh(f"arm_{side}_mesh"))
        bpy.context.scene.collection.objects.link(arm)
        arm.location = node.location
        shade_smooth(arm)
        arm.modifiers.new("subsurf", "SUBSURF").levels = 1
        assign(arm, blue)
        if fuzz:
            _add_fuzz(arm, count=3500, length=0.028)
        parent_keep_world(arm, node)
        nodes[f"arm_{side}"] = node
    return nodes


# ---------------------------------------------------------------- assembly


def build(fuzz: bool = True) -> dict[str, bpy.types.Object]:
    """fuzz adds particle fur to the arms: nicer stills, but slow and noisy in animation renders."""
    root = empty("root")
    body_node = empty("body", parent=root)
    body_node.rotation_mode = "XYZ"

    body, tree, body_bm = _build_body_mesh()
    assign(body, material("skin", SKIN, roughness=0.75, sheen=0.6, bump=0.25, bump_scale=180, subsurface=0.05))
    parts = [body] + _build_face(tree) + _build_heart(tree) + _build_hair(body_bm, tree)
    for obj in parts:
        parent_keep_world(obj, body_node)
    nodes = {"root": root, "body": body_node}
    nodes.update(_build_arms(tree, body_node, fuzz))
    body_bm.free()
    return nodes
