"""Render a character from several angles in its rest pose.

    python scripts/preview_character.py doudou --out outputs/preview
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bpy  # noqa: E402

from dancebuddy.characters.spec import CharacterSpec  # noqa: E402
from dancebuddy.render import blender_utils as bu  # noqa: E402
from dancebuddy.render.animate import rest_pose  # noqa: E402
from dancebuddy.render.builders import build_character  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("character")
    parser.add_argument("--out", default="outputs/preview")
    parser.add_argument("--angles", default="0,40,90,180")
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--save-blend", action="store_true")
    args = parser.parse_args()

    spec = CharacterSpec.load(args.character)
    bu.reset_scene()
    nodes = build_character(spec.name)
    rest_pose(nodes, spec)
    bu.setup_lights_and_world()
    bu.setup_render((args.size, args.size), samples=args.samples, transparent=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str((out / f"{spec.name}.blend").resolve()))
    for angle in [float(a) for a in args.angles.split(",")]:
        cam = bu.add_camera(azimuth_deg=angle)
        bpy.context.scene.render.filepath = str((out / f"{spec.name}_{int(angle):03d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)
        print("rendered", bpy.context.scene.render.filepath)


if __name__ == "__main__":
    main()
