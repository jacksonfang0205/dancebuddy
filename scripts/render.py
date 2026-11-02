"""Render a motion (motion.json) on a character and composite it beside the dancer.

    python scripts/render.py outputs/dance_1/doudou/agent --video data/videos/dance_1.mp4
    python scripts/render.py outputs/dance_1/doudou/agent --frames-only        # transparent PNGs only

Writes <run_dir>/frames/*.png (transparent character) and <run_dir>/composite.mp4.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bpy  # noqa: E402

from dancebuddy.characters.spec import CharacterSpec  # noqa: E402
from dancebuddy.motion.plan import CharacterMotion  # noqa: E402
from dancebuddy.render import blender_utils as bu  # noqa: E402
from dancebuddy.render.animate import keyframe_motion, rest_pose  # noqa: E402
from dancebuddy.render.builders import build_character  # noqa: E402


def render_frames(run_dir: Path, character: str, size: int, samples: int, max_frames: int | None) -> Path:
    spec = CharacterSpec.load(character)
    motion = CharacterMotion.load(run_dir / "motion.json")
    bu.reset_scene()
    nodes = build_character(spec.name, fuzz=False)
    base = rest_pose(nodes, spec)
    keyframe_motion(nodes, spec, motion, base)
    bu.setup_lights_and_world()
    bu.setup_render((size, size), samples=samples, transparent=True)
    bu.add_camera(azimuth_deg=0.0, distance=3.6, height=0.8, target_z=0.6)
    scene = bpy.context.scene
    if max_frames:
        scene.frame_end = min(scene.frame_end, max_frames)
    frames = run_dir / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    scene.render.filepath = str(frames.resolve()) + "/f_"
    bpy.ops.render.render(animation=True)
    return frames


def composite(video: Path, frames: Path, fps: float, out: Path, label: str = "") -> None:
    """Dancer on the left, the character on a soft backdrop on the right, both at the video's height."""
    draw = (
        f",drawtext=text='{label}':x=16:y=16:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8"
        if label
        else ""
    )
    graph = (
        "[0:v]split[v0][v1];"
        "[v0]scale=-2:720,setsar=1[dancer];"
        "[v1]scale=720:720:force_original_aspect_ratio=increase,crop=720:720,boxblur=20:2,eq=brightness=0.05,setsar=1[bg];"
        "[1:v]scale=720:720[ch];"
        "[bg][ch]overlay=0:0:shortest=1[panel];"
        f"[dancer][panel]hstack=inputs=2:shortest=1{draw}[v]"
    )
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(video),
        "-framerate", f"{fps}", "-i", str(frames / "f_%04d.png"),
        "-filter_complex", graph,
        "-map", "[v]", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", str(out),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--video", help="dancer video to composite onto")
    parser.add_argument("--character", help="defaults to the run directory's parent name")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--label", default="")
    parser.add_argument("--frames-only", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    character = args.character or run_dir.parent.name
    frames = render_frames(run_dir, character, args.size, args.samples, args.max_frames)
    if args.frames_only or not args.video:
        print("frames in", frames)
        return
    motion = CharacterMotion.load(run_dir / "motion.json")
    out = run_dir / "composite.mp4"
    composite(Path(args.video), frames, motion.fps, out, args.label)
    print("wrote", out)


if __name__ == "__main__":
    main()
