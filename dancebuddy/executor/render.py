"""Carry curves out in Blender: render a system's curves.json to frames, and put the dancer and the renders side by
side. Blender runs as the bpy module when it is installed in this Python, else as the Blender app ($BLENDER)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def blender_cmd(script: str, args: list[str]) -> list[str]:
    """Command running dancebuddy/blender/<script> with args."""
    path = str(ROOT / "dancebuddy" / "blender" / script)
    if importlib.util.find_spec("bpy") is not None and not os.environ.get("BLENDER"):
        return [sys.executable, path, *args]
    exe = os.environ.get("BLENDER") or shutil.which("blender")
    mac = "/Applications/Blender.app/Contents/MacOS/Blender"
    if not exe and Path(mac).exists():
        exe = mac
    if not exe:
        sys.exit("Blender not found: install the bpy module (uv sync --extra render) or set BLENDER=/path/to/blender")
    return [exe, "-b", "--factory-startup", "-P", path, "--", *args]


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print("$", " ".join(cmd[:4]), "...")
    subprocess.run(cmd, check=True, **({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}))


def render_system(run_dir: Path, system: str, char: Path, engine: str = "cycles", width: int = 540,
                  height: int = 960, fps: float = 30, preview: bool = False, max_frames: int = 0) -> Path | None:
    """Render <run>/<system>/curves.json at `fps` (a 60 fps clip renders every other frame; 0 = the clip's own) into
    <system>/frames, or a quick untextured half-size look into <system>/preview."""
    curves_path = Path(run_dir) / system / "curves.json"
    if not curves_path.exists():
        print(f"skip {system}: no {curves_path}")
        return None
    curves = json.loads(curves_path.read_text())
    src_fps = float(curves["fps"])
    step = max(1, int(round(src_fps / fps))) if fps else 1
    frames = Path(run_dir) / system / ("preview" if preview else "frames")
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    render_curves = curves_path
    if step > 1:  # keep every step-th frame, so the motion keeps its timing at the lower frame rate
        sub = dict(curves, fps=src_fps / step, channels={k: v[::step] for k, v in curves["channels"].items()})
        sub["n_frames"] = len(next(iter(sub["channels"].values())))
        render_curves = frames / "curves_render.json"
        render_curves.write_text(json.dumps(sub))
    engine = "workbench" if preview else engine
    width, height = (width // 2, height // 2) if preview else (width, height)
    (frames / "render_info.json").write_text(json.dumps(
        {"fps": src_fps / step, "step": step, "engine": engine, "size": [width, height], "preview": preview}))
    args = ["--blend", str(char) + ".blend", "--curves", str(render_curves), "--frames", str(frames),
            "--engine", engine, "--width", str(width), "--height", str(height)]
    if max_frames:
        args += ["--max-frames", str(max_frames)]
    print(f"render {system}: {engine} {width}x{height} at {src_fps / step:g} fps -> {frames}")
    run(blender_cmd("render_curves.py", args))
    return frames


def compose(run_dir: Path, systems: list[str], preview: bool = False, height: int = 960) -> Path:
    """dancer | system 1 | system 2 ... as one video (side_by_side.mp4), each labelled."""
    run_dir = Path(run_dir)
    score = json.loads((run_dir / "dance_score.json").read_text())
    src = (run_dir / "source.txt").read_text().strip() if (run_dir / "source.txt").exists() else score["source"]
    inputs, labels, rates = ["-i", src], ["dancer"], []
    for s in systems:
        fr = run_dir / s / ("preview" if preview else "frames")
        if fr.exists() and any(fr.glob("frame_*.png")):
            info = fr / "render_info.json"
            r = json.loads(info.read_text())["fps"] if info.exists() else score["fps"]
            inputs += ["-framerate", f"{r:g}", "-start_number", "1", "-i", str(fr / "frame_%04d.png")]
            labels.append(s)
            rates.append(r)
    if len(labels) == 1:
        sys.exit("nothing rendered yet; render first")
    fps = max(rates)

    def graph(with_text: bool) -> str:
        parts = []
        for i, lab in enumerate(labels):
            f = f"[{i}:v]scale=-2:{height},setsar=1,fps={fps}"
            if with_text:
                f += f",drawtext=text='{lab}':x=20:y=20:fontsize=36:fontcolor=black:box=1:boxcolor=white@0.7"
            parts.append(f + f"[v{i}]")
        parts.append("".join(f"[v{i}]" for i in range(len(labels))) + f"hstack=inputs={len(labels)}:shortest=1[out]")
        return ";".join(parts)

    out = run_dir / ("side_by_side_preview.mp4" if preview else "side_by_side.mp4")
    for with_text in (True, False):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph(with_text), "-map", "[out]",
               "-map", "0:a?", "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", str(out)]
        if subprocess.run(cmd).returncode == 0:
            print(f"wrote {out}" + ("" if with_text else " (no labels: this ffmpeg has no drawtext)"))
            return out
    sys.exit("ffmpeg failed")
