"""DanceBuddy command line (MVP: a 3D plush character copies a dance video, an LLM agent directs the copy).

One-time, per character:
  python -m dancebuddy model      MODEL.glb assets/plush          # import an unrigged 3D model
  python blender/rig_soft.py -- --blend assets/plush.blend --out assets/plush_soft   # rig it from its motion spec

Per dance:
  python -m dancebuddy perceive   VIDEO runs/clip1                # pose -> keypoints + dance score (beats, accents)
  python -m dancebuddy copy       runs/clip1                      # plain copy (no LLM), for comparison
  python -m dancebuddy copy-agent runs/clip1                      # the agent: plan, check, look, write moves, revise
  python -m dancebuddy render     runs/clip1 --systems copy agent_copy --engine cycles
  python -m dancebuddy compose    runs/clip1 --systems copy agent_copy   # dancer | copy | agent, side by side
  python -m dancebuddy demo       VIDEO runs/clip1                # all of the per-dance steps, in order
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAR = ROOT / "assets" / "plush_soft"


# ------------------------------------------------------------------ helpers

def blender_cmd(script: str, args: list[str]) -> list[str]:
    """Use the bpy module if it is installed in this Python, else the Blender app."""
    path = str(ROOT / "blender" / script)
    if importlib.util.find_spec("bpy") is not None and not os.environ.get("BLENDER"):
        return [sys.executable, path, *args]
    exe = os.environ.get("BLENDER") or shutil.which("blender")
    mac = "/Applications/Blender.app/Contents/MacOS/Blender"
    if not exe and Path(mac).exists():
        exe = mac
    if not exe:
        sys.exit("Blender not found: install the bpy module (uv sync) or set BLENDER=/path/to/blender")
    return [exe, "-b", "--factory-startup", "-P", path, "--", *args]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd[:4]), "...")
    subprocess.run(cmd, check=True)


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def spec_for(char: Path) -> dict:
    p = Path(str(char).split("_")[0] + "_motion_spec.json")
    return load(p) if p.exists() else {}


# ------------------------------------------------------------------ commands

def cmd_perceive(a):
    from .perception import perceive
    Path(a.run).mkdir(parents=True, exist_ok=True)
    perceive(a.video, a.run, a.max_seconds)
    if Path(a.video).exists():
        (Path(a.run) / "source.txt").write_text(str(Path(a.video).resolve()))


def cmd_model(a):
    args = ["--model", str(Path(a.model).expanduser().resolve()), "--out", str(a.char), "--engine", a.engine]
    if a.yaw:
        args += ["--yaw", str(a.yaw)]
    run(blender_cmd("make_model.py", args))


def cmd_copy(a):
    """Copy the dancer onto the character frame by frame, following its motion spec (retarget.py)."""
    from .perception import load_keypoints
    from .retarget import retarget
    run_dir = Path(a.run)
    rig = load(Path(str(a.char) + "_rig.json"))
    score = run_dir / "dance_score.json"
    beats = load(score)["beats"] if score.exists() else None
    kp = load_keypoints(run_dir / "keypoints.npz")
    if a.plan:  # re-apply a saved plan (e.g. an agent's plan.json) without asking the LLM again
        from .signals import compute_signals
        curves = retarget(kp, rig, signals=compute_signals(kp), beats=beats, plan=load(Path(a.plan)),
                          score=load(score))
    else:
        curves = retarget(kp, rig, beats=beats)
    out = run_dir / a.name
    out.mkdir(exist_ok=True)
    (out / "curves.json").write_text(json.dumps(curves))
    if score.exists():
        from .copy_checks import run_copy_checks
        res = run_copy_checks(curves, load(score), rig, spec_for(a.char))
        (out / "checks.json").write_text(json.dumps(res, indent=1))
    print(f"copy: {curves['n_frames']} frames, {len(curves['channels'])} channels -> {out}")


def cmd_copy_agent(a):
    """The agent decides how the dancer's moves are shown on the character (copy_agent.py)."""
    from .copy_agent import run_copy_agent
    from .perception import load_keypoints
    run_dir = Path(a.run)
    rig = load(Path(str(a.char) + "_rig.json"))
    score = load(run_dir / "dance_score.json")
    source = (run_dir / "source.txt").read_text().strip() if (run_dir / "source.txt").exists() else score.get("source")
    blend = str(a.char) + ".blend"
    run_copy_agent(load_keypoints(run_dir / "keypoints.npz"), score, rig, spec_for(a.char), run_dir / a.name,
                   max_attempts=a.attempts, model=a.model, blend=blend if Path(blend).exists() else None,
                   source=source if source and Path(source).exists() else None, max_looks=a.looks)


def cmd_render(a):
    """Render systems. Default: 30 fps (a 60 fps clip renders every other frame) with the --engine given.
    --preview: a quick untextured look (workbench, half size) into <system>/preview, for checking moves."""
    run_dir = Path(a.run)
    for sysname in a.systems:
        curves_path = run_dir / sysname / "curves.json"
        if not curves_path.exists():
            print(f"skip {sysname}: no {curves_path}")
            continue
        curves = load(curves_path)
        src_fps = float(curves["fps"])
        step = max(1, int(round(src_fps / a.fps))) if a.fps else 1
        frames = run_dir / sysname / ("preview" if a.preview else "frames")
        if frames.exists():
            shutil.rmtree(frames)
        frames.mkdir(parents=True)
        render_curves = curves_path
        if step > 1:  # keep every step-th frame, so the motion keeps its timing at the lower frame rate
            sub = dict(curves, fps=src_fps / step,
                       channels={k: v[::step] for k, v in curves["channels"].items()})
            sub["n_frames"] = len(next(iter(sub["channels"].values())))
            render_curves = frames / "curves_render.json"
            render_curves.write_text(json.dumps(sub))
        engine = "workbench" if a.preview else a.engine
        width, height = (a.width // 2, a.height // 2) if a.preview else (a.width, a.height)
        (frames / "render_info.json").write_text(json.dumps(
            {"fps": src_fps / step, "step": step, "engine": engine, "size": [width, height], "preview": a.preview}))
        args = ["--blend", str(a.char) + ".blend", "--curves", str(render_curves), "--frames", str(frames),
                "--engine", engine, "--width", str(width), "--height", str(height)]
        if a.max_frames:
            args += ["--max-frames", str(a.max_frames)]
        print(f"render {sysname}: {engine} {width}x{height} at {src_fps / step:g} fps -> {frames}")
        run(blender_cmd("render_curves.py", args))


def cmd_compose(a):
    run_dir = Path(a.run)
    score = load(run_dir / "dance_score.json")
    src = (run_dir / "source.txt").read_text().strip() if (run_dir / "source.txt").exists() else score["source"]
    inputs = ["-i", src]
    labels = ["dancer"]
    rates = []
    for s in a.systems:
        fr = run_dir / s / ("preview" if a.preview else "frames")
        if fr.exists() and any(fr.glob("frame_*.png")):
            info = fr / "render_info.json"
            r = load(info)["fps"] if info.exists() else score["fps"]
            inputs += ["-framerate", f"{r:g}", "-start_number", "1", "-i", str(fr / "frame_%04d.png")]
            labels.append(s)
            rates.append(r)
    fps = max(rates) if rates else score["fps"]
    if len(labels) == 1:
        sys.exit("nothing rendered yet; run the render command first")
    h = a.height

    def graph(with_text: bool) -> str:
        parts = []
        for i, lab in enumerate(labels):
            f = f"[{i}:v]scale=-2:{h},setsar=1,fps={fps}"
            if with_text:
                f += f",drawtext=text='{lab}':x=20:y=20:fontsize=36:fontcolor=black:box=1:boxcolor=white@0.7"
            parts.append(f + f"[v{i}]")
        parts.append("".join(f"[v{i}]" for i in range(len(labels))) + f"hstack=inputs={len(labels)}:shortest=1[out]")
        return ";".join(parts)

    out = run_dir / ("side_by_side_preview.mp4" if a.preview else "side_by_side.mp4")
    for with_text in (True, False):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph(with_text), "-map", "[out]",
               "-map", "0:a?", "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", str(out)]
        if subprocess.run(cmd).returncode == 0:
            print(f"wrote {out}" + ("" if with_text else " (no labels: this ffmpeg has no drawtext)"))
            return
    sys.exit("ffmpeg failed")


def cmd_demo(a):
    """Every per-dance step: perceive, plain copy, agent, render both, side-by-side video."""
    cmd_perceive(argparse.Namespace(video=a.video, run=a.run, max_seconds=a.max_seconds))
    cmd_copy(argparse.Namespace(run=a.run, char=a.char, name="copy", plan=None))
    cmd_copy_agent(argparse.Namespace(run=a.run, char=a.char, name="agent_copy", attempts=a.attempts, model=None,
                                      looks=a.looks))
    systems = ["copy", "agent_copy"]
    cmd_render(argparse.Namespace(run=a.run, char=a.char, systems=systems, engine=a.engine, width=a.width,
                                  height=a.height, max_frames=0, fps=30, preview=a.preview))
    cmd_compose(argparse.Namespace(run=a.run, systems=systems, preview=a.preview, height=a.height))


def main(argv=None):
    p = argparse.ArgumentParser(prog="dancebuddy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *args):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        for arg in args:
            s.add_argument(arg)
        return s

    def char_opt(s):
        s.add_argument("--char", type=Path, default=DEFAULT_CHAR, help="rigged character prefix (default assets/plush_soft)")

    def render_opts(s):
        s.add_argument("--engine", default="cycles", choices=["auto", "eevee", "cycles", "workbench"])
        s.add_argument("--width", type=int, default=540)
        s.add_argument("--height", type=int, default=960)

    def agent_opts(s):
        s.add_argument("--attempts", type=int, default=4, help="plans the agent may submit")
        s.add_argument("--looks", type=int, default=4, help="times the agent may look at rendered frames")

    s = add("perceive", cmd_perceive, "video", "run"); s.add_argument("--max-seconds", type=float)
    s = add("model", cmd_model, "model"); s.add_argument("char", type=Path, help="output prefix, e.g. assets/plush")
    s.add_argument("--engine", default="auto")
    s.add_argument("--yaw", type=float, default=0.0, help="rotate the model (degrees) if it does not face the camera")
    s = add("copy", cmd_copy, "run"); char_opt(s); s.add_argument("--name", default="copy")
    s.add_argument("--plan", help="a saved copy plan (e.g. runs/x/agent_copy/plan.json) to apply, no LLM")
    s = add("copy-agent", cmd_copy_agent, "run"); char_opt(s); agent_opts(s)
    s.add_argument("--name", default="agent_copy"); s.add_argument("--model", help="LLM (OpenRouter id)")
    s = add("render", cmd_render, "run"); char_opt(s); render_opts(s)
    s.add_argument("--systems", nargs="+", default=["copy", "agent_copy"]); s.add_argument("--max-frames", type=int, default=0)
    s.add_argument("--fps", type=float, default=30, help="render frame rate (0 = the clip's own; default 30)")
    s.add_argument("--preview", action="store_true", help="quick untextured preview into <system>/preview")
    s = add("compose", cmd_compose, "run"); s.add_argument("--systems", nargs="+", default=["copy", "agent_copy"])
    s.add_argument("--preview", action="store_true", help="put the previews side by side")
    s.add_argument("--height", type=int, default=960)
    s = add("demo", cmd_demo, "video", "run"); char_opt(s); render_opts(s); agent_opts(s)
    s.add_argument("--max-seconds", type=float)
    s.add_argument("--preview", action="store_true", help="quick untextured renders instead of the full ones")

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
