"""DanceBuddy command line.

  python -m dancebuddy synth      runs/synth                      # synthetic dancer (no video needed)
  python -m dancebuddy perceive   VIDEO runs/clip1                 # pose -> dance_score.json
  python -m dancebuddy character  assets/capsule                   # build + render the capsule character
  python -m dancebuddy puppet     PICTURE assets/blob              # character picture -> rigged 2D puppet + card
  python -m dancebuddy model      MODEL.glb assets/plush           # unrigged 3D model -> rigged character + card
  python -m dancebuddy card       assets/capsule_rig.json          # VLM character card
  python -m dancebuddy agent      runs/clip1                       # plan -> check -> revise
  python -m dancebuddy baseline   runs/clip1                       # pose-copy baseline
  python -m dancebuddy apply      runs/clip1 plan.json             # compile + check a hand-written plan
  python -m dancebuddy render     runs/clip1 [--systems agent posecopy]
  python -m dancebuddy compose    runs/clip1                       # side-by-side video
  python -m dancebuddy eval       runs/clip1 [runs/clip2 ...]      # metrics table
  python -m dancebuddy demo       VIDEO runs/clip1 [--image PIC]   # everything above, in order
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
DEFAULT_CHAR = ROOT / "assets" / "capsule"


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


def card_path(char: Path) -> Path:
    return Path(str(char) + "_card.json")


def get_card(char: Path) -> dict:
    p = card_path(char)
    if p.exists():
        return load(p)
    rig = Path(str(char) + "_rig.json")
    if not rig.exists():
        sys.exit(f"no character at {char}; run: python -m dancebuddy character {char}")
    from .character import build_card
    return build_card(rig, p, use_vlm=False)


# ------------------------------------------------------------------ commands

def cmd_synth(a):
    from .perception import perceive
    from .synth import make
    out = make(a.run)
    perceive(out / "dancer.mp4", out)
    (Path(a.run) / "source.txt").write_text(str(Path(a.run) / "dancer.mp4"))


def cmd_perceive(a):
    from .perception import perceive
    Path(a.run).mkdir(parents=True, exist_ok=True)
    perceive(a.video, a.run, a.max_seconds)
    if Path(a.video).exists():
        (Path(a.run) / "source.txt").write_text(str(Path(a.video).resolve()))


def cmd_character(a):
    run(blender_cmd("make_capsule.py", ["--out", str(a.char), "--engine", a.engine]))


def cmd_puppet(a):
    from .character import build_card
    from .puppet import cut_out
    cut_out(a.image, a.char)
    run(blender_cmd("make_puppet.py", ["--out", str(a.char), "--engine", a.engine]))
    build_card(Path(str(a.char) + "_rig.json"), card_path(Path(a.char)), use_vlm=not a.no_vlm)


def cmd_model(a):
    from .character import build_card
    args = ["--model", str(Path(a.model).expanduser().resolve()), "--out", str(a.char), "--engine", a.engine]
    if a.yaw:
        args += ["--yaw", str(a.yaw)]
    run(blender_cmd("make_model.py", args))
    build_card(Path(str(a.char) + "_rig.json"), card_path(Path(a.char)), use_vlm=not a.no_vlm)


def cmd_card(a):
    from .character import build_card
    rig = Path(a.rig)
    out = Path(str(rig).replace("_rig.json", "_card.json"))
    card = build_card(rig, out, use_vlm=not a.no_vlm)
    print(json.dumps(card.get("description"), indent=1))


def cmd_agent(a):
    from .agent import run_agent
    run_dir = Path(a.run)
    score, card = load(run_dir / "dance_score.json"), get_card(Path(a.char))
    name = a.name or ("agent" + ("_nocard" if a.no_card else "") + ("_norevise" if a.no_revise else ""))
    run_agent(score, card, run_dir / name, max_attempts=1 if a.no_revise else a.attempts,
              use_card_description=not a.no_card, model=a.model)


def cmd_baseline(a):
    from .baseline import pose_copy
    from .checks import run_checks
    from .perception import load_keypoints
    run_dir = Path(a.run)
    kp, card, score = load_keypoints(run_dir / "keypoints.npz"), get_card(Path(a.char)), load(run_dir / "dance_score.json")
    curves = pose_copy(kp, card)
    out = run_dir / "posecopy"
    out.mkdir(exist_ok=True)
    (out / "curves.json").write_text(json.dumps(curves))
    (out / "checks.json").write_text(json.dumps(run_checks(curves, score, card), indent=1))
    print(f"pose-copy baseline -> {out}")


def cmd_copy(a):
    """Copy the dancer onto the character frame by frame, following its motion spec (retarget.py)."""
    from .perception import load_keypoints
    from .retarget import retarget
    run_dir = Path(a.run)
    rig = load(Path(str(a.char) + "_rig.json"))
    score = run_dir / "dance_score.json"
    beats = load(score)["beats"] if score.exists() else None
    curves = retarget(load_keypoints(run_dir / "keypoints.npz"), rig, beats=beats)
    out = run_dir / a.name
    out.mkdir(exist_ok=True)
    (out / "curves.json").write_text(json.dumps(curves))
    print(f"copy: {curves['n_frames']} frames, {len(curves['channels'])} channels -> {out}")


def cmd_apply(a):
    """Compile a plan you wrote yourself (no LLM) and check it."""
    from .checks import run_checks
    from .primitives import compile_plan
    run_dir = Path(a.run)
    score, card, plan = load(run_dir / "dance_score.json"), get_card(Path(a.char)), load(Path(a.plan))
    curves = compile_plan(plan, score, card)
    out = run_dir / a.name
    out.mkdir(exist_ok=True)
    res = run_checks(curves, score, card)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    (out / "curves.json").write_text(json.dumps(curves))
    (out / "checks.json").write_text(json.dumps(res, indent=1))
    print(f"{a.name}: passed={res['passed']} -> {out}")


def cmd_render(a):
    run_dir = Path(a.run)
    for sysname in a.systems:
        curves = run_dir / sysname / "curves.json"
        if not curves.exists():
            print(f"skip {sysname}: no {curves}")
            continue
        frames = run_dir / sysname / "frames"
        if frames.exists():
            shutil.rmtree(frames)
        args = ["--blend", str(a.char) + ".blend", "--curves", str(curves), "--frames", str(frames),
                "--engine", a.engine, "--width", str(a.width), "--height", str(a.height)]
        if a.max_frames:
            args += ["--max-frames", str(a.max_frames)]
        run(blender_cmd("render_curves.py", args))


def cmd_compose(a):
    run_dir = Path(a.run)
    score = load(run_dir / "dance_score.json")
    fps = score["fps"]
    src = (run_dir / "source.txt").read_text().strip() if (run_dir / "source.txt").exists() else score["source"]
    inputs = ["-i", src]
    labels = ["dancer"]
    for s in a.systems:
        fr = run_dir / s / "frames"
        if fr.exists() and any(fr.glob("frame_*.png")):
            inputs += ["-framerate", str(fps), "-start_number", "1", "-i", str(fr / "frame_%04d.png")]
            labels.append(s)
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

    out = run_dir / "side_by_side.mp4"
    for with_text in (True, False):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph(with_text), "-map", "[out]", "-map", "0:a?",
               "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", str(out)]
        if subprocess.run(cmd).returncode == 0:
            print(f"wrote {out}" + ("" if with_text else " (no labels: this ffmpeg has no drawtext)"))
            return
    sys.exit("ffmpeg failed")


def cmd_eval(a):
    from .checks import metrics
    rows = []
    for rd in a.runs:
        rd = Path(rd)
        score = load(rd / "dance_score.json")
        card = get_card(Path(a.char))
        for sysdir in sorted(p for p in rd.iterdir() if (p / "curves.json").exists()):
            m = metrics(load(sysdir / "curves.json"), score, card)
            extra = {}
            if list((sysdir / "frames").glob("frame_*.png")):  # rendered: add the pixel-based metric
                from .video_metrics import motion_from_dir, pixel_metrics
                extra.update(pixel_metrics(motion_from_dir(sysdir / "frames"), score))
            if (sysdir / "summary.json").exists():
                s = load(sysdir / "summary.json")
                extra["attempts"] = s["attempts"]
            rows.append({"run": rd.name, "system": sysdir.name, **m, **extra})
    if not rows:
        sys.exit("no results found")
    cols = ["run", "system", "accent_recall", "beat_precision", "beat_alignment", "mean_offset_ms",
            "limit_violations", "idle_units", "jerk", "px_accent_recall", "px_beat_precision", "all_checks_passed",
            "attempts"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    table = "\n".join(lines)
    print(table)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text(table + "\n")
    print(f"\nwrote {out} and {out.with_suffix('.md')}")


def cmd_demo(a):
    char = Path(a.char)
    if a.image:
        if a.char == DEFAULT_CHAR:
            char = ROOT / "assets" / Path(a.image).stem
        cmd_puppet(argparse.Namespace(image=a.image, char=char, engine=a.engine, no_vlm=False))
    elif not Path(str(char) + ".blend").exists():
        cmd_character(argparse.Namespace(char=char, engine=a.engine))
    if not card_path(char).exists():
        cmd_card(argparse.Namespace(rig=str(char) + "_rig.json", no_vlm=False))
    if a.video == "synth":
        cmd_synth(argparse.Namespace(run=a.run))
    else:
        cmd_perceive(argparse.Namespace(video=a.video, run=a.run, max_seconds=a.max_seconds))
    cmd_agent(argparse.Namespace(run=a.run, char=char, name=None, no_card=False, no_revise=False,
                                 attempts=a.attempts, model=None))
    cmd_baseline(argparse.Namespace(run=a.run, char=char))
    cmd_render(argparse.Namespace(run=a.run, char=char, systems=["agent", "posecopy"], engine=a.engine,
                                  width=a.width, height=a.height, max_frames=0))
    cmd_compose(argparse.Namespace(run=a.run, systems=["agent", "posecopy"], height=a.height))
    cmd_eval(argparse.Namespace(runs=[a.run], char=char, out=str(Path(a.run) / "results.json")))


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
        s.add_argument("--char", type=Path, default=DEFAULT_CHAR, help="character prefix (default assets/capsule)")

    def render_opts(s):
        s.add_argument("--engine", default="auto", choices=["auto", "eevee", "cycles", "workbench"])
        s.add_argument("--width", type=int, default=540)
        s.add_argument("--height", type=int, default=960)

    add("synth", cmd_synth, "run")
    s = add("perceive", cmd_perceive, "video", "run"); s.add_argument("--max-seconds", type=float)
    s = sub.add_parser("character"); s.set_defaults(fn=cmd_character)
    s.add_argument("char", type=Path, nargs="?", default=DEFAULT_CHAR); s.add_argument("--engine", default="auto")
    s = add("puppet", cmd_puppet, "image"); s.add_argument("char", type=Path, help="output prefix, e.g. assets/blob")
    s.add_argument("--engine", default="auto"); s.add_argument("--no-vlm", action="store_true")
    s = add("model", cmd_model, "model"); s.add_argument("char", type=Path, help="output prefix, e.g. assets/plush")
    s.add_argument("--engine", default="auto"); s.add_argument("--no-vlm", action="store_true")
    s.add_argument("--yaw", type=float, default=0.0, help="rotate the model (degrees) if it does not face the camera")
    s = add("card", cmd_card, "rig"); s.add_argument("--no-vlm", action="store_true")
    s = add("agent", cmd_agent, "run"); char_opt(s)
    s.add_argument("--name"); s.add_argument("--attempts", type=int, default=4); s.add_argument("--model")
    s.add_argument("--no-card", action="store_true", help="ablation: hide the VLM body description")
    s.add_argument("--no-revise", action="store_true", help="ablation: one attempt, no revision")
    s = add("baseline", cmd_baseline, "run"); char_opt(s)
    s = add("apply", cmd_apply, "run", "plan"); char_opt(s); s.add_argument("--name", default="handwritten")
    s = add("copy", cmd_copy, "run"); char_opt(s); s.add_argument("--name", default="copy")
    s = add("render", cmd_render, "run"); char_opt(s); render_opts(s)
    s.add_argument("--systems", nargs="+", default=["agent", "posecopy"]); s.add_argument("--max-frames", type=int, default=0)
    s = add("compose", cmd_compose, "run"); s.add_argument("--systems", nargs="+", default=["agent", "posecopy"])
    s.add_argument("--height", type=int, default=960)
    s = sub.add_parser("eval"); s.set_defaults(fn=cmd_eval); s.add_argument("runs", nargs="+"); char_opt(s)
    s.add_argument("--out", default=str(ROOT / "results" / "results.json"))
    s = add("demo", cmd_demo, "video", "run"); char_opt(s); render_opts(s)
    s.add_argument("--max-seconds", type=float); s.add_argument("--attempts", type=int, default=4)
    s.add_argument("--image", help="character picture (PNG with transparency, or on a plain background)")

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
