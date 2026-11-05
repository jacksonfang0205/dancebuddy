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
        spec_path = Path(str(a.char).split("_")[0] + "_motion_spec.json")
        res = run_copy_checks(curves, load(score), rig, load(spec_path) if spec_path.exists() else {})
        (out / "checks.json").write_text(json.dumps(res, indent=1))
    print(f"copy: {curves['n_frames']} frames, {len(curves['channels'])} channels -> {out}")


def cmd_render2d(a):
    """The 2D puppet (render2d.py): the same motion as `copy` (optionally with a saved plan), drawn with the
    character's 2D parts (blender/render_parts_2d.py) into <run>/<name>/frames."""
    from .perception import load_keypoints
    from .render2d import Parts, render
    from .retarget import retarget
    run_dir = Path(a.run)
    rig = load(Path(str(a.char) + "_rig.json"))
    score_p = run_dir / "dance_score.json"
    score = load(score_p) if score_p.exists() else None
    kp = load_keypoints(run_dir / "keypoints.npz")
    motion: dict = {}
    if a.plan:
        from .signals import compute_signals
        retarget(kp, rig, signals=compute_signals(kp), beats=score["beats"] if score else None,
                 plan=load(Path(a.plan)), score=score, motion=motion)
    else:
        retarget(kp, rig, beats=score["beats"] if score else None, motion=motion)
    step = max(1, int(round(motion["fps"] / a.fps))) if a.fps else 1
    frames = run_dir / a.name / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    n = render(motion, Parts(a.parts), frames, step=step)
    (frames / "render_info.json").write_text(json.dumps({"fps": motion["fps"] / step, "step": step, "engine": "2d"}))
    print(f"render2d: {n} frames at {motion['fps'] / step:g} fps -> {frames}")


def cmd_copy_agent(a):
    """The agent decides how the dancer's moves are shown on the character (copy_agent.py), then copies."""
    from .copy_agent import run_copy_agent
    from .perception import load_keypoints
    run_dir = Path(a.run)
    rig = load(Path(str(a.char) + "_rig.json"))
    spec_path = Path(str(a.char).split("_")[0] + "_motion_spec.json")
    spec = load(spec_path) if spec_path.exists() else {}
    run_copy_agent(load_keypoints(run_dir / "keypoints.npz"), load(run_dir / "dance_score.json"), rig, spec,
                   run_dir / a.name, max_attempts=a.attempts, model=a.model)


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
            r = load(info)["fps"] if info.exists() else score["fps"]  # older renders: every source frame
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
    s.add_argument("--plan", help="a saved copy plan (e.g. runs/x/agent_copy/plan.json) to apply, no LLM")
    s = add("render2d", cmd_render2d, "run"); char_opt(s); s.add_argument("--name", default="copy2d")
    s.add_argument("--plan", help="a saved copy plan to apply (no LLM)")
    s.add_argument("--parts", default="assets/plush_2d", help="2D parts from blender/render_parts_2d.py")
    s.add_argument("--fps", type=float, default=30)
    s = add("copy-agent", cmd_copy_agent, "run"); char_opt(s); s.add_argument("--name", default="agent_copy")
    s.add_argument("--attempts", type=int, default=4); s.add_argument("--model")
    s = add("render", cmd_render, "run"); char_opt(s); render_opts(s)
    s.add_argument("--systems", nargs="+", default=["agent", "posecopy"]); s.add_argument("--max-frames", type=int, default=0)
    s.add_argument("--fps", type=float, default=30, help="render frame rate (0 = the clip's own; default 30)")
    s.add_argument("--preview", action="store_true", help="quick untextured preview into <system>/preview")
    s = add("compose", cmd_compose, "run"); s.add_argument("--systems", nargs="+", default=["agent", "posecopy"])
    s.add_argument("--preview", action="store_true", help="put the previews side by side")
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
