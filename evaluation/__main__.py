"""DanceBuddy evaluation: the agent against baselines and its own ablations, on the same dances and character.

  python -m evaluation prepare                         # perception on every dance in evaluation/data/dances.json
  python -m evaluation run     --systems pose_copy plain_copy planner agent agent_no_looks agent_no_moves
  python -m evaluation metrics                         # measure every system's motion, write the results table
  python -m evaluation render  --dance ID --systems plain_copy agent     # optional: a side-by-side video
  python -m evaluation apply   PLAN.json --dance synth                    # the planner baseline with a hand-written plan

Runs go to runs/<dance id>/<system>/ (not committed); results to evaluation/results/results.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
RESULTS = Path(__file__).resolve().parent / "results"


def dances(only: list[str] | None = None) -> list[dict]:
    ds = json.loads((DATA / "dances.json").read_text())["dances"]
    return [d for d in ds if not only or d["id"] in only]


def char_prefix(cid: str | None) -> Path:
    cs = json.loads((DATA / "characters.json").read_text())
    cid = cid or cs["default"]
    c = next((c for c in cs["characters"] if c["id"] == cid), None)
    return ROOT / (c["prefix"] if c else cid)


def run_dir(runs: Path, d: dict) -> Path:
    return runs / d["id"]


def cmd_prepare(a):
    from dancebuddy.perception.video import perceive
    from evaluation.data.synth import make
    for d in dances(a.dance):
        rd = run_dir(a.runs, d)
        if (rd / "keypoints.npz").exists() and not a.force:
            print(f"{d['id']}: ready")
            continue
        rd.mkdir(parents=True, exist_ok=True)
        if d["video"] == "synth":
            make(rd)
            perceive(rd / "dancer.mp4", rd)
            (rd / "source.txt").write_text(str((rd / "dancer.mp4").resolve()))
        else:
            video = Path(d["video"]) if Path(d["video"]).is_absolute() else ROOT / d["video"]
            perceive(video, rd)
            (rd / "source.txt").write_text(str(video.resolve()))
        print(f"{d['id']}: perceived -> {rd}")


def cmd_run(a):
    from evaluation.baselines import LLM_SYSTEMS, run_system
    char = char_prefix(a.char)
    for d in dances(a.dance):
        rd = run_dir(a.runs, d)
        for s in a.systems:
            if (rd / s / "curves.json").exists() and not a.force:
                print(f"{d['id']}/{s}: done")
                continue
            if s in LLM_SYSTEMS and a.no_llm:
                continue
            print(f"{d['id']}/{s} ...")
            run_system(s, rd, char, attempts=a.attempts, looks=a.looks, model=a.model)


def _blender(script: str, args: list[str]) -> None:
    from dancebuddy.executor.render import blender_cmd
    subprocess.run(blender_cmd(script, args), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cmd_metrics(a):
    from dancebuddy.perception.video import load_keypoints
    from evaluation import metrics as M
    char = char_prefix(a.char)
    rows = []
    for d in dances(a.dance):
        rd = run_dir(a.runs, d)
        if not (rd / "dance_score.json").exists():
            continue
        score, kp = json.loads((rd / "dance_score.json").read_text()), load_keypoints(rd / "keypoints.npz")
        for sd in sorted(p for p in rd.iterdir() if (p / "curves.json").exists() and (not a.systems or p.name in a.systems)):
            mpath = sd / "measure.json"
            if not mpath.exists() or a.force:
                _blender("measure_motion.py", ["--blend", str(char) + ".blend", "--curves", str(sd / "curves.json"),
                                               "--out", str(mpath), "--every", str(a.every)])
            measure = json.loads(mpath.read_text())
            shape = None
            if a.arms:
                spath = sd / "arm_shape.json"
                if not spath.exists() or a.force:
                    _blender("check_arm_shape.py", ["--blend", str(char) + ".blend", "--curves",
                                                    str(sd / "curves.json"), "--frames", "48", "--out", str(spath)])
                shape = json.loads(spath.read_text())
            row = {"dance": d["id"], "system": sd.name, **M.rhythm(measure, score), **M.fidelity(measure, kp),
                   **M.integrity(measure, shape), **M.cost(sd)}
            if list((sd / "frames").glob("frame_*.png")):
                from evaluation.metrics.pixels import motion_from_dir, pixel_metrics
                info = sd / "frames" / "render_info.json"
                step = json.loads(info.read_text()).get("step", 1) if info.exists() else 1
                motion = motion_from_dir(sd / "frames")
                row.update(pixel_metrics(M.np.repeat(motion, step), score))
            rows.append(row)
            print(f"{d['id']}/{sd.name}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("dance", "system")))
    if not rows:
        raise SystemExit("nothing to measure: run `python -m evaluation prepare` and `run` first")
    cols = ["dance", "system", "accent_recall", "beat_precision", "mean_offset_ms", "motion_corr", "arm_follow",
            "hands_in_frame", "arm_folds", "arm_sinking", "px_accent_recall", "llm_calls", "cost_usd", "passed"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |" for r in rows]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out} and {out.with_suffix('.md')}")


def cmd_render(a):
    from dancebuddy.executor.render import compose, render_system
    rd = a.runs / a.dance
    for s in a.systems:
        render_system(rd, s, char_prefix(a.char), a.engine, preview=a.preview)
    compose(rd, a.systems, a.preview)


def cmd_apply(a):
    """The planner baseline with a hand-written plan (no LLM): compile, check."""
    from evaluation.baselines.planner.checks import run_checks
    from evaluation.baselines.planner.primitives import compile_plan
    from dancebuddy.knowledge import character
    rd = a.runs / a.dance
    score, card = json.loads((rd / "dance_score.json").read_text()), character.card(char_prefix(a.char))
    plan = json.loads(Path(a.plan).read_text())
    curves = compile_plan(plan, score, card)
    out = rd / a.name
    out.mkdir(parents=True, exist_ok=True)
    res = run_checks(curves, score, card)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    (out / "curves.json").write_text(json.dumps(curves))
    (out / "checks.json").write_text(json.dumps(res, indent=1))
    print(f"{a.name}: passed={res['passed']} -> {out}")


def main(argv=None):
    from evaluation.baselines import SYSTEMS
    p = argparse.ArgumentParser(prog="evaluation", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, systems=False):
        s.add_argument("--runs", type=Path, default=ROOT / "runs", help="where runs go (default runs/)")
        s.add_argument("--char", help="character id from evaluation/data/characters.json (default plush)")
        s.add_argument("--dance", nargs="*", help="dance ids (default: all in dances.json)")
        if systems:
            s.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
        s.add_argument("--force", action="store_true", help="redo what exists")

    s = sub.add_parser("prepare"); s.set_defaults(fn=cmd_prepare); common(s)
    s = sub.add_parser("run"); s.set_defaults(fn=cmd_run); common(s, systems=True)
    s.add_argument("--attempts", type=int, default=4); s.add_argument("--looks", type=int, default=4)
    s.add_argument("--model"); s.add_argument("--no-llm", action="store_true", help="only the systems without an LLM")
    s = sub.add_parser("metrics"); s.set_defaults(fn=cmd_metrics); common(s)
    s.add_argument("--systems", nargs="*"); s.add_argument("--every", type=int, default=2, help="measure every n-th frame")
    s.add_argument("--arms", action="store_true", help="also check arm folds / sinking (slower)")
    s.add_argument("--out", default=str(RESULTS / "results.json"))
    s = sub.add_parser("render"); s.set_defaults(fn=cmd_render)
    s.add_argument("--runs", type=Path, default=ROOT / "runs"); s.add_argument("--char")
    s.add_argument("--dance", required=True); s.add_argument("--systems", nargs="+", required=True)
    s.add_argument("--engine", default="cycles"); s.add_argument("--preview", action="store_true")
    s = sub.add_parser("apply"); s.set_defaults(fn=cmd_apply)
    s.add_argument("plan"); s.add_argument("--runs", type=Path, default=ROOT / "runs"); s.add_argument("--char")
    s.add_argument("--dance", default="synth"); s.add_argument("--name", default="planner_handwritten")
    a = p.parse_args(argv)
    if isinstance(getattr(a, "dance", None), str) and a.cmd not in ("render", "apply"):
        a.dance = [a.dance]
    a.fn(a)


if __name__ == "__main__":
    main()
