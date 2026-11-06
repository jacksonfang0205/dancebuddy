"""Ways for the agent to see what a plan made, beyond the numbers (critics/numbers.py), cheapest first:

  check_arm_geometry  poses the rig (no rendering, a few seconds per dozen frames): where an arm folds (its tightest
                      bend is tighter than the arm is thick) or sinks into the body, per move unit
  look_at_frames      renders a few frames small (Cycles, about a second each) next to the dancer at the same
                      moments, and has a vision model compare them against the dancer and the character's motion
                      spec; returns its notes per frame and the picture
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from dancebuddy import llm

ROOT = Path(__file__).resolve().parents[2]  # the repository
CELL = (180, 320)  # width, height of one picture in the contact sheet


def _blender(script: str, args: list[str]) -> list[str]:
    path = str(ROOT / "dancebuddy" / "blender" / script)
    if importlib.util.find_spec("bpy") is not None and not os.environ.get("BLENDER"):
        return [sys.executable, path, *args]
    exe = os.environ.get("BLENDER") or shutil.which("blender")
    return [exe, "-b", "--factory-startup", "-P", path, "--", *args]


def _subset(curves: dict, frames: list[int]) -> dict:
    return dict(curves, n_frames=len(frames), channels={k: [v[f] for f in frames] for k, v in curves["channels"].items()})


def pick_frames(score: dict, n_frames: int, frames: list[int] | None = None, unit_id: int | None = None,
                most: int = 8) -> list[int]:
    """The frames asked for, or a unit's accents, beats and its start and end (at most `most`)."""
    if frames:
        return sorted({int(f) for f in frames if 0 <= int(f) < n_frames})[:most]
    units = [u for u in score["units"] if unit_id is None or u["id"] == unit_id]
    picks: list[int] = []
    for u in units:
        cand = [u["start"], *u.get("accents", []), *u.get("beats", []), min(u["end"], n_frames - 1)]
        picks += [int(f) for f in cand if 0 <= f < n_frames]
    picks = sorted(set(picks))
    if len(picks) > most:
        idx = np.linspace(0, len(picks) - 1, most).round().astype(int)
        picks = [picks[i] for i in idx]
    return picks


def check_arm_geometry(curves_path: Path, blend: str, score: dict, frames: list[int], work: Path) -> dict:
    """Arm folds and sinking into the body on the given frames (dancebuddy/blender/check_arm_shape.py), per unit."""
    curves = json.loads(Path(curves_path).read_text())
    work.mkdir(parents=True, exist_ok=True)
    sub = work / "geometry_curves.json"
    sub.write_text(json.dumps(_subset(curves, frames)))
    rows_path = work / "geometry.json"
    subprocess.run(_blender("check_arm_shape.py", ["--blend", blend, "--curves", str(sub), "--frames", "0",
                                                   "--out", str(rows_path)]),
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = json.loads(rows_path.read_text())["rows"]
    flagged = []
    for f, r in zip(frames, rows):
        for s in "LR":
            x = r[s]
            probs = []
            if x["fold"] < 1.0:
                probs.append(f"folds (tightest bend {x['fold']:.2f} x its half thickness)")
            if x["inside_pts"] > 40 and x["inside_cm"] > 3:
                probs.append(f"sinks {x['inside_cm']:.0f} cm into the body ({x['inside_pts']} points)")
            if probs:
                unit = next((u["id"] for u in score["units"] if u["start"] <= f <= u["end"]), None)
                flagged.append({"frame": f, "unit_id": unit, "arm": "left" if s == "L" else "right",
                                "problems": probs})
    return {"frames_checked": frames, "problems": flagged,
            "ok": not flagged, "note": "arm geometry on the posed character, no rendering"}


def _dancer_frames(source: str, frames: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(source)
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        out.append(im if ok else np.zeros((CELL[1], CELL[0], 3), np.uint8))
    cap.release()
    return out


def _fit(im: np.ndarray) -> np.ndarray:
    w, h = CELL
    s = min(w / im.shape[1], h / im.shape[0])
    r = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 255, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def contact_sheet(curves_path: Path, blend: str, score: dict, source: str, frames: list[int], work: Path) -> Path:
    """Dancer (top) and character (bottom) at the same frames, labelled; renders the character small."""
    curves = json.loads(Path(curves_path).read_text())
    work.mkdir(parents=True, exist_ok=True)
    sub = work / "look_curves.json"
    sub.write_text(json.dumps(_subset(curves, frames)))
    out_frames = work / "look_frames"
    if out_frames.exists():
        shutil.rmtree(out_frames)
    subprocess.run(_blender("render_curves.py", ["--blend", blend, "--curves", str(sub), "--frames", str(out_frames),
                                                 "--engine", "cycles", "--width", "270", "--height", "480"]),
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    accents = set(score.get("accents", []))
    fps = float(score["fps"])
    top, bottom = [], []
    for i, (f, d) in enumerate(zip(frames, _dancer_frames(source, frames))):
        c = cv2.imread(str(out_frames / f"frame_{i + 1:04d}.png"))
        cell_d, cell_c = _fit(d), _fit(c if c is not None else np.zeros((480, 270, 3), np.uint8))
        label = f"#{f} {f / fps:.1f}s" + (" ACCENT" if f in accents else "")
        cv2.putText(cell_d, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)
        top.append(cell_d)
        bottom.append(cell_c)
    sheet = np.vstack([np.hstack(top), np.hstack(bottom)])
    path = work / "look.png"
    cv2.imwrite(str(path), sheet)
    return path


CRITIC = """You compare an animated plush character (bottom row) with a human dancer (top row) at the same moments.
The character has no legs (leg moves become hops, squashes and leans of its body), no neck and no elbows (its
soft arms bend as one round curve). Its motion spec says:
{rules}

For each column (labelled with its frame number in the top row) judge:
- copy: does the character show the dancer's move at that moment (arm directions and heights, body lean and turn,
  accents as a clear hit)? 2 = clearly, 1 = partly, 0 = not
- problems with the character itself: arms folding or breaking, arms sinking into the body, skin pulled up under a
  raised arm, parts leaving the picture, an unnatural pose for this character
- one concrete fix, in terms of the character (e.g. "raise the left arm higher", "lean the body more to the right")

Answer only with JSON: {{"frames": [{{"frame": int, "copy": 0|1|2, "problems": [str], "fix": str}}],
"overall": str}}"""


def look_at_frames(curves_path: Path, blend: str, score: dict, spec: dict, source: str, frames: list[int],
                   work: Path, model: str | None = None) -> dict:
    sheet = contact_sheet(curves_path, blend, score, source, frames, work)
    rules = "\n".join(f"- {k}: {v}" for k, v in spec.get("body", {}).get("connections", {}).items())
    msg = [{"role": "user", "content": [{"type": "text", "text": CRITIC.format(rules=rules)}, llm.image_block(sheet)]}]
    resp = llm.call(msg, model=model, max_tokens=3000)
    try:
        notes = llm.extract_json(resp["text"])
    except Exception:  # noqa: BLE001 - keep the raw answer for the agent
        notes = {"raw": resp["text"]}
    (work / "look_notes.json").write_text(json.dumps(notes, indent=1))
    return {"frames": frames, "picture": str(sheet), "notes": notes,
            "cost_usd": float((resp.get("usage") or {}).get("cost") or 0.0)}
