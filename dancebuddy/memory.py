"""The agent's memory for one character: what it learned from its own runs, kept between runs.

  lessons    short notes the agent wrote at the end of a run ("works", "fails" or "tip"), with how often they were
             confirmed; read at the start of the next run
  runs       what each run did (checks, looks, the moves of its best attempt), recorded automatically
  proposals  changes to the character's motion spec the agent suggests from what it saw; a person approves or
             rejects them (`python -m dancebuddy memory <char> --approve N`). Approved ones are added to the spec's
             "learned" rules, which every later run reads with the spec. The agent never edits the spec itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

KINDS = ("works", "fails", "tip")
MAX_LESSONS = 40


def path_for(char: str | Path) -> Path:
    return Path(str(char).split("_")[0] + "_memory.json")


def load(p: Path) -> dict:
    if p.exists():
        return json.loads(p.read_text())
    return {"lessons": [], "runs": [], "proposals": []}


def save(p: Path, mem: dict) -> None:
    mem["lessons"] = sorted(mem["lessons"], key=lambda x: (-x["count"], -x["last"]))[:MAX_LESSONS]
    mem["runs"] = mem["runs"][-20:]
    p.write_text(json.dumps(mem, indent=1, ensure_ascii=False))


def view(mem: dict) -> dict:
    """What the agent reads at the start of a run."""
    return {"lessons": [{k: x[k] for k in ("kind", "text", "count")} for x in mem["lessons"]],
            "recent_runs": [{k: r.get(k) for k in ("run", "passed", "accents", "moves_written", "reviewer")}
                            for r in mem["runs"][-3:]],
            "pending_spec_proposals": [p["change"] for p in mem["proposals"] if p["status"] == "pending"]}


def remember(mem: dict, lessons: list[dict], proposal: dict | None, run: str) -> dict:
    """Add the agent's lessons (merged with the same text seen before) and a spec proposal."""
    now = time.time()
    added = 0
    for x in lessons or []:
        kind, text = str(x.get("kind", "tip")), str(x.get("text", "")).strip()
        if kind not in KINDS or not text:
            continue
        same = next((y for y in mem["lessons"] if y["text"].lower() == text.lower()), None)
        if same:
            same["count"] += 1
            same["last"] = now
            same["runs"] = sorted(set(same["runs"]) | {run})
        else:
            mem["lessons"].append({"kind": kind, "text": text[:300], "count": 1, "last": now, "runs": [run]})
            added += 1
    if proposal and str(proposal.get("change", "")).strip():
        mem["proposals"].append({"change": str(proposal["change"])[:400], "why": str(proposal.get("why", ""))[:400],
                                 "run": run, "status": "pending"})
    return {"lessons_added": added, "lessons_total": len(mem["lessons"]),
            "proposal": "recorded for a person to review" if proposal else None}


def decide(mem: dict, spec_path: Path, index: int, approve: bool) -> str:
    """A person approves (into the spec's learned rules) or rejects a pending proposal."""
    pending = [p for p in mem["proposals"] if p["status"] == "pending"]
    if not 0 <= index < len(pending):
        raise IndexError(f"no pending proposal {index}; there are {len(pending)}")
    p = pending[index]
    p["status"] = "approved" if approve else "rejected"
    if approve:
        spec = json.loads(spec_path.read_text())
        spec.setdefault("learned", []).append(p["change"])
        spec_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    return p["status"]
