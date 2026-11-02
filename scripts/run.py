"""Run one method on one dance and one character.

    python scripts/run.py --video data/videos/clip.mp4 --character doudou --method agent
    python scripts/run.py --pose outputs/clip/pose.npz --character doudou --method direct

Methods: agent (LLM agent), agent-single (no verification loop), agent-no-understanding, direct (B1), rules (B2).
Outputs go to outputs/<clip>/<character>/<method>/: plan.json, motion.json, metrics.json (+ trace for the agent).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancebuddy.characters.spec import CharacterSpec  # noqa: E402
from dancebuddy.eval.metrics import evaluate  # noqa: E402
from dancebuddy.motion.plan import compile_plan  # noqa: E402
from dancebuddy.perception.features import extract_features  # noqa: E402
from dancebuddy.perception.pose_sequence import PoseSequence  # noqa: E402
from dancebuddy.planners.baselines import direct_copy_plan, fixed_rules_plan  # noqa: E402

AGENT_METHODS = {
    "agent": {},
    "agent-single": {"use_verification": False},
    "agent-no-understanding": {"use_understanding": False},
}


def load_pose(args) -> tuple[PoseSequence, str]:
    if args.pose:
        return PoseSequence.load(args.pose), Path(args.pose).parent.name
    clip = Path(args.video).stem
    cache = Path(args.out) / clip / "pose.npz"
    if cache.exists():
        return PoseSequence.load(cache), clip
    from dancebuddy.perception.extract import extract_pose

    seq = extract_pose(args.video)
    cache.parent.mkdir(parents=True, exist_ok=True)
    seq.save(cache)
    return seq, clip


def default_views(spec: CharacterSpec) -> list[Path]:
    view_dir = Path("outputs/views") / spec.name
    return sorted(view_dir.glob(f"{spec.name}_*.png"))


def dry_run_llm(spec, features):
    """Scripted replies (fixed-rules plan) so the full agent loop can be exercised without an API key."""
    from dancebuddy.agent.llm import ScriptedLLM

    plan = fixed_rules_plan(spec, features).to_dict()
    for seg in plan["segments"]:
        seg["intent"] = "dry run"
        for m in seg["mappings"]:
            m.setdefault("offset", 0.0)
    understanding = {
        "body_summary": spec.description,
        "parts_present": spec.parts_present,
        "parts_missing_or_merged": spec.parts_missing,
        "expressive_channels": [{"channel": c, "reads_as": spec.channels[c].description} for c in spec.channels],
        "translation_ideas": [],
    }
    return ScriptedLLM([understanding] + [plan] * 5)


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--pose")
    parser.add_argument("--character", default="doudou")
    parser.add_argument("--method", default="agent", choices=list(AGENT_METHODS) + ["direct", "rules"])
    parser.add_argument("--views", nargs="*", help="character render PNGs for the understanding step")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="use scripted LLM replies instead of the API")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    seq, clip = load_pose(args)
    features = extract_features(seq)
    spec = CharacterSpec.load(args.character)
    out = Path(args.out) / clip / spec.name / args.method
    out.mkdir(parents=True, exist_ok=True)

    if args.method in AGENT_METHODS:
        from dancebuddy.agent.agent import DanceBuddyAgent
        from dancebuddy.agent.llm import ClaudeLLM

        llm = dry_run_llm(spec, features) if args.dry_run else ClaudeLLM()
        agent = DanceBuddyAgent(llm, max_rounds=args.rounds, **AGENT_METHODS[args.method])
        views = [Path(v) for v in args.views] if args.views else default_views(spec)
        result = agent.run(features, spec, views=views)
        result.save(out)
        plan, motion = result.plan, result.motion
    else:
        plan = direct_copy_plan(spec, features) if args.method == "direct" else fixed_rules_plan(spec, features)
        motion = compile_plan(plan, spec, features)
        (out / "plan.json").write_text(json.dumps(plan.to_dict(), indent=1))
        motion.save(out / "motion.json")

    metrics = evaluate(features, motion, spec)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({"output": str(out), **{k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}}, indent=1))


if __name__ == "__main__":
    main()
