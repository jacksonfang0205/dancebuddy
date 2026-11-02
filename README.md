# DanceBuddy

An agent that teaches non-humanoid 3D cartoon characters to dance along with a human dancer (Stanford CS329Z project).

Given a single-dancer video and a rigged character, DanceBuddy extracts the dancer's motion, decides how each move should
be expressed on the character's body (for example, footwork becomes hops for a character without legs), animates the
character in Blender, checks the result against the dancer's rhythm, and revises its plan when checks fail.

## How it works

```
dance video ──► Perception ──► dance features ─┐
                                               ├─► Planner (LLM) ──► motion plan (JSON)
character ────► Character understanding (LLM) ─┘          ▲                 │
                                                          │            Compiler (code)
                                                   feedback text            │
                                                          │        per-frame channel values
                                                      Verifier ◄────────────┤
                                                                            ▼
                                                             Blender render ──► composite video
```

The LLM decides *what* should move and how strongly; code computes exact timing and values.

| Module | Role |
|---|---|
| `dancebuddy/perception/` | MediaPipe pose extraction; interpretable signals (arm raise, torso lean, hip bounce, foot lift, …); per-part motion energy; **accents** (moments where a move hits or holds); steps; segmentation by dominant body part |
| `dancebuddy/characters/specs/` | Each character's control interface: parts present/missing and controllable **channels** with ranges, units and rig bindings |
| `dancebuddy/motion/plan.py` | Motion plan format and compiler. A plan maps dancer signals to character channels per segment, in one of three modes: `copy`, `follow` (shape rescaled onto another part) or `pulse` (a bump on every event, e.g. a hop on every step) |
| `dancebuddy/agent/` | The agent: character understanding, planning, verification-driven revision (Claude API, structured outputs) |
| `dancebuddy/planners/baselines.py` | **B1 direct copy** (anatomical one-to-one copy; motion with no counterpart is dropped) and **B2 fixed rules** (direct copy plus one fixed translation per missing body part) |
| `dancebuddy/eval/metrics.py` | Code-based metrics: accent coverage / precision, synchronization error, activity correlation, idle-while-dancer-moves, jitter, clip rate |
| `dancebuddy/render/` | Procedural characters in Blender (`bpy`), keyframing, transparent rendering |

All methods share perception, character, compiler, renderer and metrics; only the plan author differs.

## Setup

Python 3.11+ (tested on 3.13, Linux). Blender is used as a Python module (`bpy`), no Blender install needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh        # MediaPipe pose model
export ANTHROPIC_API_KEY=...           # only needed for the agent methods
```

On a headless Linux machine MediaPipe also needs EGL: `sudo apt-get install libegl1 libgles2`.

## Run

Put dance clips in `data/videos/` (not committed). Then:

```bash
# 1. Render the character from a few angles; the agent looks at these in its understanding step
python scripts/preview_character.py doudou --out outputs/views/doudou --angles 0,40,90

# 2. Run a method: agent | agent-single | agent-no-understanding | direct | rules
python scripts/run.py --video data/videos/clip.mp4 --character doudou --method agent
python scripts/run.py --video data/videos/clip.mp4 --character doudou --method direct

# 3. Render the result beside the dancer
python scripts/render.py outputs/clip/doudou/agent --video data/videos/clip.mp4 --label "DanceBuddy agent"
```

Each run writes `outputs/<clip>/<character>/<method>/` with `plan.json`, `motion.json` and `metrics.json`; agent runs
also write `understanding.json`, `verification.json` and `trace.json` (every round's plan and verifier report).

`--dry-run` replaces the API with scripted replies, to exercise the full agent loop without a key.

Tests: `python -m pytest tests`

## Characters

| Character | Body | Status |
|---|---|---|
| `doudou` | Plush potato body (head and torso merged), curly hair, two fluffy arms, **no legs** | Spec + procedural model |
| `capsule` | Rigid pill body (no neck or waist), arms without elbows, tiny legs | Spec only; model to do |
| `humanoid` | Human proportions; control character | Spec only; model to do |

## Status (midpoint)

- Done: perception, character specs, plan compiler, baselines B1/B2, metrics, agent loop (understand → plan → compile →
  verify → revise), Doudou model, rendering and side-by-side compositing, tests.
- Next: models for `capsule` and `humanoid`; dance clip set; full evaluation matrix; human pairwise study; LLM-as-judge
  calibrated against human ratings; a pixel-based rhythm metric for the diffusion baseline (Animate-X).
