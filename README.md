# DanceBuddy

An agent that makes a **non-humanoid 3D character dance along with a human dancer** (CS329Z, Fall 2026). Given a
single-dancer video and a plush character (no legs, no neck, soft arms), it copies what the body can copy,
translates what it cannot (legs, head), and an LLM agent directs the copy: it reads the dance, decides how each
move is shown on this body, checks the result in numbers, in the arms' geometry and by looking at rendered frames
next to the dancer, writes its own moves where the copy does not read, and revises.

This is the **mid-term MVP** (branch `agent-vision`).

```
dance video ──► perceive ──► keypoints + dance score (beats, accents, move units, signals)
                                  │
character: 3D model ─► rig_soft ─► rigged character + motion spec (how it moves, learned from reference clips)
                                  │
                                  ▼
          ┌──────────────────── copy agent (LLM, tools) ────────────────────┐
          │ read: dance score, signals, leg events, motion spec, plan format │
          │ act:  submit a plan = knobs per unit + leg translation + accent  │
          │       moves + its own moves (code lines: arm / body / hit)       │
          │ check, cheapest first, where it chooses:                         │
          │   numbers (accents, completeness, in frame, range) ~0 s          │
          │   arm geometry on the posed rig (folds, sinking)    ~10 s        │
          │   look: frames rendered next to the dancer, reviewed ~30 s       │
          │ revise; keep the best attempt                                    │
          └──────────────────────────────┬───────────────────────────────────┘
                                         ▼
                     retarget (deterministic) ──► curves.json ──► Blender render ──► dancer | copy | agent
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and `ffmpeg`.
2. `uv sync --extra render` (includes Blender as the `bpy` module), or `uv sync` and
   `export BLENDER=/path/to/Blender` (4.2 or newer).
3. OpenRouter key: `cp .env.example .env` and fill in `OPENROUTER_API_KEY`.
4. `uv run pytest` (no API calls, no Blender, ~15 s).

## Run

```bash
# once per character
uv run python -m dancebuddy model MODEL.glb assets/plush                               # import the 3D model
uv run python blender/rig_soft.py -- --blend assets/plush.blend --out assets/plush_soft  # rig it

# per dance (all in one: uv run python -m dancebuddy demo my_dance.mp4 runs/my)
uv run python -m dancebuddy perceive   my_dance.mp4 runs/my
uv run python -m dancebuddy copy       runs/my                    # plain copy, no LLM (the comparison)
uv run python -m dancebuddy copy-agent runs/my                    # the agent -> runs/my/agent_copy
uv run python -m dancebuddy render     runs/my --systems copy agent_copy           # Cycles, 30 fps
uv run python -m dancebuddy compose    runs/my --systems copy agent_copy           # side_by_side.mp4
```

`render --preview` / `compose --preview` give a quick untextured look. `copy --plan runs/my/agent_copy/plan.json`
re-applies a saved plan without the LLM. `copy-agent --attempts N --looks N` bound the agent.

## The agent (`dancebuddy/copy_agent.py`)

- **Tools to read**: `get_dance_score` (units, accents, beats), `get_dance_signals`, `get_leg_events`,
  `get_character_spec`, `get_plan_format`.
- **Tool to act**: `submit_copy_plan`, a plan with
  - knobs per move unit (`copy_plan.py`): body, arms, groove, head, travel;
  - per leg event: skip or scale its translation;
  - emphasis moves on accents (squash, hop, nod, tilt, twist);
  - **its own moves** (`moves.py`): code lines such as `arm("both", 3.5, pose="overhead", fist=True)`,
    `arm("L", 0.7, raise_=70, forward=140, bend=40)`, `body(2.7, tilt=-12)`, `hit(3.5, "squash")`. A move blends in,
    holds and blends back into the copy. The code is parsed, never run: only these calls with constant arguments.
- **Tools to check** (`inspect_tools.py`), chosen by the agent: the plan's numbers come back with every submit;
  `check_arm_geometry` poses the rig on chosen frames (no rendering); `look_at_frames` renders up to 8 frames small
  (Cycles) under the dancer at the same moments and a vision model notes per frame whether the move shows, what
  looks wrong and a fix. Looks are limited per run.
- It keeps the best attempt and writes `trace.json` (every call), `summary.json` and, per attempt, `plan.json`,
  `curves.json`, `checks.json` and its looks.

## The copy (deterministic, `dancebuddy/retarget.py`, `legs.py`, `signals.py`)

- **Motion spec** (`assets/plush_motion_spec.json`): how this character moves, as qualitative rules learned from
  reference clips of it dancing (pose vocabulary, how parts connect, how human parts translate, status).
- **Arms**: directions from the picture; the dancer's hanging pose is mapped onto the character's own; each arm
  bends as one circular arc that keeps its width and length; the whole arm is kept out of the body (smoothed over
  time; a pose the agent sets may lie on the body); the shoulder socket slides up as the arm rises, so raised hands
  reach above the head; hands ball up into fists for raised, bent arms.
- **Body**: turn (all the way round, from the shoulder and hip widths), side tilt and lean, a soft C/S bend with a
  bulge, rising like a roly-poly toy on its base, a groove on the beat.
- **Legs** (`legs.py`): jumps, kicks, steps, squats, rises and slides become squashes, hops, leans and slides.
- **Rig** (`blender/rig_soft.py`): spine, a hidden sliding shoulder per arm, one 16-segment B-bone per arm, fist
  shape keys (`blender/fist.py`); the model's shape is not changed. `blender/check_arm_shape.py` and
  `check_arm_width.py` measure the arms in a run.
- **Checks** (`copy_checks.py`): accents with a visible hit, completeness per unit, in frame, body range.

## The first prototype (Jackson's code base, kept alongside)

The MVP grew out of the midpoint prototype, which is still here and still runs:

- **Characters**: the built-in 3D capsule (`character`, `blender/make_capsule.py`), any picture as a rigged 2D
  cut-out puppet (`puppet`, `dancebuddy/puppet.py`, `blender/make_puppet.py`), and a VLM character card
  (`card`, `dancebuddy/character.py`). Assets: `capsule*`, `character*`, `sample_character*`, `plush_card.json`,
  `plush_rig.json`, `plush_views/`.
- **Agent** (`agent`, `dancebuddy/agent.py`, `primitives.py`): plans primitive moves on a 4-bone spine from the
  card, compiles them, checks them (`checks.py`) and revises. `apply` compiles a hand-written plan
  (`examples/`).
- **Baseline** (`baseline`, `dancebuddy/baseline.py`): direct pose copy onto the spine, `runs/<x>/posecopy`.
- **Metrics** (`eval`, `checks.py`, `video_metrics.py`): accent recall, beat precision and alignment, offsets,
  limit violations, idle units, jerk, and pixel-based accent recall on rendered frames; `results/`.
- `demo-v1 VIDEO RUN [--image PIC]` runs it end to end; `synth RUN` makes a synthetic dancer (no video needed).

```bash
uv run python -m dancebuddy synth runs/synth
uv run python -m dancebuddy baseline runs/synth                                  # capsule by default
uv run python -m dancebuddy apply runs/synth examples/handwritten_plan_synth.json
uv run python -m dancebuddy eval runs/synth
```

## Models and cost

OpenRouter, default `openai/gpt-6-sol` (`DANCEBUDDY_MODEL` to change; it must support tool calling and image input).
One agent run is about 15-25 calls and about 0.3 USD; exact costs go to `.cache/llm/ledger.jsonl` (not committed).

## Known limitations (MVP)

- A raised arm pulls a sheet of skin up from the body side: the generated model fuses the arm to the body along
  its inner side. Planned: a clean separate arm.
- The accent check is fragile (any jitter counts as a hit); the vision review is the better judge of "does it
  read", but it is a model's opinion.
- The in-frame check covers the body only; wide reaches can push a hand out of the picture.
- Not in the MVP: the agent's memory across runs (in history, commit 37d1617), a pose library from the reference
  clips.

## Repository layout

```
dancebuddy/  perception.py beats.py signals.py synth.py      dance video -> keypoints, dance score, signals
             legs.py retarget.py                             the copy (MVP)
             copy_plan.py moves.py copy_checks.py            the agent's plan, its moves, the checks (MVP)
             copy_agent.py inspect_tools.py llm.py           the agent loop, its geometry and vision tools (MVP)
             agent.py primitives.py checks.py baseline.py    first prototype: primitive agent, checks, baseline
             character.py puppet.py video_metrics.py         first prototype: card, puppets, pixel metrics
             __main__.py                                     command line
blender/     rig_soft.py fist.py make_model.py               character set-up (MVP)
             make_capsule.py make_puppet.py                  first prototype characters
             render_curves.py scene_utils.py                 rendering
             check_arm_shape.py check_arm_width.py           arm measurements
assets/      plush.blend (imported model), plush_soft.blend + plush_soft_rig.json (rigged),
             plush_motion_spec.json (MVP); capsule*, character*, sample_character*, plush_card/rig/views
             (first prototype); pose_landmarker_full.task (MediaPipe, downloaded on first use)
examples/ results/ tools/                            first prototype: hand-written plans, its results, a measuring tool
tests/       test_pipeline.py
```
