# DanceBuddy

**An agent that makes a non-humanoid character dance along with a human dancer.** (CS329Z, Fall 2026)

Give it a video of one person dancing and a character whose body is nothing like theirs: a plush toy with no legs,
no neck and soft arms. DanceBuddy copies what that body can copy, translates what it cannot (leg moves, head
moves), and an LLM directs the result. It reads the dance and the character, decides how each move should show
on this body, checks what it made, looks at rendered frames next to the dancer, writes its own moves where the
copy does not read, and revises.

## The agent

```
                         ┌──────────────────────────── planner (LLM) ────────────────────────────┐
 dance video ─► perception ─►  read   get_dance_score · get_dance_signals · get_leg_events        │
                         │            get_character_spec ◄── knowledge (rig, motion spec)        │
                         │            get_plan_format    ◄── actions (knobs, emphasis, own moves) │
                         │    act    submit_plan ─► executor (retarget, legs) ─► critics.numbers  │
                         │    check  check_arm_geometry · look_at_frames ◄── critics.inspect     │
                         │    revise, keep the best attempt                                       │
                         └────────────────────────────────────────────────┬──────────────────────┘
                                                                          ▼
                                             executor.render (Blender) ─► dancer | character video
```

| Component | Folder | What it does |
|---|---|---|
| **Perception** | `dancebuddy/perception/` | Dance video → pose keypoints (MediaPipe), a dance score (beats, accents, move units) and the dancer's continuous signals (arm raise, torso lean / side / twist, head nod / tilt, hips, feet). |
| **Knowledge** | `dancebuddy/knowledge/` | The character: its rig and its **motion spec** (`assets/plush_motion_spec.json`: how this body moves, as qualitative rules learned from reference clips of it dancing), and a short description of its body (a vision model on its rendered views). |
| **Planner** | `dancebuddy/planner/` | The LLM and its loop: system prompt, tool calls, attempts, keeping the best. |
| **Actions** | `dancebuddy/actions/` | What the agent decides. **Plan** (`plan.py`): per move unit how strongly to use the body, arms, groove, head and travel; per leg event how to translate it; emphasis moves on accents. **Own moves** (`moves.py`): code lines such as `arm("both", 3.5, pose="overhead", fist=True)`, `body(2.7, tilt=-12)`, `hit(3.5, "squash")`. A move blends in, holds and blends back. The code is parsed, never run: only these calls with constant arguments. |
| **Executor** | `dancebuddy/executor/` | Carries actions out. **Retarget** (`retarget.py`): the dancer and the plan become the character's bone curves. Arms bend as one round arc that keeps its width and length, stay out of the body and reach above the head; the soft body bends, bulges and turns all the way round. **Legs** (`legs.py`): jumps, kicks, steps, squats and slides become squashes, hops, leans and slides. **Render** (`render.py`): Blender renders, side-by-side video. |
| **Critics** | `dancebuddy/critics/` | How the agent checks itself. **Numbers** (`numbers.py`): accents hit, completeness per unit, in frame, range. **Inspect** (`inspect.py`): the arms' geometry on the posed rig, and frames rendered next to the dancer and reviewed by a vision model. |
| **Tools** | `dancebuddy/tools.py` | The 8 tools the LLM calls, on top of the components above, with the run's state and switches for ablations. |

### Tools

| Tool | Kind | Cost | Returns |
|---|---|---|---|
| `get_dance_score` | read | — | move units (time, label, body regions), accents, beats |
| `get_dance_signals` | read | — | the dancer's signals: meaning, range, what moves most per unit |
| `get_leg_events` | read | — | leg moves found and how each is translated by default |
| `get_character_spec` | read | — | the motion spec |
| `get_plan_format` | read | — | knobs, emphasis moves, the move language and named poses |
| `submit_plan` | act | ~1 s | the plan carried out, numeric checks per unit |
| `check_arm_geometry` | check | ~10 s | arms folding or sinking into the body, on chosen frames |
| `look_at_frames` | check | ~30 s, limited | per frame: does the character show the move (0–2), what looks wrong, a fix |

Every run writes `trace.json` (every call), `summary.json` (attempts, looks, LLM calls, cost), and, per attempt, its
plan, curves, checks and looks.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and `ffmpeg`.
2. `uv sync --extra render` (includes Blender as the `bpy` module), or `uv sync` and
   `export BLENDER=/path/to/Blender` (4.2 or newer).
3. OpenRouter key: `cp .env.example .env` and fill in `OPENROUTER_API_KEY`. Default model `openai/gpt-6-sol`
   (`DANCEBUDDY_MODEL` to change; it must support tools and image input). One run is ~15–25 calls, ~0.3 USD.
4. `uv run pytest` (22 tests, no API calls, no Blender).

## Run the agent

```bash
uv run python -m dancebuddy demo my_dance.mp4 runs/my          # everything below, in order

uv run python -m dancebuddy perceive my_dance.mp4 runs/my       # perception
uv run python -m dancebuddy agent    runs/my                    # the agent -> runs/my/agent
uv run python -m dancebuddy render   runs/my                    # Cycles, 30 fps (--preview: quick look)
uv run python -m dancebuddy compose  runs/my                    # dancer | agent: runs/my/side_by_side.mp4
uv run python -m dancebuddy replay   runs/my runs/my/agent/plan.json   # carry a saved plan out again, no LLM
```

`agent --attempts N --looks N` bound the agent; `--no-looks`, `--no-geometry`, `--no-moves` switch a capability
off. The default character is `assets/plush_soft`; `--char` picks another.

## Characters

```bash
uv run python -m dancebuddy character model   MODEL.glb assets/plush             # import a 3D model
uv run python -m dancebuddy character rig     assets/plush assets/plush_soft     # rig it from its motion spec
uv run python -m dancebuddy character capsule assets/capsule                     # a built-in 3D capsule
uv run python -m dancebuddy character puppet  assets/sample_character.png assets/sample_character   # a picture
uv run python -m dancebuddy character card    assets/plush_soft                  # describe its body
```

The plush rig (`blender/rig_soft.py`): a spine, a hidden shoulder per arm that slides up as the arm rises, one
16-segment bendable bone per arm, fist shape keys (`blender/fist.py`); the model's shape is not changed.

## Evaluation

A separate module, `evaluation/`, compares the agent with baselines and with its own ablations, on the same dances
and the same character, with metrics that read only the character's motion, not how it was made.

```bash
uv run python -m evaluation prepare            # perception on every dance in evaluation/data/dances.json
uv run python -m evaluation run                # every system on every dance (--no-llm: only the free ones)
uv run python -m evaluation metrics --arms     # -> evaluation/results/results.{json,md}
uv run python -m evaluation render --dance ID --systems plain_copy agent      # optional side-by-side video
```

**Data** (`evaluation/data/`): `dances.json` lists the dance videos (not committed: add your own with their path);
`synth` is a built-in synthetic dancer. `characters.json` lists the characters (default: the plush).

**Systems** (`evaluation/baselines/`):

| System | LLM | What it is |
|---|---|---|
| `pose_copy` | no | the dancer's torso angles copied straight onto the character's bones |
| `plain_copy` | no | the DanceBuddy executor with no plan: perception, leg translation, retarget |
| `planner` | yes | an LLM plans primitive moves per unit from a description of the body, without copying the dancer frame by frame (`baselines/planner/`) |
| `agent` | yes | DanceBuddy, every tool |
| `agent_no_looks` | yes | ablation: no looking at rendered frames |
| `agent_no_moves` | yes | ablation: knobs only, no own moves |
| `agent_numbers` | yes | ablation: numeric checks only |

**Metrics** (`evaluation/metrics/`), from the posed character (`blender/measure_motion.py`, no rendering):

| Metric | Meaning |
|---|---|
| `accent_recall` | share of the dancer's accents the character hits (a stop after fast motion, ±0.1 s) |
| `beat_precision` | share of the character's hits on a beat |
| `mean_offset_ms` | how far its hits are from the accents |
| `motion_corr` | correlation of the character's and the dancer's amount of motion over time |
| `arm_follow` | correlation of each hand's position (relative to the shoulders) with the dancer's wrist, both hands, both axes |
| `hands_in_frame` | share of frames with both hands in the picture |
| `arm_folds`, `arm_sinking` | share of arm-frames where an arm folds or sinks into the body (`--arms`) |
| `px_accent_recall` | accent recall from rendered frames' pixels, when they exist |
| `llm_calls`, `cost_usd`, `passed` | what the system spent, and whether its own checks passed |

## Known limitations

- A raised arm pulls a sheet of skin up from the body side: the generated model fuses the arm to the body along
  its inner side.
- The vision review is a model's opinion, not ground truth; the agent's own accent check is fragile.
- The agent's numeric in-frame check covers the body only (the evaluation's `hands_in_frame` covers the hands).

## Repository layout

```
dancebuddy/                 the agent
  perception/  video.py beats.py signals.py
  knowledge/   character.py card.py measure_reference_video.py
  planner/     loop.py
  actions/     plan.py moves.py
  executor/    retarget.py legs.py render.py
  critics/     numbers.py inspect.py
  characters/  puppet.py
  tools.py llm.py __main__.py
evaluation/                 data, baselines, metrics (python -m evaluation)
  data/        dances.json characters.json synth.py
  baselines/   __init__.py (the systems) pose_copy.py planner/ (agent.py primitives.py checks.py examples/)
  metrics/     __init__.py (rhythm, fidelity, integrity, cost) pixels.py
  results/     results.{json,md}; v1/ from the first prototype
blender/                    Blender scripts
  make_model.py rig_soft.py fist.py make_capsule.py make_puppet.py      characters
  render_curves.py pose_utils.py scene_utils.py                          rendering
  check_arm_shape.py check_arm_width.py measure_motion.py                measuring
assets/                     plush (model, rig, motion spec), capsule, characters from pictures
tests/test_pipeline.py
```
