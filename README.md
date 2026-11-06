# DanceBuddy

**An agent that makes a non-humanoid character dance along with a human dancer.** (CS329Z, Fall 2026)

Give it a video of one person dancing and a character whose body is nothing like theirs: a plush toy with no legs,
no neck and soft arms. DanceBuddy copies what that body can copy, translates what it cannot (leg moves, head
moves), and an LLM directs the result. It reads the dance and the character, decides how each move should show
on this body, checks what it made, looks at rendered frames next to the dancer, writes its own moves where the
copy does not read, and revises. The output is a video: the dancer and the character, side by side.

**Contents:** [1 Requirements](#1-requirements) · [2 Install](#2-install) · [3 API key](#3-api-key) ·
[4 Try it in 10 minutes](#4-try-it-in-10-minutes-no-video-needed) · [5 Run on your own dance](#5-run-the-agent-on-your-own-dance-video) ·
[6 Reproduce the evaluation](#6-reproduce-the-evaluation) · [7 How the agent works](#7-how-the-agent-works) ·
[8 Evaluation design](#8-evaluation-design) · [9 Characters](#9-characters) · [10 Troubleshooting](#10-troubleshooting) ·
[11 Repository layout](#11-repository-layout)

## 1. Requirements

- macOS or Linux (Windows: use WSL). No GPU needed: everything, including rendering, runs on the CPU.
- About 2 GB of disk (Python packages including Blender as a module) plus the repository (~100 MB of character assets).
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.13 for you) and `ffmpeg`.
- For the agent only: an [OpenRouter](https://openrouter.ai) API key with a few dollars of credit (one agent run on a
  20-second dance costs about 0.1–0.4 USD). The tests, the baselines without an LLM and recomputing the reported
  results need no key.

## 2. Install

```bash
git clone https://github.com/tianrunyan-tech/dancebuddy.git
cd dancebuddy

# uv and ffmpeg, if you do not have them
curl -LsSf https://astral.sh/uv/install.sh | sh        # or: brew install uv
brew install ffmpeg                                     # macOS;  Ubuntu: sudo apt install ffmpeg

uv sync --extra render      # Python 3.13 + all dependencies, including Blender as the bpy module (~400 MB download)
```

If you already have the Blender app (4.2 or newer) and want to skip the bpy download: `uv sync`, then
`export BLENDER=/path/to/blender` (macOS: `/Applications/Blender.app/Contents/MacOS/Blender`).

Check the set-up and run the tests (~20 s, no key, no network):

```bash
uv run python -m dancebuddy check     # lists anything missing and how to fix it
uv run pytest                          # 22 tests
```

The first `perceive` downloads the MediaPipe pose model (~9 MB) into `assets/`.

## 3. API key

The agent and the `planner` baseline call LLMs through [OpenRouter](https://openrouter.ai) (one key for many models).

```bash
cp .env.example .env          # then edit .env: OPENROUTER_API_KEY=sk-or-v1-...
uv run python -m dancebuddy check --llm      # makes one tiny call to confirm the key works
```

- Create the key at <https://openrouter.ai/keys>. `.env` is git-ignored: never commit it.
- Default model: `openai/gpt-6-sol`. To use another, set `DANCEBUDDY_MODEL=<openrouter model id>` in `.env` (it must
  support tool calling and image input; the vision review sends rendered frames).
- Every call is cached in `.cache/llm/` (re-running an unchanged step is free) and its exact cost is logged in
  `.cache/llm/ledger.jsonl`; each agent run also writes its calls and cost to `summary.json`.

## 4. Try it in 10 minutes (no video needed)

The repository has a built-in synthetic dancer, so you can see the whole system work before using a real video.

```bash
uv run python -m evaluation prepare                 # makes runs/synth: a 12 s synthetic dance + its perception
uv run python -m dancebuddy agent  runs/synth       # the agent (needs the key; ~3 min, ~0.1-0.3 USD)
uv run python -m dancebuddy render runs/synth --preview      # quick untextured render (~2 min)
uv run python -m dancebuddy compose runs/synth --preview     # -> runs/synth/side_by_side_preview.mp4
```

Without a key, run the free systems instead and compare them:
`uv run python -m evaluation run --no-llm && uv run python -m evaluation metrics` (about a minute; prints a table).

What the agent leaves in `runs/synth/agent/`: `summary.json` (attempts, whether its checks passed, LLM calls, cost),
`trace.json` (every message and tool call), the best `plan.json` / `curves.json` / `checks.json`, and an
`attempt_N/` folder per plan with its looks (`look_N/look.png`: the dancer above, the character below, and the
reviewer's notes).

## 5. Run the agent on your own dance video

Use a video of **one person, whole body in view**, filmed from the front, 10–30 seconds (any phone video works).

```bash
uv run python -m dancebuddy perceive my_dance.mp4 runs/my      # pose, beats, accents, move units (~1 min)
uv run python -m dancebuddy agent    runs/my                    # the agent -> runs/my/agent (~3-8 min)
uv run python -m dancebuddy render   runs/my                    # full render, Cycles, 30 fps (~3 s a frame on a laptop CPU)
uv run python -m dancebuddy compose  runs/my                    # -> runs/my/side_by_side.mp4
# or all four at once:
uv run python -m dancebuddy demo my_dance.mp4 runs/my
```

- `render --preview` / `compose --preview`: a quick untextured look first.
- `agent --attempts N --looks N` bound the agent (defaults 4 and 4); `--no-looks`, `--no-geometry`, `--no-moves`
  switch a capability off.
- `replay runs/my runs/my/agent/plan.json --name again`: carry a saved plan out again without the LLM.
- `--char assets/<name>` uses another rigged character (default `assets/plush_soft`).

## 6. Reproduce the evaluation

The evaluation compares the agent with baselines and with its own ablations on the same dances and the same
character (section 8). There are two ways to reproduce it.

**A. Recompute the reported table exactly (free, ~5 min, no key, no videos).** The runs behind the report are in
the repository (`evaluation/results/paper/runs/`; until the final evaluation is exported the folder holds only a
note): for each dance its keypoints and dance score, for each system its
curves, plan, checks, summary and LLM trace; no videos. The metrics are computed again from them:

```bash
uv run python -m evaluation metrics --runs evaluation/results/paper/runs --arms --out evaluation/results/paper/recomputed.json
diff evaluation/results/paper/results.md evaluation/results/paper/recomputed.md      # no difference expected
```

**B. Run everything again from the videos (needs the videos and a key; LLM results vary from run to run).**

1. Get the dance videos listed in `evaluation/data/dances.json` (see that file for where each one comes from) and
   put them at the paths it gives.
2. Run:

```bash
uv run python -m evaluation prepare              # perception on every dance (~1 min each)
uv run python -m evaluation run                  # every system on every dance (LLM systems: ~5 min and ~0.3 USD each)
uv run python -m evaluation metrics --arms       # -> evaluation/results/results.{json,md}
uv run python -m evaluation render --dance <id> --systems plain_copy agent   # optional: side-by-side videos
```

`run --systems agent agent_no_looks` limits the systems, `--dance <id>` the dances, `--no-llm` keeps to the free
ones; finished runs are skipped unless `--force`. The LLM systems are not deterministic, so B gives numbers close
to, not identical with, the reported ones; A gives the identical numbers.

To share a new evaluation the same way: `uv run python -m evaluation export --name <name>` copies the runs (without
videos or rendered frames) into `evaluation/results/<name>/`.

## 7. How the agent works

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
| **Knowledge** | `dancebuddy/knowledge/` | The character: its rig and its **motion spec** (`assets/plush_motion_spec.json`: how this body moves, as qualitative rules learned from reference clips of it dancing), and a short description of its body. |
| **Planner** | `dancebuddy/planner/` | The LLM and its loop: system prompt, tool calls, attempts, keeping the best. |
| **Actions** | `dancebuddy/actions/` | What the agent decides. **Plan** (`plan.py`): per move unit how strongly to use the body, arms, groove, head and travel; per leg event how to translate it; emphasis moves on accents. **Own moves** (`moves.py`): code lines such as `arm("both", 3.5, pose="overhead", fist=True)`, `body(2.7, tilt=-12)`, `hit(3.5, "squash")`; a move blends in, holds and blends back. The code is parsed, never run: only these calls with constant arguments. |
| **Executor** | `dancebuddy/executor/` | **Retarget** (`retarget.py`): the dancer and the plan become the character's bone curves; arms bend as one round arc that keeps its width and length, stay out of the body and reach above the head; the soft body bends, bulges and turns all the way round. **Legs** (`legs.py`): jumps, kicks, steps, squats and slides become squashes, hops, leans and slides. **Render** (`render.py`): Blender renders, side-by-side video. |
| **Critics** | `dancebuddy/critics/` | **Numbers** (`numbers.py`): accents hit, completeness per unit, in frame, range. **Inspect** (`inspect.py`): the arms' geometry on the posed rig, and frames rendered next to the dancer and reviewed by a vision model. |
| **Tools** | `dancebuddy/tools.py` | The 8 tools the LLM calls (below), with the run's state and the ablation switches. |

| Tool | Kind | Time | Returns |
|---|---|---|---|
| `get_dance_score` | read | — | move units (time, label, body regions), accents, beats |
| `get_dance_signals` | read | — | the dancer's signals: meaning, range, what moves most per unit |
| `get_leg_events` | read | — | leg moves found and how each is translated by default |
| `get_character_spec` | read | — | the motion spec |
| `get_plan_format` | read | — | knobs, emphasis moves, the move language and named poses |
| `submit_plan` | act | ~1 s | the plan carried out, numeric checks per unit |
| `check_arm_geometry` | check | ~10 s | arms folding or sinking into the body, on chosen frames |
| `look_at_frames` | check | ~30 s, limited | per frame: does the character show the move (0–2), what looks wrong, a fix |

## 8. Evaluation design

**Data** (`evaluation/data/`): `dances.json` lists the dances (videos are not committed; the file says where each
comes from; `synth` is built in). `characters.json` lists the characters (default: the plush).

**Systems** (`evaluation/baselines/`), all on the same dance and character:

| System | LLM | What it is |
|---|---|---|
| `pose_copy` | no | the dancer's torso angles copied straight onto the character's bones |
| `plain_copy` | no | the DanceBuddy executor with no plan: perception, leg translation, retarget |
| `planner` | yes | an LLM plans primitive moves per unit from a description of the body, without copying the dancer frame by frame (`baselines/planner/`) |
| `agent` | yes | DanceBuddy, every tool |
| `agent_no_looks` | yes | ablation: no looking at rendered frames |
| `agent_no_moves` | yes | ablation: knobs only, no own moves |
| `agent_numbers` | yes | ablation: numeric checks only |

**Metrics** (`evaluation/metrics/`), the same for every system: they read only the character's motion, posed in
Blender without rendering (`blender/measure_motion.py`), the dancer's keypoints and dance score.

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

## 9. Characters

The default character is the plush (`assets/plush_soft`). To make others:

```bash
uv run python -m dancebuddy character model   MODEL.glb assets/plush             # import a 3D model
uv run python -m dancebuddy character rig     assets/plush assets/plush_soft     # rig it from its motion spec
uv run python -m dancebuddy character capsule assets/capsule                     # a built-in 3D capsule
uv run python -m dancebuddy character puppet  assets/sample_character.png assets/sample_character   # a picture
uv run python -m dancebuddy character card    assets/plush_soft                  # describe its body
```

The plush rig (`blender/rig_soft.py`): a spine, a hidden shoulder per arm that slides up as the arm rises, one
16-segment bendable bone per arm, fist shape keys (`blender/fist.py`); the model's shape is not changed. The agent's
own moves and the arm checks are made for this rig; the baselines run on any rigged character.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `Blender not found` | `uv sync --extra render`, or install Blender 4.2+ and `export BLENDER=/path/to/blender` |
| `OPENROUTER_API_KEY is not set` | `cp .env.example .env` and put the key in it; `uv run python -m dancebuddy check --llm` |
| 401 / 402 from OpenRouter | the key is wrong or out of credit (<https://openrouter.ai/settings/credits>) |
| `perceive` finds no person | one dancer, whole body in view, not too dark; trim the video to the dance |
| rendering is slow | `--preview` for a quick look; the full Cycles render takes ~3 s a frame on a laptop CPU |
| side-by-side has no labels | your ffmpeg lacks `drawtext`; the video is still written, without labels |
| `uv sync` cannot find bpy for your platform | install the Blender app instead (see section 2) |

Known limitations: a raised arm pulls a sheet of skin up from the body side (the generated model fuses the arm to
the body); the vision review is a model's opinion, not ground truth; the agent's own in-frame check covers the body
only (the evaluation's `hands_in_frame` covers the hands).

## 11. Repository layout

```
dancebuddy/                 the agent (python -m dancebuddy)
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
  results/     paper/ (the reported runs and table), v1/ (the first prototype's results and example runs)
blender/                    Blender scripts
  make_model.py rig_soft.py fist.py make_capsule.py make_puppet.py      characters
  render_curves.py pose_utils.py scene_utils.py                          rendering
  check_arm_shape.py check_arm_width.py measure_motion.py                measuring
assets/                     plush (model, rig, motion spec), capsule, characters from pictures
tests/test_pipeline.py
```
