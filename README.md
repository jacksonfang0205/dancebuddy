# DanceBuddy

**An agent that makes a non-humanoid character dance along with a human dancer.** (CS329Z, Fall 2026)

Input: a video of one person dancing. Output: a video of a plush character (no legs, no neck, soft arms) dancing
the same dance, side by side with the dancer. The report describes how the agent works and how it is evaluated;
this README is about running it.

**Contents:** [1 Requirements](#1-requirements) · [2 Install](#2-install) · [3 API key](#3-api-key) ·
[4 Try it in 10 minutes](#4-try-it-in-10-minutes-no-video-needed) · [5 Run on your own dance](#5-run-the-agent-on-your-own-dance-video) ·
[6 Reproduce the evaluation](#6-reproduce-the-evaluation) · [7 Characters](#7-characters) · [8 Troubleshooting](#8-troubleshooting)

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
character. Systems: `pose_copy`, `plain_copy`, `planner` (baselines), `agent`, `agent_no_looks`, `agent_no_moves`,
`agent_numbers` (the agent and its ablations); the report describes them and the metrics. There are two ways to
reproduce it.

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

## 7. Characters

The default character is the plush (`assets/plush_soft`). To make others:

```bash
uv run python -m dancebuddy character model   MODEL.glb assets/plush             # import a 3D model
uv run python -m dancebuddy character rig     assets/plush assets/plush_soft     # rig it from its motion spec
uv run python -m dancebuddy character capsule assets/capsule                     # a built-in 3D capsule
uv run python -m dancebuddy character puppet  assets/sample_character.png assets/sample_character   # a picture
uv run python -m dancebuddy character card    assets/plush_soft                  # describe its body
```

The agent needs a character rigged with `character rig` (it also needs a motion spec, `assets/<name>_motion_spec.json`);
the baselines run on any rigged character.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Blender not found` | `uv sync --extra render`, or install Blender 4.2+ and `export BLENDER=/path/to/blender` |
| `OPENROUTER_API_KEY is not set` | `cp .env.example .env` and put the key in it; `uv run python -m dancebuddy check --llm` |
| 401 / 402 from OpenRouter | the key is wrong or out of credit (<https://openrouter.ai/settings/credits>) |
| `perceive` finds no person | one dancer, whole body in view, not too dark; trim the video to the dance |
| rendering is slow | `--preview` for a quick look; the full Cycles render takes ~3 s a frame on a laptop CPU |
| side-by-side has no labels | your ffmpeg lacks `drawtext`; the video is still written, without labels |
| `uv sync` cannot find bpy for your platform | install the Blender app instead (see section 2) |
