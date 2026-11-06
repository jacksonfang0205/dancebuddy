# DanceBuddy

**An agent that makes a non-humanoid character dance along with a human dancer.** (CS329Z, Fall 2026)

Input: a video of one person dancing. Output: a video of a plush character (no legs, no neck, soft arms) dancing
the same dance, side by side with the dancer.

## 1. Setup

Requirements are Python 3.13, [uv](https://docs.astral.sh/uv/), FFmpeg, and Blender. Install FFmpeg through your system package manager and make sure `ffmpeg` is on your path.

```bash
git clone https://github.com/tianrunyan-tech/dancebuddy.git
cd dancebuddy
uv sync
```

Use an installed Blender application by setting its executable path. For example, on macOS:

```bash
export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
```

Alternatively, install the repository's optional Blender Python dependency:

```bash
uv sync --extra render
```

Configure OpenRouter in a local `.env` file:

```bash
cp .env.example .env
```

```dotenv
OPENROUTER_API_KEY=your-key
DANCEBUDDY_MODEL=openai/gpt-6-sol
```

Choose an OpenRouter model that supports tool calling and image input for visual review. Requests are cached locally. Token usage and reported costs are recorded in `.cache/llm/ledger.jsonl`, and each agent run records its own cost summary.

Run the offline tests:

```bash
uv run pytest
```

The tests use synthetic data and mocked model responses. They cover the planner baseline, retargeting, tool behavior, motion-language parsing, CLI options, and evaluation metrics.

## 2. Quick start (synthetic dancer, no video needed)

```bash
uv run python -m evaluation prepare                       # runs/synth: a 12 s synthetic dance + perception
uv run python -m dancebuddy agent   runs/synth            # ~3 min
uv run python -m dancebuddy render  runs/synth --preview
uv run python -m dancebuddy compose runs/synth --preview  # -> runs/synth/side_by_side_preview.mp4
```

Without a key: `uv run python -m evaluation run --no-llm && uv run python -m evaluation metrics`.

The agent writes `runs/synth/agent/`: `summary.json`, `trace.json` (all messages and tool calls), the best
`plan.json` / `curves.json` / `checks.json`, and one `attempt_N/` per plan with its rendered looks.

## 3. Run on your own video

One person, whole body in view, front view, 10–30 s.

```bash
uv run python -m dancebuddy demo my_dance.mp4 runs/my     # = perceive, agent, render, compose -> runs/my/side_by_side.mp4
```

Each step also runs on its own (`perceive`, `agent`, `render`, `compose`). Options: `--preview` for a quick
untextured render; `agent --attempts N --looks N`, `--no-looks`, `--no-geometry`, `--no-moves`;
`replay runs/my <plan.json> --name again` re-executes a saved plan without the LLM; `--char assets/<name>` picks
another rigged character (default `assets/plush_soft`).

## 4. Reproduce the evaluation

Systems: `pose_copy`, `plain_copy`, `planner` (baselines), `agent`, `agent_no_looks`, `agent_no_moves`,
`agent_numbers` (ablations).

**Exact numbers (no key, no videos).** The reported runs are committed in `evaluation/results/paper/runs/`:

```bash
uv run python -m evaluation metrics --runs evaluation/results/paper/runs --arms --out evaluation/results/paper/recomputed.json
diff evaluation/results/paper/results.md evaluation/results/paper/recomputed.md
```

**From scratch.** Put the videos listed in `evaluation/data/dances.json` at their paths, then:

```bash
uv run python -m evaluation prepare
uv run python -m evaluation run            # --systems ..., --dance <id>, --no-llm, --force
uv run python -m evaluation metrics --arms # -> evaluation/results/results.{json,md}
```

LLM systems are nondeterministic, so these numbers will be close to, not identical with, the reported ones.
`evaluation export --name <name>` saves a run set (no videos) to `evaluation/results/<name>/`.
