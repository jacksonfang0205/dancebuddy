# DanceBuddy

**An agent that makes a non-humanoid character dance along with a human dancer.** (CS329Z, Fall 2026)

Input: a video of one person dancing. Output: a video of a plush character (no legs, no neck, soft arms) dancing
the same dance, side by side with the dancer.

## 1. Requirements

- macOS or Linux, no GPU (rendering runs on the CPU), ~2 GB disk.
- [uv](https://docs.astral.sh/uv/) (installs Python 3.13) and `ffmpeg`.
- For the agent only: an [OpenRouter](https://openrouter.ai) API key (~0.1–0.4 USD per run on a 20 s dance).
  Tests, LLM-free baselines and recomputing the reported results need no key.

## 2. Install

```bash
git clone https://github.com/tianrunyan-tech/dancebuddy.git && cd dancebuddy
uv sync --extra render                 # dependencies, including Blender as the bpy module
uv run python -m dancebuddy check      # reports anything missing
uv run pytest                          # 22 tests, ~20 s
```

With Blender 4.2+ already installed: `uv sync` and `export BLENDER=/path/to/blender` instead.

## 3. API key

```bash
cp .env.example .env                       # set OPENROUTER_API_KEY=sk-or-v1-...
uv run python -m dancebuddy check --llm    # one tiny call to confirm the key
```

Default model `openai/gpt-6-sol`; override with `DANCEBUDDY_MODEL` (needs tool calling and image input).
LLM calls are cached in `.cache/llm/`, with costs in `.cache/llm/ledger.jsonl`.

## 4. Quick start (synthetic dancer, no video needed)

```bash
uv run python -m evaluation prepare                       # runs/synth: a 12 s synthetic dance + perception
uv run python -m dancebuddy agent   runs/synth            # ~3 min
uv run python -m dancebuddy render  runs/synth --preview
uv run python -m dancebuddy compose runs/synth --preview  # -> runs/synth/side_by_side_preview.mp4
```

Without a key: `uv run python -m evaluation run --no-llm && uv run python -m evaluation metrics`.

The agent writes `runs/synth/agent/`: `summary.json`, `trace.json` (all messages and tool calls), the best
`plan.json` / `curves.json` / `checks.json`, and one `attempt_N/` per plan with its rendered looks.

## 5. Run on your own video

One person, whole body in view, front view, 10–30 s.

```bash
uv run python -m dancebuddy demo my_dance.mp4 runs/my     # = perceive, agent, render, compose -> runs/my/side_by_side.mp4
```

Each step also runs on its own (`perceive`, `agent`, `render`, `compose`). Options: `--preview` for a quick
untextured render; `agent --attempts N --looks N`, `--no-looks`, `--no-geometry`, `--no-moves`;
`replay runs/my <plan.json> --name again` re-executes a saved plan without the LLM; `--char assets/<name>` picks
another rigged character (default `assets/plush_soft`).

## 6. Reproduce the evaluation

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
