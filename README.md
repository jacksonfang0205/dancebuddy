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

## 2. Run a dance

The bundled default character is `assets/plush_soft`. A run starts from either your own dance video or the built-in synthetic dancer.

**With your own video.** Use a video of one person with the whole body in view, filmed from the front, 10–30 seconds long. Use a fresh run directory for each source video, because perception reuses cached keypoints in an existing directory.

```bash
uv run python -m dancebuddy perceive path/to/dance.mp4 runs/my_dance
```

**Without a video.** The repository generates a 12-second synthetic dance and its perception in `runs/synth`. Use `runs/synth` in place of `runs/my_dance` in the commands below.

```bash
uv run python -m evaluation prepare --dance synth
```

Then run the agent and preview the result:

```bash
uv run python -m dancebuddy agent runs/my_dance --attempts 4 --looks 4
uv run python -m dancebuddy render runs/my_dance --preview
uv run python -m dancebuddy compose runs/my_dance --preview
```

The preview uses Workbench at half the requested dimensions. Its side-by-side video is saved to `runs/my_dance/side_by_side_preview.mp4`.

For a textured render:

```bash
uv run python -m dancebuddy render runs/my_dance --engine eevee
uv run python -m dancebuddy compose runs/my_dance
```

The final video is saved to `runs/my_dance/side_by_side.mp4`. Rendering defaults to Cycles and a target of 30 fps. Frame-rate reduction uses an integer sampling step based on the source rate.

The full workflow is also available as one command:

```bash
uv run python -m dancebuddy demo path/to/dance.mp4 runs/another_dance --preview
```

### Inspect and replay a run

The agent writes to `runs/my_dance/agent/`: `summary.json` (attempts, whether its checks passed, LLM calls, cost), `trace.json` (every message and tool call), the best `plan.json`, `curves.json` and `checks.json`, and one `attempt_N/` folder per submitted plan with its rendered looks (`look_N/look.png`).

Carry out a saved plan again without the LLM:

```bash
uv run python -m dancebuddy replay runs/my_dance runs/my_dance/agent/plan.json --name replay
uv run python -m dancebuddy render runs/my_dance --systems replay --preview
```

`agent --no-looks`, `--no-geometry` and `--no-moves` switch a capability off; `--name` sets the output folder and `--char assets/<name>` another rigged character.

## 3. Reproduce the evaluation

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
