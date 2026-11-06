# The evaluation reported in the paper

Filled in after the final evaluation runs:

```bash
uv run python -m evaluation prepare && uv run python -m evaluation run && uv run python -m evaluation metrics --arms
uv run python -m evaluation export --name paper
```

Then this folder holds `results.{json,md}` (the reported table) and `runs/` (per dance its keypoints and dance score,
per system its curves, plan, checks, summary and LLM trace; no videos), from which anyone can recompute the table:
see the main README, section 6A.
