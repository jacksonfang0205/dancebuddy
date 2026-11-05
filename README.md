# DanceBuddy

An agent that makes a **non-humanoid 3D character dance along with a human dancer** in the same rhythm
(CS329Z, Fall 2026). Given a single-dancer video and a rigged character, the agent reads the dance and the
character's body, plans how each move should be expressed on that body, renders it in Blender, checks the
result against the dancer's accents, and revises.

This is the **midpoint prototype**. Inputs are **a dance video** and **a character**, given either as a
**picture** (turned into a rigged 2D cut-out puppet) or as the built-in 3D "capsule". It includes the full
plan → check → revise loop, a pose-copy baseline, and code-based metrics.

```
video ──► perception ──► dance_score.json ─┐
                                           ├─► agent (LLM)  ──► plan.json ──► compile ──► curves.json ──► checks
character ─► Blender rig + views ─► VLM ─► character card ─┘          ▲                                       │
                                                                       └────────── failed checks ◄────────────┘
curves.json ──► Blender render ──► frames ──► ffmpeg side-by-side with the dancer
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and `ffmpeg` (`brew install ffmpeg` on macOS).
2. Install dependencies. Pick one way of getting Blender:

   ```sh
   uv sync --extra render      # includes Blender as a Python module (bpy, ~400 MB). Simplest.
   # or
   uv sync                     # then point at an installed Blender app (version 4.2 or newer):
   export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
   ```

3. Add your API key:

   ```sh
   cp .env.example .env        # then fill in OPENROUTER_API_KEY (your course key)
   ```

4. Run the tests (no API calls, no Blender, about 2 seconds):

   ```sh
   uv run pytest
   ```

## Quick start

**No video and no API key needed:** a synthetic stick-figure dancer, a hand-written plan, and the pose-copy baseline.

```sh
uv run python -m dancebuddy character                       # builds assets/capsule.blend + 4 rendered views
uv run python -m dancebuddy synth runs/synth                # synthetic dancer -> runs/synth/dance_score.json
uv run python -m dancebuddy baseline runs/synth             # pose copy -> runs/synth/posecopy/curves.json
uv run python -m dancebuddy apply runs/synth examples/handwritten_plan_synth.json   # a plan written by hand
uv run python -m dancebuddy eval runs/synth                 # metrics table
uv run python -m dancebuddy render runs/synth --systems handwritten posecopy --width 270 --height 480
uv run python -m dancebuddy compose runs/synth --systems handwritten posecopy --height 480
open runs/synth/side_by_side.mp4
```

**The full agent on a real clip** (8–15 s, one dancer, full body in frame works best) and your character picture:

```sh
uv run python -m dancebuddy demo path/to/dance.mp4 runs/clip1 --image path/to/character.png
```

**A 3D model instead of a picture** (an unrigged GLB/GLTF/OBJ/FBX, e.g. from Tripo's image-to-3D):

```sh
uv run python -m dancebuddy model path/to/plush.glb assets/plush       # decimate, auto-rig (spine + arms), card
uv run python -m dancebuddy demo path/to/dance.mp4 runs/clip2 --char assets/plush
```

3D models are not committed to git (they are large); put them in `assets/` yourself. If the model faces away from
the camera, add `--yaw 180` (or 90/-90) to `model`.

Leave out `--image` and `--char` to use the 3D capsule. The picture can be a PNG with a transparent background, or any image of
one character on a plain, single-colour background (it is keyed out automatically). Try
`examples/sample_character.png` first.

`demo` runs every step below in order and writes `runs/clip1/side_by_side.mp4` and `runs/clip1/results.md`.

## Steps

| Command | What it does | Output |
| --- | --- | --- |
| `puppet PICTURE assets/NAME` | Cuts the character out of the picture, builds a flat mesh with a 4-bone spine (weighted by height), writes the character card. Use `--char assets/NAME` in later steps | `NAME.blend`, `NAME_rig.json`, `NAME_sprite.png`, `NAME_card.json` |
| `model MODEL.glb assets/NAME` | Imports an unrigged 3D model, decimates it to ~80k faces, scales it to fit, adds a 4-bone spine plus an arm bone for each limb sticking out sideways, renders 4 views, writes the card | `NAME.blend`, `NAME_rig.json`, `NAME_views/`, `NAME_card.json` |
| `character [assets/capsule]` | Builds the capsule character in Blender, renders front/side/¾/back views | `capsule.blend`, `capsule_rig.json`, `capsule_views/` |
| `card assets/capsule_rig.json` | The LLM looks at the views + rig and describes the body (parts, missing human parts, substitutions). `--no-vlm` for rig only | `capsule_card.json` |
| `perceive VIDEO RUN` | MediaPipe pose → kinematic beats and accents → ~2 s move units with body-region shares | `keypoints.npz`, `dance_score.json` |
| `agent RUN` | The LLM plans with tools, `submit_plan` compiles + checks each plan, it revises on failures (max 4 attempts) | `agent/plan.json`, `curves.json`, `checks.json`, `trace.json`, `attempt_*/` |
| `baseline RUN` | Direct pose copy: dancer spine/head/hip angles straight onto the bone chain | `posecopy/curves.json` |
| `apply RUN PLAN` | Compiles and checks a plan you wrote yourself (no LLM), e.g. to stub the planner | `handwritten/` |
| `render RUN` | Applies curves to the rig and renders PNG frames (EEVEE on macOS, Cycles elsewhere; `--engine`) | `<system>/frames/` |
| `compose RUN` | Dancer and each system side by side | `side_by_side.mp4` |
| `eval RUN [RUN ...]` | Metrics for every system in each run dir | `results/results.json`, `.md` |

Ablations: `agent RUN --no-card` (planner sees bone names only), `agent RUN --no-revise` (one attempt).
Results go to `agent_nocard/` and `agent_norevise/`, and `eval` picks them up automatically.

## How it works

- **Dance score** (`dancebuddy/perception.py`, `beats.py`). A *hit* is a local minimum of body speed right after
  fast motion. Beats are all hits; accents are hits that are strong compared with their neighbours (±1 s).
  The clip is cut into ~2 s move units, each labelled with which regions move (arms, legs, torso, head)
  and simple features (vertical bounce, torso twist range). The score also stores 11 continuous **dancer signals**
  (`signals.py`: arm raise, torso lean/side/twist, head nod/tilt, hip bounce/sway, foot lift), each with its
  5th–95th percentile, for the `follow` primitive. Scores made before this change have no signals: re-run `perceive`.
- **Character card** (`character.py`, `blender/make_capsule.py`). Rig facts from Blender plus a VLM description.
- **Motion plan → curves** (`primitives.py`). The LLM never writes keyframes. It picks primitives per unit
  (`bounce`, `squash`, `nod`, `tilt`, `sway`, `twist`, `rock`, `lean_hold`, `follow`), bones, amplitude and which
  accents to hit. A deterministic compiler turns that into per-frame channel curves, timed so each hit stops exactly on
  its accent. **`follow`** is the continuous one: it drives a bone channel (`lean`, `twist` or `sway`) with a dancer
  signal rescaled by its percentiles, so the character *copies the shape* of moves its body can do (e.g. torso twist →
  body twist) and translates only the rest (footwork → hops). It fades in and out over 0.15 s at unit borders.
- **Checks** (`checks.py`). Run on the curves, so they apply to every system the same way:
  `accent_recall` (≥ 0.75 of dancer accents have a character hit within ±3 frames), `beat_precision`
  (≥ 0.5 of character hits land on a dancer beat), `limits` (per-channel angle/height limits), `idle_units`.
  Hit strength is measured against the motion within ±1 s (not the whole clip), so a unit with large continuous
  motion does not hide the hits of other units. `checks.json` also has a **per-unit report** (`units`): for each unit
  with a problem, the dancer's label and regions, missed accents, off-beat hits, how much the character moves there
  and how much the dancer moves, plus plain-language problems ("missed 3 of 3 accents"), so the agent knows *where*
  to revise.
- **Agent** (`agent.py`). Any OpenRouter model with tool calling (default `openai/gpt-6-sol`, set `DANCEBUDDY_MODEL`): `get_dance_score`, `get_character_card`, `list_primitives`,
  `get_dance_signals` (signal meanings and ranges, and which signals move most in each unit), `submit_plan`. The best attempt is kept; every attempt and the full trace are saved.
- **Caching** (`llm.py`). Every model call is cached by its exact request under `.cache/llm/`, so re-running a step is
  free and deterministic. Commit the cache for the example clip and the quick start replays the agent without a key.

## Copying a dancer onto a character (motion spec + soft rig)

The goal is a character that dances *like the dancer*, so the main path now copies every move the body can do and
translates only the rest:

```bash
uv run python blender/rig_soft.py -- --blend assets/plush.blend --out assets/plush_soft   # once per character
uv run python -m dancebuddy perceive my_dance.mp4 runs/my
uv run python -m dancebuddy copy runs/my --char assets/plush_soft
uv run python -m dancebuddy render runs/my --systems copy --char assets/plush_soft --preview          # ~1 min look
uv run python -m dancebuddy compose runs/my --systems copy --preview                                  # side_by_side_preview.mp4
uv run python -m dancebuddy render runs/my --systems copy --char assets/plush_soft --engine cycles   # final, 30 fps
uv run python -m dancebuddy compose runs/my --systems copy
```

- **Motion spec** (`assets/plush_motion_spec.json`). How this character moves, as qualitative rules learned by
  watching a reference video of it dancing: its pose vocabulary (rest, point, wave, both arms up, sweep across the
  chest, clap, hands to face, turn, bounce) and, for each, how the parts move and connect (the shoulder is a
  socket that slides up the body side as the arm rises; arms always pass in front of the body and hair; the body
  outline never changes; arms lead and the body follows; hair trails), plus how human parts translate (no legs:
  kicks tilt the body, bounces lift and squash it). `tools/measure_reference_video.py` measures a reference video.
- **Rendering speed.** `render` renders at 30 fps by default (a 60 fps clip renders every other frame; `--fps 0`
  keeps the clip's own rate). `--preview` is a quick untextured look (workbench, half size, ~1 min) into
  `<system>/preview`; `compose --preview` puts the previews side by side. Cycles on a 4-core CPU takes ~3.5 s a frame.
- **Rig** (`blender/rig_soft.py`). Designed from the motion spec; the mesh is not edited. Root at the base
  (tilt / turn / lift / squash); a hidden shoulder bone per arm, pivoting just inside the seam, slides the seam up
  the body as the arm rises and takes half of the arm's turn; each arm is ONE bendable bone (16-segment B-bone) that
  every arm point belongs to alone, so the arm bends as one curve and keeps its width and length (only the seam
  blends with the shoulder; the arms do not squash with the body). `blender/check_arm_width.py` measures the arm's
  width along its length in sampled frames of a run. Each mitten can ball up into a fist: a shape key per hand
  (`fist_L`, `fist_R`, `blender/fist.py`) that rolls the fingers up toward the back of the hand and rounds them off;
  the mesh itself is unchanged. The hair has no bones of its own and moves with the head.
  Body weights are heat-diffusion weights computed on a closed voxel copy (generated meshes are open and in pieces)
  and copied back by position; parts are found from the base-colour texture (blue = arms, dark brown = hair).
- **Retarget** (`dancebuddy/retarget.py`). Arm directions come from the picture (monocular depth is unreliable, so
  the 3D depth is damped), kept in front of the hair and the body; the shoulder socket slides up as the arm rises;
  a small bend is one even arc, a big bend gathers in the middle like a soft elbow (a rounded L); hands ball up into
  fists when the forearm is raised (fully when the arm is also bent) and open when the arm reaches out or drops.
  The body takes the dancer's side tilt and lean, compressed smoothly near the character's limits (never cut flat),
  and her turn all the way round: how far she has turned comes from how wide her shoulders look in the picture
  (past side-on only when both shoulders are clearly seen swapped for a few frames; small wobbles near facing the
  camera are ignored).
- **Groove** (`retarget.py`). On top of the copied motion the soft body grooves on the beat: side sway, a twist a
  quarter beat later, an upper-body counter-bend and a small squash pulse, toned down during the dancer's big moves.
- **Soft body** (`retarget.py`, SOFT). The body is a soft bag that keeps its volume: part of every sideways tilt is a
  bend of the body itself (a C curve from the base), the dancer's hip lift bends it toward the lifted hip, hips
  swaying under a still torso make an S, and the bent lower body bulges (wider and as much shorter). Tilting on the
  base rises like a roly-poly toy, so the bottom stays on the floor.
- **Leg translation** (`dancebuddy/legs.py`). The character has no legs, so every leg movement is turned into a
  visible movement of the body before any renderer or video model sees the motion: jumps (wind-up squash, stretch,
  hop, landing squash), kicks (lean away from the leg), steps (bob, lean onto the step), squats, rising on the
  toes, and moving across the floor (slide). The events found are saved in `curves.json` (`leg_events`).

## Agent on the copy pipeline (branch `agent-copy`)

The copy above is deterministic. `copy-agent` puts an LLM agent (OpenRouter, `OPENROUTER_API_KEY`) in charge of how
the dancer's moves are shown on the character, without writing keyframes:

```bash
uv run python -m dancebuddy copy-agent runs/my --char assets/plush_soft      # -> runs/my/agent_copy
uv run python -m dancebuddy render runs/my --systems agent_copy --char assets/plush_soft --engine cycles
uv run python -m dancebuddy compose runs/my --systems copy agent_copy
# re-apply a saved plan (e.g. after a rig change) without asking the LLM again:
uv run python -m dancebuddy copy runs/my --char assets/plush_soft --plan runs/my/agent_copy/plan.json --name agent_copy_v2
```

- **Plan** (`copy_plan.py`): per move unit, knobs for the body, arms, groove and head layers (head = the dancer's
  head nods and tilts moved onto the upper body, since the character has no neck); per leg event, skip or scale its
  translation; extra emphasis moves (squash, hop, nod, tilt, twist) on chosen accents. The default plan is exactly
  the plain copy.
- **Checks** (`copy_checks.py`), per unit: accents without a visible hit, completeness (does the character move
  about as much as the dancer, relatively; much less = the copy ate the move), frames out of the picture, range.
  `copy` writes the same checks for the plain copy, for comparison.
- **Loop** (`copy_agent.py`): tools to read the dance, the leg events, the signals and the motion spec; submit a
  plan, read the checks, revise (4 attempts); the best is kept with its trace.

## Metrics (`eval`)

| Metric | Meaning |
| --- | --- |
| `accent_recall` | Share of dancer accents the character hits (±100 ms at 30 fps) |
| `beat_precision` | Share of character hits that land on a dancer beat |
| `beat_alignment` | Beat alignment score: mean of exp(−d²/2σ²) over character hits, d = distance to the nearest dancer beat, σ = 3 frames |
| `mean_offset_ms` | Mean timing offset of matched accents (positive = character late) |
| `limit_violations`, `idle_units` | Channels over their limit; move units where the character barely moves |
| `jerk` | Mean absolute second difference of motion energy (lower = smoother) |
| `px_accent_recall`, `px_beat_precision` | The same two scores measured on the **rendered frames** (`video_metrics.py`), only when `<system>/frames/` exists: motion energy is the mean pixel difference between consecutive frames, hits are sharp stops in it. The agent never sees this signal, so it is an independent check on the curve metrics, and it also scores systems that have no curves (e.g. a video diffusion baseline: `motion_from_video(path, crop)`). |

## Models and cost

All model calls go through OpenRouter with your course key (`OPENROUTER_API_KEY`). The default is
`openai/gpt-6-sol`; switch with `DANCEBUDDY_MODEL=<openrouter model id>` (it must support tool calling, and image
input for the `card` step). One agent run is roughly 5 to 10 calls. Each call's exact cost is written to
`.cache/llm/ledger.jsonl`; `python -c "from dancebuddy import llm; print(llm.spent())"` prints the total.
Cached calls are free, so re-running an unchanged step costs nothing.

## Known limitations (midpoint)

- Rigs are a vertical 4-bone spine. 3D models also get arm bones when limbs stick out sideways; pictures do not yet, and legs/tails are not detected.
- Picture puppets are flat: `sway` reads well; `lean` and `twist` turn the card out of the picture plane, so their limits are lower.
- A humanoid control and more body plans (quadruped, snake, bird) come next.
- Beat detection is kinematic only. Music beats (e.g. AIST++ clips with audio) are not used yet.
- Move-unit labels are heuristics from region energy, not a learned classifier.
- The VLM looks at the character's views but not at the rendered result. Verification is code-only for now; the
  pixel metric is the first check computed from the rendered result.
- There is no automatic check that `follow` copies the dancer faithfully: curve channels have arbitrary units, so
  correlation with the dancer was not a reliable score. Faithfulness is left to the human study.

## Repository layout

```
dancebuddy/        perception, signals, beats, primitives, checks, video_metrics, baseline, character, agent, llm,
                   CLI (__main__.py)
blender/           make_capsule.py, render_curves.py, scene_utils.py (run with bpy or blender -b -P)
examples/          handwritten plans for the synthetic dancer (`_follow.json` uses the follow primitive)
tools/             measure_reference_video.py (motion style from a reference video)
tests/             fast tests with a synthetic dancer and a scripted model
assets/            generated character files (capsule.blend, rig, views, card)
runs/              one directory per clip
```
