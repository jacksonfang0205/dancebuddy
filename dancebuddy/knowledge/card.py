"""Character understanding: rig info + rendered views -> character card.

The card tells the planner what body it is working with: which parts exist,
which human parts are missing or merged, and which bones/channels can stand in
for them. The rig half is parsed by Blender (make_capsule.py writes it); the
descriptive half comes from a VLM looking at the rendered views.
"""

from __future__ import annotations

import json
from pathlib import Path

from dancebuddy import llm

PROMPT = """You are helping an animation agent make a non-humanoid 3D character dance along with a human dancer.
Attached are rendered views of the character ({views}) and its rig is below.

Rig (JSON):
{rig}

Rig conventions: every bone points up. Rotation channels per bone are lean (forward/back), twist (turn around the
vertical axis) and sway (side to side), in degrees. The root bone also has lift (hop height, metres) and squash.

Describe the character's body as JSON with exactly these keys:
{{
  "body_plan": "one sentence",
  "visible_parts": ["..."],
  "missing_human_parts": ["arms", "legs", "neck", ...],
  "bone_regions": {{"<bone name>": "which visible part this bone moves"}},
  "substitutions": {{"arms": "how arm motion could be expressed on this body", "legs": "...", "hips": "...", "head": "...", "torso": "..."}},
  "constraints": ["things that would look wrong, e.g. sinking into the floor"]
}}
Return only the JSON."""


def load_rig(rig_json: str | Path) -> dict:
    rig = json.loads(Path(rig_json).read_text())
    rig["_dir"] = str(Path(rig_json).resolve().parent)
    return rig


def build_card(rig_json: str | Path, out: str | Path, use_vlm: bool = True, model: str | None = None) -> dict:
    rig = load_rig(rig_json)
    base = {k: v for k, v in rig.items() if not k.startswith("_") and k != "views"}
    card = dict(base)
    card["description"] = None
    card["source"] = "rig-only"
    if use_vlm and rig.get("views"):
        views = rig["views"]
        content = []
        for name, rel in views.items():
            content.append({"type": "text", "text": f"View: {name}"})
            content.append(llm.image_block(Path(rig["_dir"]) / rel))
        content.append({"type": "text", "text": PROMPT.format(views=", ".join(views), rig=json.dumps(base, indent=1))})
        resp = llm.call([{"role": "user", "content": content}], model=model, max_tokens=4000)
        card["description"] = llm.extract_json(resp["text"])
        card["source"] = "rig+vlm"
    Path(out).write_text(json.dumps(card, indent=1))
    print(f"character card ({card['source']}) -> {out}")
    return card
