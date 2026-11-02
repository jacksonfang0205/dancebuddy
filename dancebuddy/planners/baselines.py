"""Non-agent baselines.

B1 direct copy: every character channel copies its anatomical counterpart on the dancer; dancer motion with no
counterpart is dropped. This is what a standard pose-copy retargeting pipeline does.

B2 fixed rules: direct copy plus a fixed, hand-written translation for each missing body part, applied the same way
to every dance (no per-dance or per-segment decisions).
"""

from __future__ import annotations

from ..characters.spec import CharacterSpec
from ..motion.plan import Mapping, MotionPlan, whole_clip
from ..perception.features import DanceFeatures

# Root translation is measured in dancer torso lengths; a torso is roughly 0.3 of body height.
TORSO_TO_BODY_HEIGHT = 0.3

# character channel -> (dancer feature, gain) for an anatomical one-to-one copy
DIRECT_COUNTERPARTS = {
    "root_bounce": ("bounce", TORSO_TO_BODY_HEIGHT),
    "root_sway": ("sway", TORSO_TO_BODY_HEIGHT),
    # A merged head-torso body is the trunk, so it copies the torso.
    "body_tilt": ("torso_lean_side", 1.0),
    "body_lean": ("torso_lean_fwd", 1.0),
    "body_twist": ("torso_twist", 1.0),
    "torso_lean_side": ("torso_lean_side", 1.0),
    "torso_lean_fwd": ("torso_lean_fwd", 1.0),
    "torso_twist": ("torso_twist", 1.0),
    "head_nod": ("head_nod", 1.0),
    "head_tilt": ("head_tilt", 1.0),
    "leg_L_lift": ("hip_L_flex", 1.0),
    "leg_R_lift": ("hip_R_flex", 1.0),
}
for _side in ("L", "R"):
    for _joint in ("raise", "forward", "elbow"):
        DIRECT_COUNTERPARTS[f"arm_{_side}_{_joint}"] = (f"arm_{_side}_{_joint}", 1.0)
    DIRECT_COUNTERPARTS[f"hip_{_side}_flex"] = (f"hip_{_side}_flex", 1.0)
    DIRECT_COUNTERPARTS[f"knee_{_side}_bend"] = (f"knee_{_side}_bend", 1.0)


def direct_copy_mappings(spec: CharacterSpec) -> list[Mapping]:
    mappings = []
    for name in spec.channels:
        if name in DIRECT_COUNTERPARTS:
            source, gain = DIRECT_COUNTERPARTS[name]
            mappings.append(Mapping(target=name, source=source, mode="copy", gain=gain))
    return mappings


def direct_copy_plan(spec: CharacterSpec, features: DanceFeatures) -> MotionPlan:
    plan = whole_clip(features, direct_copy_mappings(spec), label="direct copy")
    plan.rationale = "B1: copy each channel's anatomical counterpart; drop dancer motion with no counterpart."
    return plan


def fixed_rules_plan(spec: CharacterSpec, features: DanceFeatures) -> MotionPlan:
    mappings = direct_copy_mappings(spec)
    ch = spec.channels
    missing = " ".join(spec.parts_missing).lower()
    if "legs" in missing and "root_bounce" in ch:
        mappings.append(Mapping("root_bounce", "steps", "pulse", gain=0.08, width_sec=0.3))
    if "squash" in ch:
        mappings.append(Mapping("squash", "accents", "pulse", gain=-0.12, width_sec=0.25))
    if "neck" in missing and "body_lean" in ch:
        mappings.append(Mapping("body_lean", "head_nod", "follow", gain=6.0))
    if "knees" in missing:
        for side in ("L", "R"):
            if f"leg_{side}_lift" in ch:
                mappings.append(Mapping(f"leg_{side}_lift", f"step_{side}", "pulse", gain=15.0, width_sec=0.3))
    plan = whole_clip(features, mappings, label="fixed rules")
    plan.rationale = "B2: direct copy plus one fixed translation per missing body part, identical for every dance."
    return plan
