"""JSON schemas for the agent's structured outputs (strict: every field required, no extra fields)."""

CHARACTER_UNDERSTANDING = {
    "type": "object",
    "properties": {
        "body_summary": {"type": "string"},
        "parts_present": {"type": "array", "items": {"type": "string"}},
        "parts_missing_or_merged": {"type": "array", "items": {"type": "string"}},
        "expressive_channels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"channel": {"type": "string"}, "reads_as": {"type": "string"}},
                "required": ["channel", "reads_as"],
                "additionalProperties": False,
            },
        },
        "translation_ideas": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["body_summary", "parts_present", "parts_missing_or_merged", "expressive_channels", "translation_ideas"],
    "additionalProperties": False,
}

_MAPPING = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "source": {"type": "string"},
        "mode": {"type": "string", "enum": ["copy", "follow", "pulse"]},
        "gain": {"type": "number"},
        "offset": {"type": "number"},
        "width_sec": {"type": "number"},
    },
    "required": ["target", "source", "mode", "gain", "offset", "width_sec"],
    "additionalProperties": False,
}

MOTION_PLAN = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "label": {"type": "string"},
                    "intent": {"type": "string"},
                    "mappings": {"type": "array", "items": _MAPPING},
                },
                "required": ["start", "end", "label", "intent", "mappings"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rationale", "segments"],
    "additionalProperties": False,
}
