"""Thin wrapper around the Claude API that returns schema-constrained JSON and keeps an append-only history."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = os.environ.get("DANCEBUDDY_MODEL", "claude-opus-5-5")
DEFAULT_EFFORT = os.environ.get("DANCEBUDDY_EFFORT", "high")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


@dataclass
class Conversation:
    """One multi-turn exchange. Assistant turns are appended verbatim (including thinking blocks)."""

    system: str
    messages: list[dict] = field(default_factory=list)

    def add_user(self, content: Any) -> None:
        self.messages.append({"role": "user", "content": content})


class LLM(Protocol):
    def ask_json(self, convo: Conversation, user_content: Any, schema: dict) -> dict: ...


class ClaudeLLM:
    """Calls Claude with structured outputs. Refusals are retried server-side through `fallbacks: "default"`."""

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, max_tokens: int = 16000):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or an `ant auth login` profile
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.usage: list[dict] = []

    def ask_json(self, convo: Conversation, user_content: Any, schema: dict) -> dict:
        convo.add_user(user_content)
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=convo.system,
                messages=convo.messages,
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except self._anthropic.APIStatusError as e:
            convo.messages.pop()
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            convo.messages.pop()
            raise LLMError(f"could not reach the Claude API: {e}") from e

        self.usage.append({"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens})
        if response.stop_reason == "refusal":
            convo.messages.pop()
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise LLMError(f"request declined (category: {category})")
        if response.stop_reason == "max_tokens":
            convo.messages.pop()
            raise LLMError("response hit max_tokens before the JSON was complete")
        # Keep the full assistant turn so later turns stay consistent with earlier reasoning.
        convo.messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in response.content]})
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("response contained no text block")
        return json.loads(text)


class ScriptedLLM:
    """Offline stand-in that replays canned responses in order. Used by tests and `--dry-run`."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.usage: list[dict] = []

    def ask_json(self, convo: Conversation, user_content: Any, schema: dict) -> dict:
        convo.add_user(user_content)
        if not self.responses:
            raise LLMError("ScriptedLLM ran out of responses")
        reply = self.responses.pop(0)
        convo.messages.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(reply)}]})
        return reply
