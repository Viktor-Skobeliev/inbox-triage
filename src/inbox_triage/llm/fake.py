"""Scripted client used by the tests.

It exists so the failure paths can be tested on purpose: a truncated JSON, a
category that does not exist, a fenced response, a model that never recovers.
Those cases are rare enough in a live run that waiting for them would be
useless, and expensive enough in production to be worth a test each.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from ..models import TokenUsage
from .base import LLMError, LLMResponse

VALID_PAYLOAD: dict[str, Any] = {
    "category": "автоматизація",
    "target_department": "маркетинг",
    "priority": "medium",
    "short_summary": "Потрібен щотижневий звіт по Google Ads без ручного вивантаження.",
    "requested_actions": ["Автоматизувати щотижневий звіт по Google Ads"],
    "needs_clarification": False,
    "work_item_type": "project",
    "is_recurring": True,
    "language": "uk",
    "mentioned_systems": ["Google Ads"],
    "urgency_signals": [],
    "clarification_questions": [],
}


def valid_json(**overrides: Any) -> str:
    payload = {**VALID_PAYLOAD, **overrides}
    return json.dumps(payload, ensure_ascii=False)


class FakeLLMClient:
    """Replays a fixed script of responses.

    Each element is either a string (returned as-is) or an exception (raised).
    Once the script runs out the last entry repeats, which is what makes
    "the model never returns valid JSON" easy to express.
    """

    def __init__(
        self,
        responses: Sequence[str | Exception] | Callable[[str], str],
        *,
        name: str = "fake",
    ) -> None:
        self._responses = responses
        self.name = name
        self.calls: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        self.calls.append(prompt)
        if callable(self._responses):
            return LLMResponse(
                text=self._responses(prompt),
                usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
            )

        if not self._responses:
            raise LLMError("fake client has no scripted responses")
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        item = self._responses[index]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, usage=TokenUsage(prompt_tokens=10, completion_tokens=10))
