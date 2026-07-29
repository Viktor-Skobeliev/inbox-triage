"""Optional delivery of the run summary to somewhere people already look.

Everything here is off unless configured, and a failure to deliver never fails
the run: the files on disk are the deliverable, this is a convenience on top.

Telegram is implemented because it needs nothing but the HTTP client already in
the project. Google Sheets is not implemented, and the extension point below is
what it would plug into: a callable that takes the aggregates and the run
metadata and does something with them.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from .aggregate import Aggregates
from .models import RunMetadata

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 3500


class Sink(Protocol):
    """Where a finished run can be published.

    A Google Sheets sink implements this same call: build the rows from
    ``aggregates.by_category`` and friends, then append them to a worksheet.
    It is left out because it needs a service account, its own credentials file
    and an OAuth scope, which is a lot of moving parts for a summary that is
    already written to disk.
    """

    name: str

    def publish(self, aggregates: Aggregates, run: RunMetadata) -> None: ...


def build_digest(aggregates: Aggregates, run: RunMetadata) -> str:
    """Short text summary, the same numbers as the report header."""
    lines = [
        "Розбір інбоксу завершено.",
        "",
        f"Запитів: {aggregates.total}",
        f"Розібрано: {aggregates.succeeded}, не вдалося: {aggregates.failed}",
        f"Готові до роботи: {aggregates.actionable_count}",
        f"Потребують уточнення: {len(aggregates.needs_clarification)}",
        f"Дублікати: {len(aggregates.duplicates)}",
    ]

    if aggregates.by_priority.get("high"):
        high = [
            record.id
            for record in aggregates.actionable
            if record.extraction and record.extraction.priority.value == "high"
        ]
        if high:
            lines += ["", f"Високий пріоритет: {', '.join(high)}"]

    lines += ["", f"Модель: {run.model}, токенів: {run.token_usage.total}"]
    return "\n".join(lines)[:MAX_MESSAGE_CHARS]


class TelegramSink:
    """Posts the digest to a chat. Never raises: delivery is best-effort."""

    name = "telegram"

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._timeout = timeout
        # Injectable so the tests can drive this without a network call: an
        # optional feature nobody ever exercises is worse than no feature.
        self._client = client

    def publish(self, aggregates: Aggregates, run: RunMetadata) -> None:
        url = f"{TELEGRAM_API}/bot{self._token}/sendMessage"
        payload = {"chat_id": self._chat_id, "text": build_digest(aggregates, run)}
        try:
            if self._client is not None:
                response = self._client.post(url, json=payload, timeout=self._timeout)
            else:
                response = httpx.post(url, json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            # The token sits in the URL, so log the exception type and message
            # only. httpx puts the full URL into str() of some errors.
            log.warning("telegram digest not sent: %s", type(exc).__name__)
            return

        if response.status_code != 200:
            # The body is quoted because the reason matters, but Telegram
            # echoes the request URL on some errors, and the token lives in
            # that URL.
            log.warning(
                "telegram digest not sent: HTTP %s %s",
                response.status_code,
                self._scrub(response.text[:200]),
            )
            return
        log.info("telegram digest sent")

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "[REDACTED]") if self._token else text
