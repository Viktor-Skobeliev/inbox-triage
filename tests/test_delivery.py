"""The Telegram digest.

Optional and off by default, which is exactly why it needs tests: a feature
nobody exercises is worse than no feature. Nothing here touches the network.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from inbox_triage.aggregate import aggregate
from inbox_triage.delivery import TelegramSink, build_digest
from inbox_triage.models import (
    Category,
    Department,
    Priority,
    RecordStatus,
    RequestExtraction,
    RunMetadata,
    TokenUsage,
    TriageRecord,
    WorkItemType,
)

TOKEN = "1234567890:AAFsecret-token-value-that-must-not-leak"
CHAT_ID = "987654321"


def extraction(**overrides: Any) -> RequestExtraction:
    base: dict[str, Any] = {
        "category": Category.AUTOMATION,
        "target_department": Department.MARKETING,
        "priority": Priority.MEDIUM,
        "short_summary": "Суть запиту.",
        "requested_actions": ["зробити"],
        "needs_clarification": False,
        "work_item_type": WorkItemType.PROJECT,
        "mentioned_systems": [],
        "urgency_signals": [],
        "clarification_questions": [],
    }
    base.update(overrides)
    return RequestExtraction.model_validate(base)


def record(request_id: str, **overrides: Any) -> TriageRecord:
    base: dict[str, Any] = {
        "id": request_id,
        "channel": "Slack",
        "timestamp": "2026-06-08 09:00",
        "raw_text": "текст",
        "status": RecordStatus.OK,
        "extraction": extraction(),
    }
    base.update(overrides)
    return TriageRecord.model_validate(base)


def metadata() -> RunMetadata:
    return RunMetadata(
        started_at="2026-07-29T10:00:00+00:00",
        finished_at="2026-07-29T10:01:00+00:00",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-2.5-flash",
        temperature=0.0,
        prompt_version="v2",
        max_attempts=3,
        total_requests=4,
        succeeded=4,
        failed=0,
        retried=1,
        cache_hits=0,
        llm_calls=5,
        token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=200),
    )


def sample_records() -> list[TriageRecord]:
    return [
        record("REQ-005", extraction=extraction(priority=Priority.HIGH)),
        record("REQ-001"),
        record("REQ-011", extraction=extraction(needs_clarification=True)),
        record("REQ-013", duplicate_of="REQ-001"),
    ]


class TestDigestText:
    def test_carries_the_headline_numbers(self) -> None:
        text = build_digest(aggregate(sample_records()), metadata())
        assert "Запитів: 4" in text
        assert "Розібрано: 4" in text
        assert "Потребують уточнення: 1" in text
        assert "Дублікати: 1" in text

    def test_names_the_high_priority_requests(self) -> None:
        # The point of a digest is to say what needs attention, not just how
        # many things happened.
        text = build_digest(aggregate(sample_records()), metadata())
        assert "REQ-005" in text

    def test_mentions_the_model_and_the_token_cost(self) -> None:
        text = build_digest(aggregate(sample_records()), metadata())
        assert "google/gemini-2.5-flash" in text
        assert "1200" in text

    def test_is_short_enough_for_one_message(self) -> None:
        many = [record(f"REQ-{i:03d}") for i in range(1, 200)]
        assert len(build_digest(aggregate(many), metadata())) <= 3500

    def test_survives_an_empty_run(self) -> None:
        assert build_digest(aggregate([]), metadata())


class TestPublish:
    def test_posts_the_digest_to_the_configured_chat(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        sink = TelegramSink(
            TOKEN, CHAT_ID, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        sink.publish(aggregate(sample_records()), metadata())

        assert seen["url"].endswith("/sendMessage")
        assert seen["body"]["chat_id"] == CHAT_ID
        assert "Запитів: 4" in seen["body"]["text"]

    def test_a_rejected_request_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        # Delivery is best effort: the files on disk are the deliverable.
        handler = lambda _: httpx.Response(403, json={"description": "bot was blocked"})  # noqa: E731
        sink = TelegramSink(
            TOKEN, CHAT_ID, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with caplog.at_level("WARNING"):
            sink.publish(aggregate(sample_records()), metadata())
        assert any("not sent" in r.message for r in caplog.records)

    def test_a_network_failure_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host", request=request)

        sink = TelegramSink(
            TOKEN, CHAT_ID, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with caplog.at_level("WARNING"):
            sink.publish(aggregate(sample_records()), metadata())
        assert any("not sent" in r.message for r in caplog.records)


class TestTokenNeverLeaks:
    """The bot token sits in the URL, which is the easiest thing in the world
    to log by accident.
    """

    def test_not_in_the_log_on_a_rejected_request(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = lambda _: httpx.Response(401, text=f"unauthorized for {TOKEN}")  # noqa: E731
        sink = TelegramSink(
            TOKEN, CHAT_ID, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with caplog.at_level("WARNING"):
            sink.publish(aggregate(sample_records()), metadata())
        assert all(TOKEN not in r.getMessage() for r in caplog.records)

    def test_not_in_the_log_on_a_network_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # httpx puts the full URL into the string form of some errors.
            raise httpx.ConnectError(f"failed connecting to {request.url}", request=request)

        sink = TelegramSink(
            TOKEN, CHAT_ID, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with caplog.at_level("WARNING"):
            sink.publish(aggregate(sample_records()), metadata())
        assert all(TOKEN not in r.getMessage() for r in caplog.records)

    def test_not_in_the_digest_text(self) -> None:
        assert TOKEN not in build_digest(aggregate(sample_records()), metadata())
