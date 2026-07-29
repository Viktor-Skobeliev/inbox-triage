"""Aggregates are plain counting, so they are asserted exactly."""

from __future__ import annotations

from typing import Any

from inbox_triage.aggregate import UNKNOWN_DEPARTMENT, aggregate
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
from inbox_triage.report import render_report


def extraction(**overrides: Any) -> RequestExtraction:
    base: dict[str, Any] = {
        "category": Category.AUTOMATION,
        "target_department": Department.MARKETING,
        "priority": Priority.MEDIUM,
        "short_summary": "Суть.",
        "requested_actions": ["зробити"],
        "needs_clarification": False,
        "work_item_type": WorkItemType.PROJECT,
        "mentioned_systems": [],
        "urgency_signals": [],
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
        model="gemini-2.5-flash",
        temperature=0.0,
        prompt_version="v1",
        max_attempts=3,
        total_requests=4,
        succeeded=3,
        failed=1,
        retried=1,
        cache_hits=0,
        llm_calls=5,
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
    )


class TestAggregate:
    def test_counts_by_dimension(self) -> None:
        records = [
            record("REQ-001"),
            record("REQ-002", extraction=extraction(priority=Priority.HIGH)),
            record(
                "REQ-003",
                channel="Email",
                extraction=extraction(
                    category=Category.SUPPORT,
                    work_item_type=WorkItemType.INCIDENT,
                    target_department=None,
                ),
            ),
        ]
        result = aggregate(records)
        assert result.total == 3
        assert result.succeeded == 3
        assert result.by_category["автоматизація"] == 2
        assert result.by_priority["high"] == 1
        assert result.by_department[UNKNOWN_DEPARTMENT] == 1
        assert result.by_work_item_type["incident"] == 1
        assert result.by_channel["Email"] == 1

    def test_failed_records_are_counted_but_not_classified(self) -> None:
        records = [
            record("REQ-001"),
            record("REQ-002", status=RecordStatus.FAILED, extraction=None, error="boom"),
        ]
        result = aggregate(records)
        assert result.succeeded == 1
        assert result.failed == 1
        assert len(result.failures) == 1
        assert sum(result.by_category.values()) == 1

    def test_duplicates_are_excluded_from_actionable(self) -> None:
        records = [
            record("REQ-001"),
            record("REQ-013", duplicate_of="REQ-001"),
        ]
        result = aggregate(records)
        assert result.actionable_count == 1
        assert len(result.duplicates) == 1

    def test_questions_are_not_actionable(self) -> None:
        records = [
            record("REQ-004", extraction=extraction(work_item_type=WorkItemType.QUESTION)),
            record("REQ-008", extraction=extraction(work_item_type=WorkItemType.NON_ACTIONABLE)),
        ]
        assert aggregate(records).actionable_count == 0

    def test_rule_flags_are_tallied(self) -> None:
        records = [
            record("REQ-001", rule_flags=["flag_a", "flag_b"]),
            record("REQ-002", rule_flags=["flag_a"]),
        ]
        result = aggregate(records)
        assert result.by_flag["flag_a"] == 2
        assert result.by_flag["flag_b"] == 1


class TestReport:
    def test_report_contains_every_required_section(self) -> None:
        records = [
            record("REQ-001"),
            record("REQ-002", extraction=extraction(needs_clarification=True)),
            record("REQ-013", duplicate_of="REQ-001"),
            record("REQ-099", status=RecordStatus.FAILED, extraction=None, error="bad json"),
        ]
        text = render_report(aggregate(records), metadata())
        for heading in (
            "За категорією",
            "За пріоритетом",
            "За відділом",
            "Потребують уточнення",
            "Дублікати",
            "Не вдалося розібрати",
            "Параметри запуску",
        ):
            assert heading in text
        assert "REQ-002" in text
        assert "gemini-2.5-flash" in text
        assert "150" in text  # token total

    def test_empty_sections_say_so_instead_of_rendering_broken_tables(self) -> None:
        text = render_report(aggregate([record("REQ-001")]), metadata())
        assert "_немає_" in text
        assert "_не знайдено_" in text

    def test_skipped_input_rows_are_reported(self) -> None:
        text = render_report(
            aggregate([record("REQ-001")]), metadata(), skipped_rows=[(7, "missing id")]
        )
        assert "Пропущені рядки" in text
        assert "missing id" in text
