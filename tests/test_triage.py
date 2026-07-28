"""The failure paths, which are the reason this pipeline exists.

Each test scripts a model that misbehaves in a specific way and asserts what
the pipeline does about it.
"""

from __future__ import annotations

import json

from inbox_triage.llm.base import LLMError
from inbox_triage.llm.fake import FakeLLMClient, valid_json
from inbox_triage.models import InboxRow, RecordStatus
from inbox_triage.triage import build_record, extract_one, triage_rows

ROW = InboxRow(
    id="REQ-001",
    channel="Slack",
    timestamp="2026-06-08 09:14",
    raw_text=(
        "Привіт! Можна якось автоматизувати щотижневий звіт по Google Ads? "
        "Зараз руками вивантажую CSV і копіюю в табличку щопонеділка."
    ),
)


class TestHappyPath:
    def test_single_call_when_the_model_behaves(self) -> None:
        client = FakeLLMClient([valid_json()])
        outcome = extract_one(client, ROW)
        assert outcome.extraction is not None
        assert outcome.attempts == 1
        assert outcome.llm_calls == 1
        assert outcome.extraction.category.value == "автоматизація"

    def test_tokens_are_accumulated(self) -> None:
        client = FakeLLMClient([valid_json()])
        outcome = extract_one(client, ROW)
        assert outcome.usage.total == 20


class TestRepairLoop:
    def test_fenced_response_needs_no_repair(self) -> None:
        client = FakeLLMClient([f"```json\n{valid_json()}\n```"])
        outcome = extract_one(client, ROW)
        assert outcome.extraction is not None
        assert outcome.attempts == 1

    def test_invalid_category_is_repaired_on_the_second_call(self) -> None:
        client = FakeLLMClient([valid_json(category="щось вигадане"), valid_json()])
        outcome = extract_one(client, ROW)
        assert outcome.extraction is not None
        assert outcome.attempts == 2
        assert "recovered after 2 attempts" in outcome.notes

    def test_repair_prompt_quotes_the_validator_error(self) -> None:
        client = FakeLLMClient([valid_json(category="щось вигадане"), valid_json()])
        extract_one(client, ROW)
        second_prompt = client.calls[1]
        assert "не пройшла валідацію" in second_prompt
        assert "category" in second_prompt

    def test_truncated_json_is_repaired(self) -> None:
        client = FakeLLMClient(['{"category": "автоматизація", "priority": "me', valid_json()])
        outcome = extract_one(client, ROW)
        assert outcome.extraction is not None
        assert outcome.attempts == 2

    def test_prose_only_response_is_repaired(self) -> None:
        client = FakeLLMClient(["Вибач, не можу структурувати цей запит.", valid_json()])
        outcome = extract_one(client, ROW)
        assert outcome.extraction is not None

    def test_attempt_budget_is_respected(self) -> None:
        client = FakeLLMClient([valid_json(category="вигадка")])
        outcome = extract_one(client, ROW, max_attempts=3)
        assert outcome.extraction is None
        assert outcome.attempts == 3
        assert len(client.calls) == 3


class TestGivingUp:
    def test_a_hopeless_row_produces_a_failed_record_not_an_exception(self) -> None:
        client = FakeLLMClient(["не JSON взагалі"])
        record = build_record(ROW, extract_one(client, ROW))
        assert record.status is RecordStatus.FAILED
        assert record.extraction is None
        # The row survives with everything a human needs to handle it manually.
        assert record.id == "REQ-001"
        assert record.raw_text == ROW.raw_text
        assert record.error
        assert record.raw_response == "не JSON взагалі"

    def test_raw_response_is_truncated(self) -> None:
        client = FakeLLMClient(["x" * 5000])
        outcome = extract_one(client, ROW, max_attempts=1)
        assert outcome.raw_response is not None
        assert len(outcome.raw_response) == 1200

    def test_provider_failure_stops_that_row_only(self) -> None:
        client = FakeLLMClient([LLMError("429 quota exceeded", retryable=True)])
        outcome = extract_one(client, ROW)
        assert outcome.extraction is None
        assert outcome.error is not None
        assert "llm call failed" in outcome.error
        # No point burning the repair budget on a transport failure.
        assert outcome.attempts == 1

    def test_one_bad_row_does_not_stop_the_run(self) -> None:
        good = valid_json()

        def script(prompt: str) -> str:
            return "зламана відповідь" if "REQ-002" in prompt else good

        rows = [
            ROW,
            InboxRow(id="REQ-002", channel="Telegram", timestamp="", raw_text="хлопці треба бот"),
            InboxRow(id="REQ-003", channel="Email", timestamp="", raw_text="Потрібне саммарі."),
        ]
        run = triage_rows(FakeLLMClient(script), rows, max_attempts=2)
        assert [record.status for record in run.records] == [
            RecordStatus.OK,
            RecordStatus.FAILED,
            RecordStatus.OK,
        ]
        assert run.retried == 1


class TestRuleIntegration:
    def test_rules_run_on_top_of_a_valid_extraction(self) -> None:
        payload = json.loads(valid_json())
        payload["urgency_signals"] = ["ГОРИТЬ"]
        payload["priority"] = "high"
        record = build_record(ROW, extract_one(FakeLLMClient([json.dumps(payload)]), ROW))
        assert record.status is RecordStatus.OK
        assert record.extraction is not None
        # The quote is not in the source text, so it is dropped.
        assert record.extraction.urgency_signals == []
        assert "ungrounded_urgency_signal_removed" in record.rule_flags


class TestConcurrency:
    def test_output_order_matches_input_order(self) -> None:
        rows = [
            InboxRow(id=f"REQ-{i:03d}", channel="Slack", timestamp="", raw_text=f"запит {i} " * 10)
            for i in range(1, 9)
        ]
        run = triage_rows(FakeLLMClient(lambda _: valid_json()), rows, concurrency=4)
        assert [record.id for record in run.records] == [row.id for row in rows]
