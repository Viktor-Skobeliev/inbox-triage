"""Cross-request pass, including everything the model is not allowed to get away with."""

from __future__ import annotations

import json

from inbox_triage.dedup import apply_duplicates, find_duplicates
from inbox_triage.llm.fake import FakeLLMClient, valid_json
from inbox_triage.models import InboxRow
from inbox_triage.triage import build_record, extract_one

ROWS = [
    InboxRow(id="REQ-001", channel="Slack", timestamp="2026-06-08 09:14", raw_text="Звіт по Ads"),
    InboxRow(
        id="REQ-005", channel="Telegram", timestamp="2026-06-08 11:05", raw_text="Вивантаження"
    ),
    InboxRow(
        id="REQ-013", channel="Slack", timestamp="2026-06-08 16:02", raw_text="Той самий звіт"
    ),
]


def make_records() -> list:
    client = FakeLLMClient(lambda _: valid_json())
    return [build_record(row, extract_one(client, row)) for row in ROWS]


def dedup_response(pairs: list[dict[str, str]]) -> str:
    return json.dumps({"duplicates": pairs}, ensure_ascii=False)


class TestAcceptedPairs:
    def test_valid_pair_is_applied(self) -> None:
        records = make_records()
        client = FakeLLMClient(
            [
                dedup_response(
                    [{"id": "REQ-013", "duplicate_of": "REQ-001", "reason": "той самий звіт"}]
                )
            ]
        )
        result = find_duplicates(client, records)
        apply_duplicates(records, result)
        assert result.pairs == {"REQ-013": "REQ-001"}
        assert records[2].duplicate_of == "REQ-001"
        assert "marked_duplicate" in records[2].rule_flags

    def test_empty_result_is_normal(self) -> None:
        client = FakeLLMClient([dedup_response([])])
        result = find_duplicates(client, make_records())
        assert result.pairs == {}
        assert result.error is None


class TestRejectedPairs:
    def test_unknown_id_is_rejected(self) -> None:
        client = FakeLLMClient([dedup_response([{"id": "REQ-999", "duplicate_of": "REQ-001"}])])
        result = find_duplicates(client, make_records())
        assert result.pairs == {}
        assert any("unknown id" in note for note in result.rejected)

    def test_backwards_pair_is_rejected(self) -> None:
        # Pointing an earlier request at a later one would send work to the
        # wrong place; order comes from the input, not from the model.
        client = FakeLLMClient([dedup_response([{"id": "REQ-001", "duplicate_of": "REQ-013"}])])
        result = find_duplicates(client, make_records())
        assert result.pairs == {}
        assert any("later request" in note for note in result.rejected)

    def test_self_reference_is_rejected(self) -> None:
        client = FakeLLMClient([dedup_response([{"id": "REQ-001", "duplicate_of": "REQ-001"}])])
        result = find_duplicates(client, make_records())
        assert result.pairs == {}
        assert any("self-reference" in note for note in result.rejected)

    def test_chains_are_broken(self) -> None:
        client = FakeLLMClient(
            [
                dedup_response(
                    [
                        {"id": "REQ-005", "duplicate_of": "REQ-001"},
                        {"id": "REQ-013", "duplicate_of": "REQ-005"},
                    ]
                )
            ]
        )
        result = find_duplicates(client, make_records())
        assert "REQ-013" not in result.pairs
        assert any("chain" in note for note in result.rejected)


class TestDegradation:
    def test_unusable_json_does_not_break_the_run(self) -> None:
        client = FakeLLMClient(["не JSON"])
        result = find_duplicates(client, make_records())
        assert result.pairs == {}
        assert result.error is not None

    def test_missing_duplicates_key(self) -> None:
        client = FakeLLMClient(['{"result": "нічого не знайшов"}'])
        result = find_duplicates(client, make_records())
        assert result.error is not None

    def test_single_record_skips_the_call_entirely(self) -> None:
        client = FakeLLMClient([valid_json()])
        records = make_records()[:1]
        result = find_duplicates(client, records)
        assert result.llm_calls == 0
        assert client.calls == []
