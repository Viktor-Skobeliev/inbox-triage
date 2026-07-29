"""What the recovery layer is expected to survive."""

from __future__ import annotations

import pytest

from inbox_triage.models import Category, Department, Priority, WorkItemType
from inbox_triage.parsing import ParseError, extract_json_object, normalise_payload


class TestExtractJsonObject:
    def test_plain_json(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        text = 'Ось результат:\n```json\n{"a": 1}\n```\nСподіваюсь, допоміг.'
        assert extract_json_object(text) == {"a": 1}

    def test_fence_without_language_tag(self) -> None:
        assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_prose_around_object(self) -> None:
        assert extract_json_object('Звісно! {"a": 1} Готово.') == {"a": 1}

    def test_braces_inside_strings_do_not_confuse_the_scan(self) -> None:
        payload = extract_json_object('{"note": "use {this} carefully", "ok": true}')
        assert payload["note"] == "use {this} carefully"

    def test_escaped_quote_inside_string(self) -> None:
        payload = extract_json_object('{"note": "he said \\"hi\\"", "ok": true}')
        assert payload["ok"] is True

    def test_truncated_object_is_reported_not_guessed(self) -> None:
        with pytest.raises(ParseError, match="truncated"):
            extract_json_object('{"a": 1, "b": [1, 2')

    def test_empty_response(self) -> None:
        with pytest.raises(ParseError, match="empty"):
            extract_json_object("   ")

    def test_no_object_at_all(self) -> None:
        with pytest.raises(ParseError, match="no JSON object"):
            extract_json_object("Вибач, не можу допомогти.")

    def test_array_at_top_level_is_rejected(self) -> None:
        with pytest.raises(ParseError):
            extract_json_object("[1, 2, 3]")


class TestNormalisePayload:
    def test_canonical_values_pass_through_without_notes(self) -> None:
        payload = {
            "category": "автоматизація",
            "priority": "high",
            "target_department": "маркетинг",
            "work_item_type": "project",
            "needs_clarification": False,
            "requested_actions": ["зробити"],
            "mentioned_systems": [],
            "urgency_signals": [],
        }
        data, notes = normalise_payload(payload)
        assert data["category"] is Category.AUTOMATION
        assert data["priority"] is Priority.HIGH
        assert notes == []

    def test_russian_and_english_spellings_are_mapped(self) -> None:
        data, notes = normalise_payload({"category": "Автоматизация", "priority": "Высокий"})
        assert data["category"] is Category.AUTOMATION
        assert data["priority"] is Priority.HIGH
        assert len(notes) == 2

    def test_unknown_department_degrades_to_other_instead_of_failing(self) -> None:
        data, notes = normalise_payload({"target_department": "відділ загадкових справ"})
        assert data["target_department"] is Department.OTHER
        assert any("mapped to" in note for note in notes)

    def test_null_like_strings_become_none(self) -> None:
        data, _ = normalise_payload({"target_department": "не зрозуміло"})
        assert data["target_department"] is None

    def test_string_booleans(self) -> None:
        data, notes = normalise_payload({"needs_clarification": "true"})
        assert data["needs_clarification"] is True
        assert any("needs_clarification" in note for note in notes)

    def test_bare_string_becomes_a_list(self) -> None:
        data, notes = normalise_payload({"requested_actions": "зробити бота"})
        assert data["requested_actions"] == ["зробити бота"]
        assert any("to a list" in note for note in notes)

    def test_list_of_objects_is_flattened(self) -> None:
        data, _ = normalise_payload({"requested_actions": [{"action": "зробити звіт"}]})
        assert data["requested_actions"] == ["зробити звіт"]

    def test_wrapper_object_is_unwrapped(self) -> None:
        data, notes = normalise_payload({"result": {"category": "інтеграція"}})
        assert data["category"] is Category.INTEGRATION
        assert any("unwrapped" in note for note in notes)

    def test_unexpected_keys_are_dropped_not_fatal(self) -> None:
        data, notes = normalise_payload({"category": "інтеграція", "confidence": 0.9})
        assert "confidence" not in data
        assert any("dropped unexpected" in note for note in notes)

    def test_unrecognised_category_is_left_alone_for_the_validator(self) -> None:
        # The strict model must be the one to reject it, so the repair prompt
        # can quote a real error message.
        data, _ = normalise_payload({"category": "щось нове"})
        assert data["category"] == "щось нове"

    def test_missing_list_fields_default_to_empty(self) -> None:
        data, _ = normalise_payload({"category": "інтеграція"})
        assert data["requested_actions"] == []
        assert data["mentioned_systems"] == []
        assert data["urgency_signals"] == []

    def test_work_item_aliases(self) -> None:
        data, _ = normalise_payload({"work_item_type": "Інцидент"})
        assert data["work_item_type"] is WorkItemType.INCIDENT
