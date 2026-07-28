"""The deterministic layer: what it catches, and what it must not touch."""

from __future__ import annotations

from typing import Any

from inbox_triage import rules
from inbox_triage.models import Category, Language, Priority, RequestExtraction, WorkItemType


def make_extraction(**overrides: Any) -> RequestExtraction:
    base: dict[str, Any] = {
        "category": Category.AUTOMATION,
        "target_department": None,
        "priority": Priority.MEDIUM,
        "short_summary": "Коротка суть.",
        "requested_actions": ["зробити щось конкретне"],
        "needs_clarification": False,
        "work_item_type": WorkItemType.PROJECT,
        "is_recurring": False,
        "language": Language.UK,
        "mentioned_systems": [],
        "urgency_signals": [],
    }
    base.update(overrides)
    return RequestExtraction.model_validate(base)


class TestUrgencyMarkers:
    def test_positive_marker_detected(self) -> None:
        assert rules.text_has_urgency_marker("ГОРИТЬ, треба сьогодні")

    def test_negation_is_not_read_as_urgency(self) -> None:
        # "не горить" contains "горить"; a naive substring check gets this wrong.
        assert not rules.text_has_urgency_marker("є ідея, не горить")

    def test_negative_marker_detected_separately(self) -> None:
        assert rules.text_has_negative_urgency_marker("не терміново, колись потім")

    def test_case_and_spacing_do_not_matter(self) -> None:
        assert rules.text_has_urgency_marker("Терміново   треба")


class TestGrounding:
    def test_invented_quote_is_removed(self) -> None:
        extraction = make_extraction(
            priority=Priority.HIGH, urgency_signals=["дуже терміново негайно"]
        )
        result, flags = rules.apply_rules(extraction, "Можна автоматизувати звіт?")
        assert result.urgency_signals == []
        assert rules.FLAG_UNGROUNDED_SIGNAL in flags

    def test_real_quote_is_kept(self) -> None:
        text = "ГОРИТЬ. Сьогодні до вечора треба вивантажити список."
        extraction = make_extraction(
            priority=Priority.HIGH, work_item_type=WorkItemType.ONE_OFF, urgency_signals=["ГОРИТЬ"]
        )
        result, flags = rules.apply_rules(extraction, text)
        assert result.urgency_signals == ["ГОРИТЬ"]
        assert rules.FLAG_UNGROUNDED_SIGNAL not in flags

    def test_quote_matching_ignores_case(self) -> None:
        extraction = make_extraction(priority=Priority.HIGH, urgency_signals=["горить"])
        result, _ = rules.apply_rules(extraction, "ГОРИТЬ, треба сьогодні")
        assert result.urgency_signals == ["горить"]

    def test_system_not_present_in_text_is_flagged_but_kept(self) -> None:
        extraction = make_extraction(mentioned_systems=["Salesforce"])
        result, flags = rules.apply_rules(extraction, "Треба звіт по рекламі")
        assert result.mentioned_systems == ["Salesforce"]
        assert rules.FLAG_UNGROUNDED_SYSTEM in flags


class TestPriorityConsistency:
    def test_urgent_text_with_low_priority_is_flagged(self) -> None:
        extraction = make_extraction(priority=Priority.LOW)
        _, flags = rules.apply_rules(extraction, "терміново треба інтеграція")
        assert rules.FLAG_URGENCY_IGNORED in flags

    def test_high_priority_without_support_is_flagged(self) -> None:
        extraction = make_extraction(priority=Priority.HIGH)
        _, flags = rules.apply_rules(extraction, "Можна колись зробити дашборд?")
        assert rules.FLAG_URGENCY_INVENTED in flags

    def test_incident_may_be_high_without_urgency_words(self) -> None:
        # A broken pipeline is urgent by nature; nobody writes "терміново".
        extraction = make_extraction(
            priority=Priority.HIGH,
            category=Category.SUPPORT,
            work_item_type=WorkItemType.INCIDENT,
        )
        _, flags = rules.apply_rules(extraction, "зламалась автоматизація по інвойсам")
        assert rules.FLAG_URGENCY_INVENTED not in flags


class TestActionConsistency:
    def test_actionable_without_actions_forces_clarification(self) -> None:
        extraction = make_extraction(requested_actions=[], needs_clarification=False)
        result, flags = rules.apply_rules(extraction, "нам би табличку якусь для команди продажів")
        assert result.needs_clarification is True
        assert rules.FLAG_EMPTY_ACTIONS in flags
        assert rules.FLAG_CLARIFICATION_FORCED in flags

    def test_non_actionable_with_actions_is_flagged(self) -> None:
        extraction = make_extraction(
            work_item_type=WorkItemType.NON_ACTIONABLE, requested_actions=["щось зробити"]
        )
        _, flags = rules.apply_rules(extraction, "дякую за вчора, все працює супер")
        assert rules.FLAG_NON_ACTIONABLE_WITH_ACTIONS in flags

    def test_out_of_scope_typed_as_project_is_flagged(self) -> None:
        extraction = make_extraction(
            category=Category.OUT_OF_SCOPE, work_item_type=WorkItemType.PROJECT
        )
        _, flags = rules.apply_rules(extraction, "хочемо закупити ноутбук, куди подати заявку?")
        assert rules.FLAG_OUT_OF_SCOPE_AS_PROJECT in flags


class TestShortText:
    def test_three_words_cannot_be_a_complete_request(self) -> None:
        extraction = make_extraction(needs_clarification=False)
        result, flags = rules.apply_rules(extraction, "хлопці треба бот")
        assert result.needs_clarification is True
        assert rules.FLAG_SHORT_TEXT in flags

    def test_long_specific_text_is_left_alone(self) -> None:
        text = (
            "Можна автоматизувати щотижневий звіт по Google Ads? Зараз руками "
            "вивантажую CSV і копіюю в табличку щопонеділка."
        )
        extraction = make_extraction()
        result, flags = rules.apply_rules(extraction, text)
        assert result.needs_clarification is False
        assert flags == []


class TestNonDestructive:
    def test_rules_never_mutate_the_input_object(self) -> None:
        extraction = make_extraction(urgency_signals=["вигадка"])
        rules.apply_rules(extraction, "звичайний текст без терміновості")
        assert extraction.urgency_signals == ["вигадка"]
