"""Deterministic checks that run after the schema has already passed.

The schema proves the shape is right. These rules ask whether the content makes
sense, in plain code with fixed thresholds, so the answer is the same on every
run.

Several of them compare the model against the source text rather than against
itself. That is the cheapest hallucination check available, and it costs no
extra call.

Rules flag by default. They correct only where the output contradicts itself
and one side is clearly wrong.
"""

from __future__ import annotations

import re
import unicodedata

from .models import Category, Department, RequestExtraction, WorkItemType

# Masked out before the positive markers are matched, so a denial cannot read
# as urgency. A list of exact phrases would miss "не дуже терміново", hence the
# optional intensifier and the inflected ending.
NEGATIVE_URGENCY_PATTERN = re.compile(
    r"\bне\s+(?:дуже\s+|так\s+|надто\s+|особливо\s+|сильно\s+)?"
    r"(?:термінов\w*|горит\w*|горить|спіш\w*|поспіш\w*|пал\w*|срочн\w*)"
    r"|\bno rush\b|\bnot urgent\b|\bнизьк\w*\s+пріоритет\w*"
)

POSITIVE_URGENCY_MARKERS = (
    "горить",
    "горит",
    "термінов",
    "срочно",
    "негайно",
    "asap",
    "urgent",
    "критично",
    "до вечора",
    "сьогодні до",
    "якнайшвидше",
    "як найшвидше",
)

# Departments people name explicitly. Used only to tell "the text does not say"
# apart from "the model did not notice", which are very different problems.
DEPARTMENT_HINTS: tuple[tuple[str, Department], ...] = (
    ("hr", Department.HR),
    ("рекрутинг", Department.HR),
    ("бухгалтер", Department.FINANCE),
    ("фінанс", Department.FINANCE),
    ("відділ продаж", Department.SALES),
    ("продаж", Department.SALES),
    ("аналітик", Department.ANALYTICS),
    ("маркетинг", Department.MARKETING),
    ("smm", Department.SMM),
    ("контент", Department.CONTENT),
    ("підтримк", Department.IT_SUPPORT),
)
# "#support" is a Slack channel in this dataset, not the department asking, and
# it produced a false accusation on the first live run. Channel names are
# stripped before hints are looked for.
CHANNEL_MENTION = re.compile(r"#\w+")

# Cyrillic spellings of product names, so a declined form is not reported as
# invented: people write "з Мети" and "слака з планфіксом".
SYSTEM_SPELLINGS: dict[str, tuple[str, ...]] = {
    "meta": ("мет",),
    "facebook": ("фейсбук", "фб"),
    "slack": ("слак",),
    "planfix": ("планфікс", "планфикс"),
    "bigquery": ("біквер", "бігквер"),
    "google ads": ("гугл адс",),
    "google sheets": ("гугл шіт", "гугл-таблиц"),
    "telegram": ("телеграм", "тг"),
}

# Phrases that try to talk to the model instead of describing a request. The
# inbox is internal, but a forwarded client email lands here too, so the text
# is not trusted input.
INJECTION_PATTERN = re.compile(
    r"ігноруй\s+(?:усі\s+|всі\s+)?попередн|забудь\s+(?:усі\s+|всі\s+)?попередн"
    r"|ignore\s+(?:all\s+)?(?:previous|prior|above)|disregard\s+(?:all\s+)?previous"
    r"|system\s+prompt|системн\w*\s+промпт|ти\s+тепер\b|you\s+are\s+now\b"
    r"|new\s+instructions|нові\s+інструкції|override\s+(?:your\s+)?instructions"
)

SHORT_TEXT_CHARS = 40
MAX_SUMMARY_CHARS = 300
MAX_TEXT_CHARS = 8000

FLAG_UNGROUNDED_SIGNAL = "ungrounded_urgency_signal_removed"
FLAG_URGENCY_IGNORED = "urgency_marker_in_text_but_priority_not_high"
FLAG_URGENCY_INVENTED = "priority_high_without_urgency_marker"
FLAG_URGENCY_DENIED = "priority_high_although_author_said_not_urgent"
FLAG_EMPTY_ACTIONS = "actionable_but_no_requested_actions"
FLAG_CLARIFICATION_FORCED = "needs_clarification_forced_true"
FLAG_CLARIFICATION_WITHOUT_QUESTIONS = "needs_clarification_without_questions"
FLAG_NON_ACTIONABLE_WITH_ACTIONS = "non_actionable_but_actions_listed"
FLAG_OUT_OF_SCOPE_AS_PROJECT = "out_of_scope_but_typed_as_project"
FLAG_SHORT_TEXT = "very_short_text_without_clarification"
FLAG_UNGROUNDED_SYSTEM = "system_not_found_in_text"
FLAG_SUMMARY_TOO_LONG = "summary_longer_than_one_sentence"
FLAG_DEPARTMENT_HINT_IGNORED = "department_named_in_text_but_left_empty"
FLAG_DEPARTMENT_INVENTED = "department_not_named_in_text"
FLAG_POSSIBLE_INJECTION = "possible_prompt_injection_in_text"

ACTIONABLE_TYPES = {WorkItemType.PROJECT, WorkItemType.ONE_OFF, WorkItemType.INCIDENT}


def _fold(text: str) -> str:
    """Lowercase, compose, normalise apostrophes and whitespace.

    Composing (NFC) rather than decomposing matters here: stripping combining
    marks would turn "й" into "и" and "ї" into "і", which are different letters
    in Ukrainian, and every marker containing them would quietly stop matching.
    """
    lowered = unicodedata.normalize("NFC", text.lower())
    for apostrophe in ("ʼ", "’", "`", "´"):
        lowered = lowered.replace(apostrophe, "'")
    return re.sub(r"\s+", " ", lowered).strip()


def _mask_negatives(folded: str) -> str:
    return NEGATIVE_URGENCY_PATTERN.sub(" ", folded)


def text_has_urgency_marker(raw_text: str) -> bool:
    folded = _mask_negatives(_fold(raw_text))
    return any(_fold(marker) in folded for marker in POSITIVE_URGENCY_MARKERS)


def text_has_negative_urgency_marker(raw_text: str) -> bool:
    return bool(NEGATIVE_URGENCY_PATTERN.search(_fold(raw_text)))


def text_looks_like_injection(raw_text: str) -> bool:
    return bool(INJECTION_PATTERN.search(_fold(raw_text)))


def department_hint(raw_text: str) -> Department | None:
    """Which department the text names outright, if any."""
    folded = CHANNEL_MENTION.sub(" ", _fold(raw_text))
    for marker, department in DEPARTMENT_HINTS:
        if marker in folded:
            return department
    return None


def system_is_mentioned(system: str, folded_text: str) -> bool:
    """Whether a named product can be traced to the text, allowing inflection."""
    folded = _fold(system)
    if not folded:
        return False
    if folded in folded_text:
        return True

    for spelling in SYSTEM_SPELLINGS.get(folded, ()):
        if spelling in folded_text:
            return True

    # "PlanFix" against "планфіксом": compare on a stem, not the full token.
    tokens = [token for token in re.split(r"[\s/_-]+", folded) if len(token) >= 4]
    return bool(tokens) and all(token[:-1] in folded_text for token in tokens)


def apply_rules(
    extraction: RequestExtraction, raw_text: str
) -> tuple[RequestExtraction, list[str]]:
    """Return a possibly corrected extraction plus the flags that fired."""
    flags: list[str] = []
    data = extraction.model_copy(deep=True)
    folded_text = _fold(raw_text)

    # 1. Urgency quotes must exist in the source. Anything else is invented.
    grounded = [s for s in data.urgency_signals if _fold(s) and _fold(s) in folded_text]
    if len(grounded) != len(data.urgency_signals):
        flags.append(FLAG_UNGROUNDED_SIGNAL)
        data.urgency_signals = grounded

    is_high = data.priority.value == "high"
    has_marker = text_has_urgency_marker(raw_text)

    # 2. The text says it is on fire but the model did not agree.
    if has_marker and not is_high:
        flags.append(FLAG_URGENCY_IGNORED)

    # 3. The model says high, the text gives nothing to support it.
    if is_high and not (has_marker or data.work_item_type is WorkItemType.INCIDENT):
        flags.append(FLAG_URGENCY_INVENTED)

    # 4. The author said outright it is not urgent, and was overruled anyway.
    if is_high and text_has_negative_urgency_marker(raw_text):
        flags.append(FLAG_URGENCY_DENIED)

    # 5. Work meant to be picked up, but nobody wrote down what to do.
    if data.work_item_type in ACTIONABLE_TYPES and not data.requested_actions:
        flags.append(FLAG_EMPTY_ACTIONS)
        if not data.needs_clarification:
            data.needs_clarification = True
            flags.append(FLAG_CLARIFICATION_FORCED)

    # 6. Nothing to do, yet a to-do list.
    if data.work_item_type is WorkItemType.NON_ACTIONABLE and data.requested_actions:
        flags.append(FLAG_NON_ACTIONABLE_WITH_ACTIONS)

    # 7. Out of scope for the unit, but queued as a project anyway.
    if data.category is Category.OUT_OF_SCOPE and data.work_item_type is WorkItemType.PROJECT:
        flags.append(FLAG_OUT_OF_SCOPE_AS_PROJECT)

    # 8. A handful of words cannot be a fully specified request, unless there
    #    is no request at all: "дякую за вчора" is short because it is a
    #    thank-you note, and asking its author to clarify would be absurd.
    if (
        len(raw_text.strip()) < SHORT_TEXT_CHARS
        and data.work_item_type is not WorkItemType.NON_ACTIONABLE
        and not data.needs_clarification
    ):
        flags.append(FLAG_SHORT_TEXT)
        data.needs_clarification = True
        if FLAG_CLARIFICATION_FORCED not in flags:
            flags.append(FLAG_CLARIFICATION_FORCED)

    # 9. Saying "this is too vague" without saying what to ask is useless to
    #    whoever has to write back.
    if data.needs_clarification and not data.clarification_questions:
        flags.append(FLAG_CLARIFICATION_WITHOUT_QUESTIONS)

    # 10. Named systems should be traceable to the text.
    if any(not system_is_mentioned(system, folded_text) for system in data.mentioned_systems):
        flags.append(FLAG_UNGROUNDED_SYSTEM)

    # 11. Department, both ways round. A null can mean "the text does not say"
    #     or "the model missed it", and a filled field can mean "the text says
    #     so" or "the model guessed from the topic". Without these two flags
    #     all four cases look identical in the output, and there is nothing to
    #     improve the prompt against.
    hinted = department_hint(raw_text)
    if data.target_department is None and hinted is not None:
        flags.append(FLAG_DEPARTMENT_HINT_IGNORED)
    elif data.target_department is not None and hinted is None:
        # "хлопці треба бот" came back as it/підтримка on the first live run.
        flags.append(FLAG_DEPARTMENT_INVENTED)

    # 12. "One sentence" is in the spec, so it is checkable.
    if len(data.short_summary) > MAX_SUMMARY_CHARS:
        flags.append(FLAG_SUMMARY_TOO_LONG)

    # 13. The text tried to give the model instructions.
    if text_looks_like_injection(raw_text):
        flags.append(FLAG_POSSIBLE_INJECTION)

    return data, flags
