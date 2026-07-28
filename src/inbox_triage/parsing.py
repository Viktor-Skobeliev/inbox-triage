"""Turning whatever the model said into something Pydantic can look at.

Forgiving on purpose: this is the layer that absorbs code fences, a sentence
before the JSON, a Russian spelling of a Ukrainian category, ``"true"`` instead
of ``true``, a bare string where a list was asked for.

It does not invent values. A field that is missing or unreadable stays that
way, and the strict model rejects the record, which is the point. Every fix is
recorded so a run can be judged on how much repair it needed.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from .models import Category, Department, Language, Priority, WorkItemType

_FENCE = "```"

# Kept to spellings that plausibly turn up: Russian instead of Ukrainian, and
# English. Single letters and digits are not here on purpose, they were never
# observed and the repair prompt covers the long tail.
_CATEGORY_ALIASES: dict[str, Category] = {
    "автоматизация": Category.AUTOMATION,
    "automation": Category.AUTOMATION,
    "интеграция": Category.INTEGRATION,
    "integration": Category.INTEGRATION,
    "звіт": Category.ANALYTICS,
    "отчет/аналитика": Category.ANALYTICS,
    "аналітика": Category.ANALYTICS,
    "аналитика": Category.ANALYTICS,
    "analytics": Category.ANALYTICS,
    "баг": Category.SUPPORT,
    "баг/поддержка": Category.SUPPORT,
    "support": Category.SUPPORT,
    "питання": Category.QUESTION,
    "вопрос/консультация": Category.QUESTION,
    "question": Category.QUESTION,
    "out of scope": Category.OUT_OF_SCOPE,
    "вне скоупа": Category.OUT_OF_SCOPE,
}

_PRIORITY_ALIASES: dict[str, Priority] = {
    "низький": Priority.LOW,
    "низкий": Priority.LOW,
    "середній": Priority.MEDIUM,
    "средний": Priority.MEDIUM,
    "normal": Priority.MEDIUM,
    "високий": Priority.HIGH,
    "высокий": Priority.HIGH,
    "urgent": Priority.HIGH,
    "critical": Priority.HIGH,
}

_DEPARTMENT_ALIASES: dict[str, Department] = {
    "marketing": Department.MARKETING,
    "sales": Department.SALES,
    "продажи": Department.SALES,
    "відділ продажів": Department.SALES,
    "analytics": Department.ANALYTICS,
    "аналитика": Department.ANALYTICS,
    "finance": Department.FINANCE,
    "фінанси": Department.FINANCE,
    "бухгалтерія": Department.FINANCE,
    "бухгалтерия": Department.FINANCE,
    "hr": Department.HR,
    "content": Department.CONTENT,
    "контент": Department.CONTENT,
    "smm": Department.SMM,
    "pm": Department.PM,
    "it": Department.IT_SUPPORT,
    "support": Department.IT_SUPPORT,
    "підтримка": Department.IT_SUPPORT,
    "other": Department.OTHER,
    "інше": Department.OTHER,
}

_WORK_ITEM_ALIASES: dict[str, WorkItemType] = {
    "проект": WorkItemType.PROJECT,
    "проєкт": WorkItemType.PROJECT,
    "разова": WorkItemType.ONE_OFF,
    "інцидент": WorkItemType.INCIDENT,
    "инцидент": WorkItemType.INCIDENT,
    "питання": WorkItemType.QUESTION,
    "вопрос": WorkItemType.QUESTION,
}

_LANGUAGE_ALIASES: dict[str, Language] = {
    "ukrainian": Language.UK,
    "українська": Language.UK,
    "ua": Language.UK,
    "english": Language.EN,
    "англійська": Language.EN,
    "змішана": Language.MIXED,
    "mixed": Language.MIXED,
}

_TRUE_WORDS = {"true", "yes", "так", "да"}
_FALSE_WORDS = {"false", "no", "ні", "нет"}
_NULL_WORDS = {"", "null", "none", "n/a", "невідомо", "не зрозуміло", "unknown", "-"}

ALLOWED_FIELDS = frozenset(
    {
        "category",
        "target_department",
        "priority",
        "short_summary",
        "requested_actions",
        "needs_clarification",
        "work_item_type",
        "is_recurring",
        "language",
        "mentioned_systems",
        "urgency_signals",
        "clarification_questions",
    }
)

_LIST_FIELDS = (
    "requested_actions",
    "mentioned_systems",
    "urgency_signals",
    "clarification_questions",
)
_WRAPPER_KEYS = ("result", "data", "output", "extraction", "fields", "response")


class ParseError(ValueError):
    """No JSON object could be recovered from the response at all."""


def _iter_candidates(text: str) -> list[str]:
    """Every balanced ``{...}`` span in the text, in order."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : index + 1])
    if depth and start >= 0:
        spans.append(text[start:])  # truncated tail, reported separately below
    return spans


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first parseable JSON object out of a model response.

    Tries every candidate rather than only the first ``{``: prose like
    "Ось результат {ключ: значення}:" would otherwise swallow the real object
    sitting on the next line.
    """
    if not text or not text.strip():
        raise ParseError("empty response")

    candidate = text.strip()

    if _FENCE in candidate:
        blocks = candidate.split(_FENCE)
        for block in blocks[1::2]:
            body = block
            if "\n" in body:
                first_line, rest = body.split("\n", 1)
                if first_line.strip().lower() in {"json", "json5", ""}:
                    body = rest
            body = body.strip()
            if body.startswith("{"):
                candidate = body
                break

    spans = _iter_candidates(candidate)
    if not spans:
        raise ParseError("no JSON object found in response")

    for span in spans:
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    if not spans[-1].rstrip().endswith("}"):
        raise ParseError("JSON object is truncated (unbalanced braces)")
    raise ParseError("no parseable JSON object in response")


def _as_text(value: Any) -> str:
    return str(value).strip().lower().replace("ʼ", "'")


def _coerce_enum(value: Any, enum_cls: type[Enum], aliases: dict[str, Any]) -> Any:
    """Map a loose value onto an enum member, or return it untouched.

    Returning the original on failure is intentional: the strict model then
    rejects the record with an error the repair prompt can quote.
    """
    if value is None or isinstance(value, enum_cls):
        return value
    text = _as_text(value)
    if not text:
        return value
    for member in enum_cls:
        if text == str(member.value).lower():
            return member
    if text in aliases:
        return aliases[text]
    squashed = text.replace(" ", "").replace("-", "").replace("_", "")
    for alias, member in aliases.items():
        if squashed == alias.replace(" ", "").replace("-", "").replace("_", ""):
            return member
    return value


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    text = _as_text(value)
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return value


def _coerce_str_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [] if stripped.lower() in _NULL_WORDS else [stripped]
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # Models sometimes answer [{"action": "..."}] instead of ["..."].
                for key in ("action", "question", "text", "value", "name", "title"):
                    if isinstance(item.get(key), str):
                        out.append(item[key])
                        break
            elif item is not None:
                out.append(str(item))
        return out
    return value


def _unwrap(data: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """Unwrap ``{"result": {...}}``, including when the model added a sibling.

    Requiring the wrapper to be the only key meant a chatty response lost every
    real field and produced six "Field required" errors that said nothing about
    the actual problem.
    """
    for wrapper in _WRAPPER_KEYS:
        inner = data.get(wrapper)
        if not isinstance(inner, dict):
            continue
        inner_keys = {str(k).strip().lower() for k in inner}
        if not (inner_keys & ALLOWED_FIELDS):
            continue
        siblings = sorted(set(data) - {wrapper})
        notes.append(f"unwrapped '{wrapper}' object")
        if siblings:
            notes.append(f"discarded sibling key(s) of the wrapper: {', '.join(siblings)}")
        return {str(k).strip().lower(): v for k, v in inner.items()}
    return data


def normalise_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Canonicalise a raw payload before strict validation.

    Returns the cleaned payload and notes describing what had to be fixed.
    """
    notes: list[str] = []
    data = {str(k).strip().lower(): v for k, v in payload.items()}
    data = _unwrap(data, notes)

    if (
        isinstance(data.get("target_department"), str)
        and _as_text(data["target_department"]) in _NULL_WORDS
    ):
        data["target_department"] = None

    enum_fields: list[tuple[str, type[Enum], dict[str, Any]]] = [
        ("category", Category, _CATEGORY_ALIASES),
        ("priority", Priority, _PRIORITY_ALIASES),
        ("target_department", Department, _DEPARTMENT_ALIASES),
        ("work_item_type", WorkItemType, _WORK_ITEM_ALIASES),
        ("language", Language, _LANGUAGE_ALIASES),
    ]
    for key, enum_cls, aliases in enum_fields:
        if key not in data:
            continue
        original = data[key]
        coerced = _coerce_enum(original, enum_cls, aliases)
        # An exact match is not a repair, so it is not worth a note.
        exact = isinstance(coerced, enum_cls) and _as_text(original) == str(coerced.value).lower()
        if coerced is not original and not exact:
            notes.append(f"normalised {key}={original!r}")
        data[key] = coerced

    # An unknown department is a normalisation problem, not a reason to lose the
    # record: the report needs stable buckets more than the model's improvised
    # label.
    dept = data.get("target_department")
    if dept is not None and not isinstance(dept, Department):
        notes.append(f"unknown department {dept!r} mapped to '{Department.OTHER.value}'")
        data["target_department"] = Department.OTHER

    for key in ("needs_clarification", "is_recurring"):
        if key in data:
            original = data[key]
            coerced = _coerce_bool(original)
            if coerced is not original:
                notes.append(f"normalised {key}={original!r}")
            data[key] = coerced

    for key in _LIST_FIELDS:
        if key in data:
            original = data[key]
            coerced = _coerce_str_list(original)
            if type(coerced) is not type(original):
                notes.append(f"normalised {key} to a list")
            data[key] = coerced
        else:
            data[key] = []

    if isinstance(data.get("short_summary"), str):
        data["short_summary"] = " ".join(data["short_summary"].split())

    unexpected = sorted(set(data) - ALLOWED_FIELDS)
    for key in unexpected:
        data.pop(key)
    if unexpected:
        notes.append(f"dropped unexpected field(s): {', '.join(unexpected)}")

    return data, notes
