"""Counting. No model involved, so these numbers are exact."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .models import RecordStatus, TriageRecord


@dataclass
class Aggregates:
    total: int = 0
    succeeded: int = 0
    failed: int = 0

    by_category: Counter[str] = field(default_factory=Counter)
    by_priority: Counter[str] = field(default_factory=Counter)
    by_department: Counter[str] = field(default_factory=Counter)
    by_work_item_type: Counter[str] = field(default_factory=Counter)
    by_channel: Counter[str] = field(default_factory=Counter)
    by_flag: Counter[str] = field(default_factory=Counter)

    needs_clarification: list[TriageRecord] = field(default_factory=list)
    duplicates: list[TriageRecord] = field(default_factory=list)
    failures: list[TriageRecord] = field(default_factory=list)
    actionable: list[TriageRecord] = field(default_factory=list)

    @property
    def actionable_count(self) -> int:
        return len(self.actionable)


UNKNOWN_DEPARTMENT = "не визначено"
ACTIONABLE_TYPES = {"project", "one_off", "incident"}


def aggregate(records: list[TriageRecord]) -> Aggregates:
    result = Aggregates(total=len(records))

    for record in records:
        result.by_channel[record.channel or "невідомо"] += 1

        for flag in record.rule_flags:
            result.by_flag[flag] += 1

        if record.status is not RecordStatus.OK or record.extraction is None:
            result.failed += 1
            result.failures.append(record)
            continue

        result.succeeded += 1
        extraction = record.extraction
        result.by_category[extraction.category.value] += 1
        result.by_priority[extraction.priority.value] += 1
        result.by_department[
            extraction.target_department.value
            if extraction.target_department
            else UNKNOWN_DEPARTMENT
        ] += 1
        result.by_work_item_type[extraction.work_item_type.value] += 1

        if extraction.needs_clarification:
            result.needs_clarification.append(record)
        if record.duplicate_of:
            result.duplicates.append(record)
        elif extraction.work_item_type.value in ACTIONABLE_TYPES:
            result.actionable.append(record)

    return result
