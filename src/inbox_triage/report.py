"""Markdown summary of a run.

Written for the person who has to decide what the AI unit picks up on Monday,
so it leads with what is actually workable and keeps the failures visible
instead of tucking them into a log nobody opens.
"""

from __future__ import annotations

from collections import Counter

from .aggregate import Aggregates
from .dedup import DedupResult
from .models import RunMetadata, TriageRecord

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _cell(text: str) -> str:
    """Neutralise markdown in anything that goes into a table cell.

    Applies to input as well as model output: `channel` is a free column of the
    incoming CSV, and one pipe there splits the row.
    """
    return text.replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")


def _counter_table(title: str, counter: Counter[str], total: int) -> list[str]:
    if not counter:
        return [f"### {title}", "", "_немає даних_", ""]
    lines = [f"### {title}", "", "| Значення | Запитів | Частка |", "|---|---:|---:|"]
    for key, count in counter.most_common():
        share = f"{(count / total * 100):.0f}%" if total else "-"
        lines.append(f"| {_cell(key)} | {count} | {share} |")
    lines.append("")
    return lines


def _priority_table(counter: Counter[str], total: int) -> list[str]:
    if not counter:
        return ["### За пріоритетом", "", "_немає даних_", ""]
    lines = ["### За пріоритетом", "", "| Пріоритет | Запитів | Частка |", "|---|---:|---:|"]
    for key in sorted(counter, key=lambda k: _PRIORITY_ORDER.get(k, 99)):
        count = counter[key]
        share = f"{(count / total * 100):.0f}%" if total else "-"
        lines.append(f"| {key} | {count} | {share} |")
    lines.append("")
    return lines


def _short(text: str, limit: int = 90) -> str:
    """Collapse whitespace, clip, and neutralise markdown that came from a model.

    A pipe in a summary splits a three-column row into five cells, and a
    bracketed span turns into a link nobody put there.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return _cell(cleaned)


def _record_list(records: list[TriageRecord]) -> list[str]:
    lines = ["| ID | Канал | Суть | Що запитати |", "|---|---|---|---|"]
    for record in records:
        summary = record.extraction.short_summary if record.extraction else record.raw_text
        questions = ""
        if record.extraction and record.extraction.clarification_questions:
            questions = "; ".join(_short(q, 60) for q in record.extraction.clarification_questions)
        lines.append(
            f"| {_cell(record.id)} | {_cell(record.channel)} | {_short(summary)} | {questions} |"
        )
    lines.append("")
    return lines


def _work_queue(records: list[TriageRecord]) -> list[str]:
    """The one question the report exists to answer: what to pick up first."""
    if not records:
        return ["### Черга робіт", "", "_порожня_", ""]

    ordered = sorted(
        records,
        key=lambda r: _PRIORITY_ORDER.get(r.extraction.priority.value, 99) if r.extraction else 99,
    )
    lines = [
        "### Черга робіт",
        "",
        "Запити, готові до роботи: не дублікати, не питання, не потребують уточнення.",
        "",
        "| ID | Пріоритет | Тип | Суть | Системи |",
        "|---|---|---|---|---|",
    ]
    for record in ordered:
        extraction = record.extraction
        if extraction is None:
            continue
        systems = ", ".join(extraction.mentioned_systems) or "-"
        lines.append(
            f"| {_cell(record.id)} | {extraction.priority.value} | "
            f"{extraction.work_item_type.value} | {_short(extraction.short_summary, 70)} | "
            f"{_short(systems, 40)} |"
        )
    lines.append("")
    return lines


def render_report(
    aggregates: Aggregates,
    run: RunMetadata,
    *,
    dedup: DedupResult | None = None,
    skipped_rows: list[tuple[int, str]] | None = None,
) -> str:
    total = aggregates.succeeded
    lines: list[str] = [
        "# Звіт по інбоксу запитів",
        "",
        f"Оброблено запитів: **{aggregates.total}**  ",
        f"Розібрано успішно: **{aggregates.succeeded}**  ",
        f"Не вдалося розібрати: **{aggregates.failed}**  ",
        f"Готові до роботи (без дублів і уточнень): **{aggregates.actionable_count}**",
        "",
    ]

    lines += _work_queue(aggregates.actionable)
    lines += _counter_table("За категорією", aggregates.by_category, total)
    lines += _priority_table(aggregates.by_priority, total)
    lines += _counter_table("За відділом", aggregates.by_department, total)
    lines += _counter_table("За типом роботи", aggregates.by_work_item_type, total)

    lines += ["### Потребують уточнення", ""]
    if aggregates.needs_clarification:
        lines.append(
            f"{len(aggregates.needs_clarification)} запит(ів) не можна брати в роботу як є."
        )
        lines.append("")
        lines += _record_list(aggregates.needs_clarification)
    else:
        lines += ["_немає_", ""]

    lines += ["### Дублікати", ""]
    if aggregates.duplicates:
        lines += ["| ID | Дублює | Чому |", "|---|---|---|"]
        for record in aggregates.duplicates:
            reason = ""
            if dedup and record.id in dedup.reasons:
                reason = _short(dedup.reasons[record.id], 70)
            lines.append(f"| {_cell(record.id)} | {_cell(record.duplicate_of or '')} | {reason} |")
        lines.append("")
    else:
        lines += ["_не знайдено_", ""]

    if aggregates.failures:
        lines += ["### Не вдалося розібрати", ""]
        lines += ["| ID | Спроб | Помилка |", "|---|---:|---|"]
        for record in aggregates.failures:
            lines.append(
                f"| {_cell(record.id)} | {record.attempts} | {_short(record.error or '', 70)} |"
            )
        lines.append("")

    if aggregates.by_flag:
        lines += [
            "### Спрацювали перевірки поверх схеми",
            "",
            "Нормалізація, повторні спроби і детерміновані правила. Це не помилки "
            "обробки, а місця, де вивід моделі довелося правити або де він "
            "розійшовся з текстом запиту.",
            "",
        ]
        lines += ["| Правило | Спрацювань |", "|---|---:|"]
        for flag, count in aggregates.by_flag.most_common():
            lines.append(f"| `{flag}` | {count} |")
        lines.append("")

    if skipped_rows:
        lines += ["### Пропущені рядки вхідного файлу", ""]
        lines += ["| Рядок | Причина |", "|---|---|"]
        for line_no, reason in skipped_rows:
            lines.append(f"| {line_no} | {reason} |")
        lines.append("")

    if dedup and (dedup.error or dedup.rejected):
        lines += ["### Нотатки проходу по дублікатах", ""]
        if dedup.error:
            lines += [f"- {dedup.error}"]
        for note in dedup.rejected:
            lines += [f"- відхилено: {note}"]
        lines.append("")

    lines += [
        "---",
        "",
        "### Параметри запуску",
        "",
        f"- провайдер: `{run.provider}`, модель: `{run.model}`, temperature: `{run.temperature}`",
        f"- версія промпту: `{run.prompt_version}`",
        f"- звернень до моделі: {run.llm_calls} (з кешу: {run.cache_hits})",
        f"- запитів із повторною спробою: {run.retried} із {run.total_requests}",
        f"- токени: {run.token_usage.prompt_tokens} вхідних + "
        f"{run.token_usage.completion_tokens} вихідних = {run.token_usage.total}",
        f"- запуск: {run.started_at} → {run.finished_at}",
        "",
    ]
    return "\n".join(lines)
