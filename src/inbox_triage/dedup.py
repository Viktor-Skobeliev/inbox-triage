"""Cross-request duplicate pass.

Kept separate from extraction for a reason worth stating: a per-request call
cannot possibly know that REQ-013 is a repeat, because the text only says "той
самий звіт, що Оля просила в понеділок". The duplicate exists between rows, so
it needs a pass that sees all of them.

One extra call for the whole file, and every pair the model proposes is checked
in code: both ids must exist, the original must be older, no self-reference, no
chains. A model that invents a pair gets it dropped, not written to the output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from .llm.base import LLMClient, LLMError
from .models import RecordStatus, TokenUsage, TriageRecord
from .parsing import ParseError, extract_json_object
from .prompts import DEDUP_SYSTEM_INSTRUCTION, build_dedup_prompt


@dataclass
class DedupResult:
    pairs: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0


TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M")


def _parse_timestamp(raw: str) -> float | None:
    """Seconds since the epoch, or None when the format is unknown."""
    stamp = (raw or "").strip()
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(stamp, fmt).timestamp()
        except ValueError:
            continue
    return None


MAX_TEXT_IN_BLOCK = 400


def build_summary_block(records: list[TriageRecord]) -> str:
    """One line per request, with the original text alongside the summary.

    The summary alone is not enough, and the supplied dataset shows why: what
    makes REQ-013 a duplicate is the phrase about the same report someone else
    asked for on Monday, and a one-sentence summary drops exactly that. The
    first live run found no duplicates at all for this reason.
    """
    items = []
    for record in records:
        if record.status is not RecordStatus.OK or record.extraction is None:
            continue
        items.append(
            {
                "id": record.id,
                "timestamp": record.timestamp,
                "systems": record.extraction.mentioned_systems,
                "summary": record.extraction.short_summary,
                "text": " ".join(record.raw_text.split())[:MAX_TEXT_IN_BLOCK],
            }
        )
    # Serialised, not interpolated: this prompt carries user text too, and it
    # was the one input still going in raw.
    return json.dumps(items, ensure_ascii=False, indent=1)


def find_duplicates(client: LLMClient, records: list[TriageRecord]) -> DedupResult:
    eligible = [r for r in records if r.status is RecordStatus.OK and r.extraction is not None]
    if len(eligible) < 2:
        return DedupResult()

    result = DedupResult()
    try:
        response = client.generate(
            build_dedup_prompt(build_summary_block(eligible)),
            system=DEDUP_SYSTEM_INSTRUCTION,
        )
    except LLMError as exc:
        result.error = f"duplicate pass skipped: {exc}"
        return result
    except Exception as exc:
        # This runs after every row has been paid for. Losing the whole run
        # here costs more than anywhere else in the pipeline.
        result.error = f"duplicate pass skipped: {type(exc).__name__}: {exc}"
        return result

    result.llm_calls = 1
    result.usage.add(response.usage)

    try:
        payload = extract_json_object(response.text)
    except ParseError as exc:
        result.error = f"duplicate pass returned unusable JSON: {exc}"
        return result

    raw_pairs = payload.get("duplicates")
    if not isinstance(raw_pairs, list):
        result.error = "duplicate pass returned no 'duplicates' list"
        return result

    position = {record.id: index for index, record in enumerate(eligible)}
    moment = {record.id: _parse_timestamp(record.timestamp) for record in eligible}
    accepted: dict[str, str] = {}

    for item in raw_pairs:
        if not isinstance(item, dict):
            result.rejected.append(f"not an object: {item!r}")
            continue
        later = str(item.get("id", "")).strip()
        earlier = str(item.get("duplicate_of", "")).strip()
        reason = str(item.get("reason", "")).strip()

        if later not in position or earlier not in position:
            result.rejected.append(f"unknown id in pair {later or '?'} -> {earlier or '?'}")
            continue
        if later == earlier:
            result.rejected.append(f"self-reference on {later}")
            continue

        # The original has to be the older message, otherwise the pair points
        # work at the wrong request. Compare timestamps when both parsed, and
        # fall back to position in the file when either did not: an unreadable
        # date is not evidence that the pair is backwards.
        earlier_at, later_at = moment[earlier], moment[later]
        if earlier_at is not None and later_at is not None:
            out_of_order = earlier_at > later_at
        else:
            out_of_order = position[earlier] > position[later]
        if out_of_order:
            result.rejected.append(f"{later} -> {earlier} points at a later request")
            continue
        if later in accepted:
            result.rejected.append(f"{later} is already matched with {accepted[later]}")
            continue
        if earlier in accepted:
            result.rejected.append(f"{later} -> {earlier} would chain duplicates")
            continue
        accepted[later] = earlier
        if reason:
            result.reasons[later] = reason

    # A duplicate that is itself marked as a duplicate target would create a
    # chain; drop those rather than resolve them silently.
    for later, earlier in list(accepted.items()):
        if earlier in accepted:
            result.rejected.append(f"{later} -> {earlier} would chain duplicates")
            accepted.pop(later, None)

    result.pairs = accepted
    return result


def apply_duplicates(records: list[TriageRecord], result: DedupResult) -> None:
    for record in records:
        target = result.pairs.get(record.id)
        if target:
            # Not added to rule_flags: duplicate_of is a field of its own, and
            # the flags table is about the model contradicting the text.
            record.duplicate_of = target
