"""The pipeline: prompt, validate, repair, give up cleanly.

The order of defence matters and is the whole point of the exercise:

1. ask for JSON and pin the temperature;
2. recover the object from whatever came back;
3. normalise known-loose values;
4. validate strictly, and on failure send the validator's message back to the
   model as a repair prompt;
5. after the attempt budget is spent, write a ``failed`` record and move on.

Step 5 is the one that keeps a bad row from taking the run down with it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import ValidationError

from .llm.base import LLMClient, LLMError
from .llm.http_client import redact
from .models import (
    InboxRow,
    RecordStatus,
    RequestExtraction,
    TokenUsage,
    TriageRecord,
)
from .parsing import ParseError, extract_json_object, normalise_payload
from .prompts import (
    SYSTEM_INSTRUCTION,
    build_extraction_prompt,
    build_repair_prompt,
)
from .rules import apply_rules

MAX_RAW_RESPONSE_CHARS = 1200


def format_validation_error(exc: ValidationError) -> str:
    """Compact, quotable version of a Pydantic error.

    The default rendering carries urls and input echoes that make the repair
    prompt long and the model distracted.
    """
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "(root)"
        parts.append(f"поле '{location}': {error['msg']}")
    return "; ".join(parts[:8])


@dataclass
class ExtractionOutcome:
    extraction: RequestExtraction | None
    attempts: int
    error: str | None = None
    raw_response: str | None = None
    notes: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0


def extract_one(client: LLMClient, row: InboxRow, *, max_attempts: int = 3) -> ExtractionOutcome:
    prompt = build_extraction_prompt(
        request_id=row.id, channel=row.channel, timestamp=row.timestamp, text=row.raw_text
    )
    usage = TokenUsage()
    notes: list[str] = []
    last_text = ""
    last_error = "no attempt was made"
    calls = 0

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.generate(prompt, system=SYSTEM_INSTRUCTION)
        except LLMError as exc:
            # Transport retries already happened inside the client; a failure
            # here is final for this row.
            return ExtractionOutcome(
                extraction=None,
                attempts=attempt,
                error=f"llm call failed: {exc}",
                raw_response=last_text[:MAX_RAW_RESPONSE_CHARS] or None,
                notes=notes,
                usage=usage,
                llm_calls=calls,
            )
        except Exception as exc:
            # Anything the client did not translate into LLMError: a gateway
            # returning a shape nobody expected, a bug in this code. It costs
            # one row, not the run, because losing seventeen good rows to one
            # surprise is the worse outcome. Redacted like every other error
            # text, since this one also lands in output.json.
            return ExtractionOutcome(
                extraction=None,
                attempts=attempt,
                error=redact(f"unexpected client error: {type(exc).__name__}: {exc}"),
                raw_response=last_text[:MAX_RAW_RESPONSE_CHARS] or None,
                notes=notes,
                usage=usage,
                llm_calls=calls,
            )

        calls += 1
        usage.add(response.usage)
        last_text = response.text

        try:
            payload = extract_json_object(response.text)
        except ParseError as exc:
            last_error = f"відповідь не є валідним JSON: {exc}"
        else:
            payload, fix_notes = normalise_payload(payload)
            try:
                extraction = RequestExtraction.model_validate(payload)
            except ValidationError as exc:
                last_error = format_validation_error(exc)
            else:
                if attempt > 1:
                    notes.append(f"recovered after {attempt} attempts")
                notes.extend(fix_notes)
                return ExtractionOutcome(
                    extraction=extraction,
                    attempts=attempt,
                    notes=notes,
                    usage=usage,
                    llm_calls=calls,
                )

        if attempt < max_attempts:
            prompt = build_repair_prompt(
                previous_response=response.text,
                error=last_error,
                request_id=row.id,
                channel=row.channel,
                timestamp=row.timestamp,
                text=row.raw_text,
            )

    return ExtractionOutcome(
        extraction=None,
        attempts=max_attempts,
        error=last_error,
        raw_response=last_text[:MAX_RAW_RESPONSE_CHARS] or None,
        notes=notes,
        usage=usage,
        llm_calls=calls,
    )


def build_record(row: InboxRow, outcome: ExtractionOutcome) -> TriageRecord:
    if outcome.extraction is None:
        return TriageRecord(
            id=row.id,
            channel=row.channel,
            timestamp=row.timestamp,
            raw_text=row.raw_text,
            status=RecordStatus.FAILED,
            attempts=outcome.attempts,
            error=outcome.error,
            raw_response=outcome.raw_response,
            rule_flags=list(outcome.notes),
        )

    try:
        corrected, flags = apply_rules(outcome.extraction, row.raw_text)
    except Exception as exc:
        # A broken rule must not discard an extraction that already validated.
        corrected, flags = outcome.extraction, [f"rules_failed: {type(exc).__name__}"]

    return TriageRecord(
        id=row.id,
        channel=row.channel,
        timestamp=row.timestamp,
        raw_text=row.raw_text,
        status=RecordStatus.OK,
        extraction=corrected,
        attempts=outcome.attempts,
        rule_flags=[*outcome.notes, *flags],
    )


@dataclass
class TriageRun:
    records: list[TriageRecord]
    usage: TokenUsage
    llm_calls: int
    retried: int


def triage_rows(
    client: LLMClient,
    rows: list[InboxRow],
    *,
    max_attempts: int = 3,
    concurrency: int = 1,
) -> TriageRun:
    """Process every row, preserving input order regardless of concurrency."""

    def work(row: InboxRow) -> tuple[InboxRow, ExtractionOutcome]:
        try:
            return row, extract_one(client, row, max_attempts=max_attempts)
        except Exception as exc:
            # ThreadPoolExecutor.map re-raises the first exception and abandons
            # the rows still in flight, so nothing may escape this function.
            return row, ExtractionOutcome(
                extraction=None,
                attempts=0,
                error=f"unexpected error while processing the row: {type(exc).__name__}: {exc}",
            )

    if concurrency > 1 and len(rows) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(work, rows))
    else:
        results = [work(row) for row in rows]

    usage = TokenUsage()
    calls = 0
    retried = 0
    records: list[TriageRecord] = []
    for row, outcome in results:
        usage.add(outcome.usage)
        calls += outcome.llm_calls
        if outcome.attempts > 1:
            retried += 1
        records.append(build_record(row, outcome))

    return TriageRun(records=records, usage=usage, llm_calls=calls, retried=retried)
