"""Data model for the triage pipeline.

Two layers on purpose:

* ``RequestExtraction`` is what the model is allowed to produce: closed enums,
  no empty strings, bounded lengths. Unknown keys are dropped by
  ``normalise_payload`` before validation rather than rejected here.
* ``TriageRecord`` wraps the original row and carries everything the pipeline
  learned about it, including the failure path when extraction never succeeded.

Keeping them apart means a bad model response cannot corrupt the output file:
a record with ``status="failed"`` still has its id, channel and raw text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    """Fixed list from the task description."""

    AUTOMATION = "автоматизація"
    INTEGRATION = "інтеграція"
    ANALYTICS = "звіт/аналітика"
    SUPPORT = "баг/підтримка"
    QUESTION = "питання/консультація"
    OUT_OF_SCOPE = "поза скоупом"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Department(StrEnum):
    """Closed list with an explicit escape hatch.

    A free-form string here would let the model invent a new department name on
    every run and quietly break the aggregation in the report. A closed list
    plus ``OTHER`` keeps the report stable; anything unrecognised is normalised
    into ``OTHER`` and flagged instead of failing the record.
    """

    MARKETING = "маркетинг"
    SALES = "продажі"
    ANALYTICS = "аналітика"
    FINANCE = "фінанси/бухгалтерія"
    HR = "hr"
    CONTENT = "контент"
    SMM = "smm"
    PM = "pm"
    IT_SUPPORT = "it/підтримка"
    OTHER = "інше"


class WorkItemType(StrEnum):
    """What the AI unit is expected to *do*, which is not the same as what the
    request is *about*.

    ``category`` answers "what topic is this". This field answers "what does it
    turn into on our side", and the two genuinely diverge in the data:
    REQ-005 is an urgent one-off export (nothing to automate), REQ-007 is a
    broken existing pipeline (not new development), REQ-008 is a thank-you note.
    Without this field all three land in the backlog as if they were projects.
    """

    PROJECT = "project"
    ONE_OFF = "one_off"
    INCIDENT = "incident"
    QUESTION = "question"
    NON_ACTIONABLE = "non_actionable"


# Bounded on purpose: with no upper limit a runaway response becomes a
# multi-megabyte output.json that still validates. A cap turns it into an
# ordinary validation error, which the repair loop already handles.
NonEmptyStr = Annotated[str, Field(min_length=1, max_length=300)]
SummaryStr = Annotated[str, Field(min_length=1, max_length=400)]


class RequestExtraction(BaseModel):
    """Structured fields the LLM must return for a single request."""

    # extra="ignore" rather than "forbid": normalise_payload already drops
    # unknown keys and records what it dropped, and rejecting a whole record
    # over one stray field the model invented is a bad trade.
    model_config = ConfigDict(extra="ignore", use_enum_values=False, str_strip_whitespace=True)

    # --- required by the task ---
    category: Category
    target_department: Department | None
    priority: Priority
    short_summary: SummaryStr
    requested_actions: list[NonEmptyStr] = Field(default_factory=list, max_length=20)
    needs_clarification: bool

    # --- extensions, each earned by a specific row in the dataset ---
    work_item_type: WorkItemType
    mentioned_systems: list[NonEmptyStr] = Field(
        default_factory=list,
        max_length=20,
        description="Named products the work touches (Google Ads, PlanFix, BigQuery). "
        "Used to route to whoever owns the integration and to spot duplicates.",
    )
    urgency_signals: list[NonEmptyStr] = Field(
        default_factory=list,
        max_length=20,
        description="Verbatim fragments that justify the priority. Makes priority "
        "auditable instead of taking the model's word for it.",
    )
    clarification_questions: list[NonEmptyStr] = Field(
        default_factory=list,
        max_length=20,
        description="What to ask the author when the request cannot be worked as is. "
        "A bare needs_clarification flag makes the reader open the original text "
        "anyway; the questions make the report actionable.",
    )

    @field_validator(
        "requested_actions",
        "mentioned_systems",
        "urgency_signals",
        "clarification_questions",
        mode="after",
    )
    @classmethod
    def _drop_blanks_and_dupes(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            cleaned = value.strip()
            if not cleaned or cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            out.append(cleaned)
        return out


class InboxRow(BaseModel):
    """One row of the input CSV, before any model touches it."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: NonEmptyStr
    channel: str = ""
    timestamp: str = ""
    raw_text: str = ""


class RecordStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class TriageRecord(BaseModel):
    """Input row plus the result of trying to structure it."""

    id: str
    channel: str
    timestamp: str
    raw_text: str

    status: RecordStatus
    extraction: RequestExtraction | None = None

    duplicate_of: str | None = Field(
        default=None,
        description="Set by the cross-request pass, not by the per-request call: "
        "REQ-013 only reads as a duplicate once REQ-001 is also on the table.",
    )
    rule_flags: list[str] = Field(
        default_factory=list,
        description="Deterministic post-schema checks that fired for this record.",
    )
    attempts: int = 0
    error: str | None = None
    raw_response: str | None = Field(
        default=None,
        description="Truncated last model response, kept only for failed records so a "
        "human can see what actually came back.",
    )


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


class RunMetadata(BaseModel):
    """Provenance. Without it an output.json cannot be reproduced or argued with."""

    started_at: str
    finished_at: str
    provider: str
    base_url: str
    model: str
    temperature: float
    prompt_version: str
    max_attempts: int
    total_requests: int
    succeeded: int
    failed: int
    retried: int
    cache_hits: int
    llm_calls: int
    token_usage: TokenUsage


SCHEMA_VERSION = "1.0"


class TriageOutput(BaseModel):
    # The output format will change; a consumer needs to be able to tell which
    # version it is holding without guessing from the field names.
    schema_version: str = SCHEMA_VERSION
    run: RunMetadata
    records: list[TriageRecord]
