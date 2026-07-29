# Inbox Triage

Reads an export of internal requests (`input_requests.csv`), extracts structured
fields from each one with an LLM, checks the model's output with deterministic
code, and writes `output.json` and `report.md`.

The idea behind it: the model does the judgement and the language, and anything
that has to be exact lives in code and is covered by tests. So there are two
checks rather than one. The schema catches the shape of the answer, separate
rules catch the content.

Output of a real run over the 18 supplied rows is in [`examples/`](examples/).
The report itself is written in Ukrainian, since that is the language of the
inbox and of the people who read it.

**Short on time?** Three files carry the idea:

- [`src/inbox_triage/triage.py`](src/inbox_triage/triage.py) - the loop that
  validates the model's answer and, when it fails, sends the validator's own
  message back as a repair prompt. When the attempts run out the row becomes a
  `failed` record instead of an exception.
- [`src/inbox_triage/rules.py`](src/inbox_triage/rules.py) - deterministic
  checks that compare the model against the request text: an urgency quote must
  exist in the original, a department is checked in three directions.
- [`examples/report.md`](examples/report.md) - what the run actually produced:
  0 failed, 2 repairs, one duplicate found, 32,243 tokens.

---

## Time and tooling

The brief suggests 2-3 hours. It took me longer: the evening I received it and
the following day. The extraction itself was not the slow part. The limits
were: what to do when the model returns output that is valid but wrong, and how
to show that to the person who reads the report.

I wrote it with Claude Code, the same tool you asked about at the first stage.
The decisions about the schema, the deterministic rules and which rows of the
dataset are traps are mine; the code and the tests we wrote together, and then I
ran reviews and live runs and fixed what they showed. Two of those runs changed
the solution: the first found no duplicates at all, because the pass only saw
one-line summaries instead of the text, and the second showed the model
inventing a department where the text names none. Both fixes are in the
history.

---

## Running it

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env             # then put a key in it
python -m inbox_triage
```

Results appear in `output/output.json` and `output/report.md`.

### Provider

The provider is not baked into the code. Anything that speaks the OpenAI
`/chat/completions` shape works: pick `LLM_PROVIDER` and put its key in
`LLM_API_KEY`.

| `LLM_PROVIDER` | Key | Default model |
|---|---|---|
| `openrouter` | https://openrouter.ai/keys | `google/gemini-2.5-flash` |
| `openai` | https://platform.openai.com/api-keys | `gpt-4o-mini` |
| `gemini` | https://aistudio.google.com/apikey (free tier is enough) | `gemini-2.5-flash` |
| `groq`, `together` | their own consoles | see `llm/http_client.py` |
| anything else | - | set `LLM_BASE_URL` |

### Environment variables

| Variable | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `openrouter` | preset base URL and model |
| `LLM_API_KEY` | required | the provider's own name works too: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `GROQ_API_KEY`, `TOGETHER_API_KEY` |
| `LLM_MODEL` | provider preset | any model available on that key |
| `LLM_BASE_URL` | provider preset | for a gateway that is not in the list |
| `LLM_TEMPERATURE` | `0` | see the section on non-determinism |
| `LLM_MAX_ATTEMPTS` | `3` | repair attempts per row when the output is invalid |
| `LLM_CONCURRENCY` | `4` | parallel requests |
| `LLM_JSON_MODE` | `true` | ask the API for strict JSON; dropped automatically if the model rejects it |
| `LLM_CACHE_DIR` | `.cache/llm` | response cache, empty value disables it |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | - | only for `--telegram` |

### CLI

```bash
python -m inbox_triage --input data/input_requests.csv --output-dir output
python -m inbox_triage --provider openai --model gpt-4o-mini
python -m inbox_triage --limit 3            # first 3 rows only
python -m inbox_triage --concurrency 1      # sequential
python -m inbox_triage --no-cache           # ignore the cache
python -m inbox_triage --no-dedup           # skip the cross-request pass
python -m inbox_triage --max-attempts 5
python -m inbox_triage --telegram           # send a digest
python -m inbox_triage --verbose            # debug logging
python -m inbox_triage --quiet              # errors only
```

Exit codes: `0` when every request was parsed, `1` when at least one ended up
`failed`, `2` when the input file or the key is unusable. Files are written in
every case except the last, and written atomically: an interrupted run cannot
leave a truncated `output.json` where the previous good one was.

### Telegram digest

Optional, off unless `--telegram` is passed. It posts a short summary of the run
to a chat: how many requests, how many need clarification, which ones are high
priority, and what the run cost in tokens.

Two values are read from the environment, nothing is hardcoded:

```bash
TELEGRAM_BOT_TOKEN=...   # from @BotFather: /newbot
TELEGRAM_CHAT_ID=...     # any chat the bot can post to
```

To find a chat id: send the bot any message, then open
`https://api.telegram.org/bot<TOKEN>/getUpdates` and read `result[].message.chat.id`.
A private chat gives a positive number, a group gives a negative one; both work.

Delivery is best effort. If the token is wrong, the chat is unreachable or the
network is down, the run logs a warning and still writes its files: the files
are the deliverable, this is a convenience on top.

### Docker

```bash
docker build -t inbox-triage .
docker run --rm --env-file .env -v "$PWD/output:/app/output" inbox-triage
```

### Tests

```bash
pytest
```

The whole suite is **offline**: no key, no network. A scripted fake stands in
for the LLM and returns, on demand, truncated JSON, an invented category, an
answer wrapped in markdown, or plain prose. The HTTP client is driven through a
mock transport: 429 with backoff, a dropped connection, a timeout, 401, an
unknown model, an error object arriving with status 200. `tests/test_edge_cases.py`
holds what breaks on malformed input, hostile text and strange provider
responses.

Static checks: `ruff check .`, `ruff format --check .`, `mypy` (strict). CI runs
the same on Python 3.11 and 3.12.

---

## Output

**`output.json`** carries `schema_version`, a `run` block with the parameters of
the run (provider, model, temperature, prompt version, tokens, how many requests
had to be retried) and a `records` array. Every record keeps the original row,
the structured output, the rules that fired and, when it did not work out, the
error and a truncated raw response from the model.

**`report.md`** leads with a **work queue** (what to pick up first, by priority),
then aggregates by category, priority, department and work type; the list of
requests that need clarification **together with the questions to ask their
authors**; duplicates found; failures; and a table of checks that fired.

---

## How it works

```
CSV
 │  csv_loader: validate the file, skipped rows with a reason
 ▼
LLM (temperature 0, response_format=json_object)
 │  disk cache keyed by sha256(provider + model + parameters + prompt)
 ▼
parsing: recover JSON from the text, normalise known variations
 ▼
models: strict Pydantic schema
 │  ✗ ──► repair prompt quoting the validator ──► another attempt
 │              ✗ budget spent ──► failed record, the run continues
 ▼
rules: deterministic checks on the content
 ▼
dedup: a separate pass over the whole set
 ▼
output.json + report.md
```

**Recovering JSON is separate from validating it.** The model returns the object
inside a code fence, with a sentence in front of it, sometimes with a curly
brace inside the prose. The scan walks every balanced object, respecting quotes
and escapes, and prefers the one that actually carries schema fields.

**Normalisation happens before the schema, not inside it.** `"Автоматизация"`
instead of `"автоматизація"`, `"true"` instead of `true`, a bare string where a
list was asked for: known defects, not worth spending an attempt on. Every fix
is recorded in `rule_flags`, so it is visible how much repair a run needed. An
unrecognised value is **left alone** on purpose, so the schema rejects it and
the repair prompt can quote a real error. The one exception is
`target_department`, where a stable aggregate matters more than the exact label.

**The repair prompt quotes the validator** and repeats the original request. If
the first answer was an apology rather than JSON there is nothing to patch, and
the model has to answer from scratch.

**A failure is a record, not an exception.** Once the attempts are spent the row
is stored with status `failed`, the error and the raw response. One bad row does
not stop the rest, and that holds for more than validation errors: any
unexpected exception from the client costs one row rather than the run.

---

## Schema

Required by the brief: `category`, `target_department`, `priority`,
`short_summary`, `requested_actions`, `needs_clarification`.

Extensions. Each one is here because a specific row of the dataset is handled
wrongly without it.

| Field | Why | Which row shows it |
|---|---|---|
| `work_item_type` | `category` says **what the request is about**, this says **what it turns into on our side** | REQ-005 is an urgent one-off export with nothing to automate. REQ-007 is an existing pipeline that broke, not new development. REQ-008 is a thank-you note. Without this field all three land in the backlog as projects |
| `clarification_questions` | the `needs_clarification` flag alone does not say **what** to ask, so the reader opens the original anyway | REQ-002 "хлопці треба бот", REQ-011 "нам би табличку якусь" |
| `mentioned_systems` | who owns the task, and what to match duplicates on | Google Ads, PlanFix, BigQuery, Meta |
| `urgency_signals` | verbatim quotes justifying `priority`, which makes it **checkable** | REQ-005 "ГОРИТЬ", REQ-014 "не горить" |
| `duplicate_of` | set by a separate pass, cannot be derived from one request | REQ-013 is the same report as REQ-001 |

Something reads every one of these: the rules, the report or the duplicate pass.
Fields that feed nothing do not stay in the schema. They cost tokens on every
request and give the model one more thing to get wrong.

Deliberately **absent**: a `confidence` field. LLM self-assessment is poorly
calibrated, and the model is equally sure when it is right and when it is not.
The quotes checked against the source, and the rules below, do that job instead.

**On `target_department`.** The field means the department that **asked**, not
the one that will do the work. Who to assign is the unit's decision; who asked
is a fact visible in the text and checkable by a rule. The list is closed (10
values including `інше`), because a free string would let the model invent a new
label every run and break the aggregate in the report. An unrecognised value is
mapped to `інше` with a note in `rule_flags`. On the supplied data the field is
empty for most requests, simply because most authors never name their
department, and guessing it from the topic would be worse than an honest `null`.

---

## Deterministic rules

They run after the schema, in ordinary code, with fixed thresholds. The same
input gives the same result regardless of the model.

Some of them check the model **against the request text** rather than against
itself. That is the cheapest hallucination check available, and it costs no
extra call:

- **`urgency_signals` must occur in the text.** A quote that is not in the
  original is removed. Negation is handled: "не горить" and even "не дуже
  терміново" do not read as urgency.
- **The department is checked both ways.** Named in the text but the field is
  empty: flag. Field filled but the text names no department: also a flag. On
  the first live run "хлопці треба бот" came back as `it/підтримка`.
- **Mentioned systems** are matched allowing for inflection, because Ukrainian
  writes "з Мети" and "слака з планфіксом".

The rest catch internal contradictions: `high` with nothing in the text to
support it; a task to be picked up with no action written down;
`needs_clarification` with no questions; a thank-you note with a to-do list;
"out of scope" queued as a project; text too short to be a complete request.

Rules **correct only where the output contradicts itself** and one side is
clearly wrong. There are three such places: `needs_clarification_forced_true`
(the request is queued but nobody wrote what to do), removing a quote that is
not in the text, and mapping an unknown department to `інше`. Everywhere else
they only flag: quietly rewriting something the model may have understood better
is worse than a visible mark.

The duplicate pass is checked in code as well: both ids must exist, the original
must be earlier **by timestamp**, self-references and chains are rejected.
Rejected pairs go into the report rather than disappearing.

### Instructions hidden in the text

Requests are free-form and written by people, and a forwarded client email lands
in the same inbox, so the text is not trusted input:

- both prompts that see user text (the per-request extraction and the duplicate
  pass) receive it **serialised as JSON** rather than glued into a string, so a
  quote inside cannot break the structure, and the system instruction states
  outright that this is data and not commands;
- text longer than 8000 characters is truncated;
- phrases like "ignore previous instructions" raise a
  `possible_prompt_injection_in_text` flag. A mark, not a block: in an internal
  inbox a false positive costs more than a miss;
- model output is escaped before it is rendered into markdown, otherwise a `|`
  breaks the table and `[text](link)` puts a clickable link in the report that
  nobody placed there.

Worth saying what this does **not** buy: closed enums and the schema stop the
model returning anything off-list, but they do not stop it setting
`priority: high` because the author asked it to. What catches that is not the
schema but the rule about a missing urgency marker in the text.

The provider key never reaches the output. Error messages go through a redactor
that strips the configured key by value as well as by pattern, and a 401 body is
not quoted at all, since that is where providers most often echo the key back.

---

## Where this breaks

### Invalid model output

**Covered.** JSON requested at the API level, object recovered from arbitrary
text, known variations normalised, strict schema with repair attempts, and after
that a `failed` record with the raw response, without taking the run down.

**Not covered:** a model that consistently returns output that is **valid but
wrong**. The schema cannot tell `priority: "low"` from `"high"` when both are
formally allowed. The rules catch only the part that is mechanically visible in
the text.

### Classification quality is not measured

The main limitation. There is no hand-labelled reference set, so it is unknown
in what share of cases `category` and `priority` match what a person would
choose. All that can be claimed is that the output is valid, internally
consistent and backed by quotes from the source. That is not the same as
correct.

What the run in `examples/` shows: over 18 rows nothing ended up `failed`, two
answers needed repairing and went through on the second attempt, and the
REQ-013 duplicate was found. The rules caught three places where the output
contradicted itself: two requests queued for work with no action written down,
and a `non_actionable` item carrying a list of actions. But that is one run on
one model, not a measurement.

How to close it: label 100-150 requests by hand, compute accuracy per field and
a confusion matrix over the categories. Without that, any number about
"accuracy" would be invented.

### Non-determinism

`temperature=0` narrows the spread but does **not** guarantee identical output:
a provider can change the model behind the same name, and a gateway like
OpenRouter can route the same request to different upstreams.

Done: a response cache keyed by `provider + model + parameters + prompt`. Re-running
the same input gives the same result and costs no tokens. The model, prompt
version and temperature are written into `output.json`, so two runs are only
comparable when those match.

What it does **not** solve: the first run over new input is still
non-deterministic, and deleting the cache brings the spread back.

### Volume

18 rows is a scale at which everything is easy. What changes at 10,000:

- **Rate limits.** There is a bounded thread pool and backoff on 429, but no
  queue and no spreading of load over time. A run will hit the provider's RPM.
- **Memory.** Every record is held in a list and serialised as one file. The
  right shape is streaming with JSON Lines written as rows complete.
- **No checkpoints.** A break at row 9,000 means running again from scratch. The
  cache softens it, but that is a side effect and not a recovery mechanism.
- **Duplicates.** The pass sends the whole set in one prompt. At 10,000 requests
  that will not fit in context. The right shape is candidate selection first, by
  embeddings or by overlap in `mentioned_systems`, and only then an LLM over
  dozens of pairs.

### Token cost

Counted and reported in `report.md` and `output.json` (`run.token_usage`).
From the run in `examples/`: 18 requests, 21 calls (18 extractions, 2 repair
attempts and the duplicate pass), 32,243 tokens of which 28,974 are input. So
the main cost is the instructions repeated on every request.

How to bring that down at scale: move the rules into the system instruction and
use provider-side context caching; batch several requests into one call; shorten
the example in the prompt. None of it is done: at 18 rows it is premature, and
without a reference set there is no way to verify that a shorter prompt did not
make the output worse.

### Privacy

Request texts go to an external API. This dataset has no personal data; a real
inbox will have it (names, amounts, counterparties). That needs either a masking
layer or a model inside your own perimeter. Not done.

### Async

Processing is parallel through a thread pool rather than `asyncio`. The work is
IO-bound, and at this scale a pool gives the same throughput for less
complexity; `asyncio` earns its keep at thousands of concurrent connections,
which are not here.

---

## What I would do next

1. **A reference set of 100-150 labelled requests** and accuracy measured per
   field. Without it everything else, prompt tuning included, is done blind.
2. **Streaming processing** with JSON Lines and checkpoints.
3. **Duplicate detection through embeddings**, with the LLM only on the final
   comparison of candidate pairs.
4. **Google Sheets.** Not done: it needs a service account, a separate
   credentials file and an OAuth scope, and the summary is already on disk. The
   extension point is the `Sink` protocol in `delivery.py`: a Sheets
   implementation builds rows from the aggregates and appends them to a
   worksheet, the way `TelegramSink` posts a digest.
5. **A webhook instead of a CSV.** In real life requests arrive one at a time,
   and batch processing a file is just a convenient shape for a test task.
