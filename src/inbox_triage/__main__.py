"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .aggregate import Aggregates, aggregate
from .config import load_settings
from .csv_loader import InputError, load_requests
from .dedup import DedupResult, apply_duplicates, find_duplicates
from .delivery import TelegramSink
from .llm.base import CachingClient, LLMClient, LLMError
from .llm.http_client import ChatClient, resolve_base_url, resolve_model
from .models import RunMetadata, TriageOutput
from .prompts import PROMPT_VERSION
from .report import render_report
from .triage import triage_rows

DEFAULT_INPUT = Path("data/input_requests.csv")
DEFAULT_OUTPUT_DIR = Path("output")

log = logging.getLogger("inbox_triage")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def positive_int(raw: str) -> int:
    """argparse type that rejects 0 and negatives instead of ignoring them."""
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {raw!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a number above zero, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inbox-triage",
        description="Structure free-form internal requests into a validated JSON file.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="input CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit", type=positive_int, default=None, help="process only the first N rows"
    )
    parser.add_argument("--provider", default=None, help="override LLM_PROVIDER")
    parser.add_argument("--model", default=None, help="override the model id")
    parser.add_argument("--concurrency", type=positive_int, default=None)
    parser.add_argument(
        "--max-attempts", type=positive_int, default=None, help="validation retries per row"
    )
    parser.add_argument("--no-dedup", action="store_true", help="skip the cross-request pass")
    parser.add_argument("--no-cache", action="store_true", help="always call the model")
    parser.add_argument(
        "--telegram", action="store_true", help="send a digest (needs TELEGRAM_* in .env)"
    )
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--verbose", action="store_true", help="debug logging")
    noise.add_argument("--quiet", action="store_true", help="errors only")
    return parser


def configure_logging(*, verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    # httpx logs every request at INFO, which buries the run's own output in a
    # wall of identical POST lines. It is also kept quiet under --verbose: the
    # Telegram URL carries the bot token, and httpx logs full URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def write_atomic(path: Path, content: str) -> None:
    """Write via a temporary file in the same directory, then replace.

    A run interrupted halfway through must not leave a truncated output.json
    where the previous good one used to be.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def maybe_send_digest(aggregates: Aggregates, metadata: RunMetadata) -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        log.warning("--telegram requested but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set")
        return
    TelegramSink(token, chat_id).publish(aggregates, metadata)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    settings = load_settings()

    provider = (args.provider or settings.provider).strip().lower()
    concurrency = args.concurrency or settings.concurrency
    max_attempts = args.max_attempts or settings.max_attempts

    try:
        loaded = load_requests(args.input)
    except InputError as exc:
        log.error("error: %s", exc)
        return 2

    rows = loaded.rows[: args.limit] if args.limit else loaded.rows
    log.info("loaded %d request(s) from %s", len(rows), args.input)
    for line_no, reason in loaded.skipped:
        log.info("  skipped line %d: %s", line_no, reason)

    # Before the first paid call, not after: an unusable output directory is an
    # input problem, and finding it out once the run is paid for is the worst
    # possible moment.
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if settings.cache_dir and not args.no_cache:
            settings.cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("error: cannot use output or cache directory: %s", exc)
        return 2

    if not settings.has_api_key:
        log.error(
            "error: no API key found.\n"
            "Copy .env.example to .env, pick LLM_PROVIDER and put its key in LLM_API_KEY.\n"
            "The test suite runs without a key: pytest"
        )
        return 2

    try:
        base_url = resolve_base_url(provider, settings.base_url)
        model = resolve_model(provider, args.model or settings.model)
    except LLMError as exc:
        log.error("error: %s", exc)
        return 2

    if not model:
        log.error("error: no model for provider '%s'. Set LLM_MODEL in .env.", provider)
        return 2

    chat = ChatClient(
        api_key=settings.api_key or "",
        model=model,
        base_url=base_url,
        temperature=settings.temperature,
        json_mode=settings.json_mode,
    )

    client: LLMClient = chat
    cache: CachingClient | None = None
    if settings.cache_dir and not args.no_cache:
        cache = CachingClient(
            client,
            settings.cache_dir,
            fingerprint=f"{base_url}|{model}|{settings.temperature}|{PROMPT_VERSION}",
        )
        client = cache

    started_at = _now()
    log.info("provider %s, model %s, temperature %s", provider, model, settings.temperature)

    dedup_result: DedupResult | None = None
    try:
        run = triage_rows(client, rows, max_attempts=max_attempts, concurrency=concurrency)

        if not args.no_dedup:
            try:
                dedup_result = find_duplicates(client, run.records)
                apply_duplicates(run.records, dedup_result)
            except Exception:
                # This step runs after every row has been paid for. Whatever
                # happened, the rows that did parse are still worth writing.
                log.exception("duplicate pass failed, continuing with what was parsed")
            else:
                if dedup_result.error:
                    log.warning("  %s", dedup_result.error)
                elif dedup_result.pairs:
                    log.info("  duplicates found: %d", len(dedup_result.pairs))
    finally:
        chat.close()

    usage = run.usage
    llm_calls = run.llm_calls
    if dedup_result:
        usage.add(dedup_result.usage)
        llm_calls += dedup_result.llm_calls

    aggregates = aggregate(run.records)
    metadata = RunMetadata(
        started_at=started_at,
        finished_at=_now(),
        provider=provider,
        base_url=base_url,
        model=model,
        temperature=settings.temperature,
        prompt_version=PROMPT_VERSION,
        max_attempts=max_attempts,
        total_requests=len(rows),
        succeeded=aggregates.succeeded,
        failed=aggregates.failed,
        retried=run.retried,
        cache_hits=cache.hits if cache else 0,
        llm_calls=llm_calls,
        token_usage=usage,
    )

    json_path = args.output_dir / "output.json"
    report_path = args.output_dir / "report.md"

    write_atomic(
        json_path, TriageOutput(run=metadata, records=run.records).model_dump_json(indent=2)
    )
    write_atomic(
        report_path,
        render_report(aggregates, metadata, dedup=dedup_result, skipped_rows=loaded.skipped),
    )

    if args.telegram:
        maybe_send_digest(aggregates, metadata)

    if aggregates.failed and aggregates.failed == aggregates.total:
        # A dead key and one bad row currently look identical in the log, and
        # under --quiet a fully failed run says nothing at all.
        reasons = Counter(record.error or "unknown" for record in aggregates.failures)
        log.error("every request failed. Most common reason: %s", reasons.most_common(1)[0][0])

    log.info(
        "done: %d ok, %d failed, %d retried, %d tokens",
        aggregates.succeeded,
        aggregates.failed,
        run.retried,
        metadata.token_usage.total,
    )
    log.info("wrote %s and %s", json_path, report_path)

    # A failed record is a result, not a crash: the files are written either
    # way and the exit code says whether everything came through.
    return 1 if aggregates.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
