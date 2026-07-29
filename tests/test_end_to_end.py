"""One run of the whole thing, from CSV to the two output files.

Unit tests prove each stage in isolation, which is exactly how a pipeline can
be green everywhere and still produce a broken output.json. This drives the
real CLI against a scripted client and then reads what landed on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inbox_triage.__main__ import main
from inbox_triage.llm.fake import valid_json
from inbox_triage.models import TriageOutput

CSV_BODY = """id,channel,timestamp,raw_text
REQ-001,Slack,2026-06-08 09:14,"Можна автоматизувати щотижневий звіт по Google Ads? Зараз руками вивантажую CSV щопонеділка."
REQ-002,Telegram,2026-06-08 09:31,хлопці треба бот
REQ-005,Telegram,2026-06-08 11:05,"ГОРИТЬ. Сьогодні до вечора треба вивантажити список контрагентів, бухгалтерія просить терміново."
REQ-013,Slack,2026-06-08 16:02,"Той самий звіт по Google Ads, що Оля просила в понеділок, мені теж потрібен."
"""


def scripted(prompt: str) -> str:
    """Answer per request id, so the run produces a varied, realistic output."""
    if "Знайди пари" in prompt or "duplicate_of" in prompt:
        return json.dumps(
            {
                "duplicates": [
                    {"id": "REQ-013", "duplicate_of": "REQ-001", "reason": "той самий звіт"}
                ]
            },
            ensure_ascii=False,
        )
    if "REQ-002" in prompt:
        return valid_json(
            category="автоматизація",
            short_summary="Просять бота без жодних деталей.",
            requested_actions=[],
            needs_clarification=True,
            work_item_type="project",
            target_department=None,
        )
    if "REQ-005" in prompt:
        return valid_json(
            category="звіт/аналітика",
            target_department="фінанси/бухгалтерія",
            priority="high",
            short_summary="Разова термінова вивантаження контрагентів.",
            requested_actions=["Вивантажити список контрагентів"],
            work_item_type="one_off",
            urgency_signals=["ГОРИТЬ", "терміново"],
        )
    return valid_json()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(CSV_BODY, encoding="utf-8")
    out_dir = tmp_path / "out"

    # A key must be present for the run to start, but no request leaves the
    # process: the client is swapped for the scripted fake.
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))

    from inbox_triage import __main__ as entry
    from inbox_triage.llm.fake import FakeLLMClient

    class Recording(FakeLLMClient):
        def __init__(self, **_: Any) -> None:
            super().__init__(scripted, name="fake")

        def close(self) -> None:
            return None

    monkeypatch.setattr(entry, "ChatClient", Recording)
    return {"csv": csv_path, "out": out_dir}


class TestFullRun:
    def test_run_writes_both_files_and_exits_clean(self, workspace: dict[str, Any]) -> None:
        code = main(["--input", str(workspace["csv"]), "--output-dir", str(workspace["out"])])
        assert code == 0
        assert (workspace["out"] / "output.json").exists()
        assert (workspace["out"] / "report.md").exists()

    def test_output_json_matches_the_schema_it_claims(self, workspace: dict[str, Any]) -> None:
        main(["--input", str(workspace["csv"]), "--output-dir", str(workspace["out"])])
        payload = json.loads((workspace["out"] / "output.json").read_text(encoding="utf-8"))

        # Round-trips through the model, so the file is not merely JSON-shaped.
        parsed = TriageOutput.model_validate(payload)
        assert len(parsed.records) == 4
        assert parsed.run.total_requests == 4
        assert parsed.run.succeeded == 4
        assert parsed.run.failed == 0
        assert parsed.run.prompt_version

    def test_every_mandatory_field_from_the_spec_is_present(
        self, workspace: dict[str, Any]
    ) -> None:
        main(["--input", str(workspace["csv"]), "--output-dir", str(workspace["out"])])
        payload = json.loads((workspace["out"] / "output.json").read_text(encoding="utf-8"))
        required = {
            "category",
            "target_department",
            "priority",
            "short_summary",
            "requested_actions",
            "needs_clarification",
        }
        for record in payload["records"]:
            assert required <= set(record["extraction"]), record["id"]

    def test_rules_and_dedup_are_visible_in_the_output(self, workspace: dict[str, Any]) -> None:
        main(["--input", str(workspace["csv"]), "--output-dir", str(workspace["out"])])
        payload = json.loads((workspace["out"] / "output.json").read_text(encoding="utf-8"))
        by_id = {record["id"]: record for record in payload["records"]}

        # Three words cannot be a complete request.
        assert by_id["REQ-002"]["extraction"]["needs_clarification"] is True
        # The duplicate pass ran and its verdict survived validation.
        assert by_id["REQ-013"]["duplicate_of"] == "REQ-001"
        # Quotes that exist in the source are kept.
        assert "ГОРИТЬ" in by_id["REQ-005"]["extraction"]["urgency_signals"]

    def test_report_contains_the_aggregates_the_task_asks_for(
        self, workspace: dict[str, Any]
    ) -> None:
        main(["--input", str(workspace["csv"]), "--output-dir", str(workspace["out"])])
        report = (workspace["out"] / "report.md").read_text(encoding="utf-8")
        for heading in ("За категорією", "За пріоритетом", "За відділом", "Потребують уточнення"):
            assert heading in report
        assert "REQ-002" in report

    def test_api_key_never_reaches_the_output(self, workspace: dict[str, Any]) -> None:
        main(["--input", str(workspace["csv"]), "--output-dir", str(workspace["out"])])
        for name in ("output.json", "report.md"):
            assert "test-key-not-real" not in (workspace["out"] / name).read_text(encoding="utf-8")

    def test_limit_flag_is_respected(self, workspace: dict[str, Any]) -> None:
        main(
            [
                "--input",
                str(workspace["csv"]),
                "--output-dir",
                str(workspace["out"]),
                "--limit",
                "2",
            ]
        )
        payload = json.loads((workspace["out"] / "output.json").read_text(encoding="utf-8"))
        assert len(payload["records"]) == 2

    def test_second_run_is_served_from_cache(self, workspace: dict[str, Any]) -> None:
        args = ["--input", str(workspace["csv"]), "--output-dir", str(workspace["out"])]
        main(args)
        main(args)
        payload = json.loads((workspace["out"] / "output.json").read_text(encoding="utf-8"))
        assert payload["run"]["cache_hits"] > 0
        assert payload["run"]["token_usage"]["prompt_tokens"] == 0


class TestFailureExitCodes:
    def test_missing_input_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_API_KEY", "x")
        assert main(["--input", str(tmp_path / "nope.csv")]) == 2

    def test_missing_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        csv_path = tmp_path / "input.csv"
        csv_path.write_text(CSV_BODY, encoding="utf-8")
        for name in (
            "LLM_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr("inbox_triage.config.load_dotenv", lambda *a, **k: False)
        assert main(["--input", str(csv_path)]) == 2

    def test_unknown_provider_without_base_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / "input.csv"
        csv_path.write_text(CSV_BODY, encoding="utf-8")
        monkeypatch.setenv("LLM_API_KEY", "x")
        monkeypatch.setenv("LLM_PROVIDER", "no-such-gateway")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        assert main(["--input", str(csv_path)]) == 2

    def test_unusable_output_dir_is_caught_before_any_llm_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Discovering this after paying for the run is the worst moment.
        csv_path = tmp_path / "input.csv"
        csv_path.write_text(CSV_BODY, encoding="utf-8")
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("i am a file", encoding="utf-8")
        monkeypatch.setenv("LLM_API_KEY", "x")

        from inbox_triage import __main__ as entry
        from inbox_triage.llm.fake import FakeLLMClient

        calls: list[str] = []

        class Counting(FakeLLMClient):
            def __init__(self, **_: Any) -> None:
                super().__init__(lambda p: (calls.append(p), valid_json())[1], name="counting")

            def close(self) -> None:
                return None

        monkeypatch.setattr(entry, "ChatClient", Counting)
        assert main(["--input", str(csv_path), "--output-dir", str(blocker)]) == 2
        assert calls == []

    def test_a_fully_failed_run_says_so_at_error_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A dead key and one bad row used to look identical in the log.
        csv_path = tmp_path / "input.csv"
        csv_path.write_text(CSV_BODY, encoding="utf-8")
        monkeypatch.setenv("LLM_API_KEY", "x")
        monkeypatch.setenv("LLM_CACHE_DIR", "")

        from inbox_triage import __main__ as entry
        from inbox_triage.llm.base import LLMError
        from inbox_triage.llm.fake import FakeLLMClient

        class Dead(FakeLLMClient):
            def __init__(self, **_: Any) -> None:
                super().__init__([], name="dead")

            def generate(self, prompt: str, *, system: str | None = None) -> Any:
                raise LLMError("auth rejected (401), check LLM_API_KEY")

            def close(self) -> None:
                return None

        monkeypatch.setattr(entry, "ChatClient", Dead)
        with caplog.at_level("ERROR"):
            code = main(["--input", str(csv_path), "--output-dir", str(tmp_path / "out")])
        # A rejected key is a configuration problem, same class as a missing
        # one, so it gets the same exit code as a missing key rather than the
        # code that means "some rows did not parse".
        assert code == 2
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_failed_row_sets_exit_code_one_but_still_writes_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / "input.csv"
        csv_path.write_text(CSV_BODY, encoding="utf-8")
        out_dir = tmp_path / "out"
        monkeypatch.setenv("LLM_API_KEY", "x")
        monkeypatch.setenv("LLM_CACHE_DIR", "")

        from inbox_triage import __main__ as entry
        from inbox_triage.llm.fake import FakeLLMClient

        class Broken(FakeLLMClient):
            def __init__(self, **_: Any) -> None:
                super().__init__(lambda _: "це точно не JSON", name="broken")

            def close(self) -> None:
                return None

        monkeypatch.setattr(entry, "ChatClient", Broken)

        code = main(["--input", str(csv_path), "--output-dir", str(out_dir), "--no-dedup"])
        assert code == 1
        payload = json.loads((out_dir / "output.json").read_text(encoding="utf-8"))
        assert payload["run"]["failed"] == 4
        # Every row survives with its original text so a human can pick it up.
        assert all(record["raw_text"] for record in payload["records"])
