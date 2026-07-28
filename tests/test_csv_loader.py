"""Input handling: the file is described as "as in real life"."""

from __future__ import annotations

from pathlib import Path

import pytest

from inbox_triage.csv_loader import InputError, load_requests

HEADER = "id,channel,timestamp,raw_text\n"


def write_csv(tmp_path: Path, body: str, *, header: str = HEADER, encoding: str = "utf-8") -> Path:
    path = tmp_path / "input.csv"
    path.write_text(header + body, encoding=encoding)
    return path


class TestHappyPath:
    def test_reads_rows(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, 'REQ-001,Slack,2026-06-08 09:14,"Треба звіт"\n')
        result = load_requests(path)
        assert len(result.rows) == 1
        assert result.rows[0].id == "REQ-001"
        assert result.rows[0].channel == "Slack"

    def test_bom_is_stripped(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "REQ-001,Slack,,текст\n", encoding="utf-8-sig")
        assert load_requests(path).rows[0].id == "REQ-001"

    def test_embedded_newlines_and_commas_survive(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, 'REQ-001,Email,,"перший рядок,\nдругий рядок"\n')
        assert "другий рядок" in load_requests(path).rows[0].raw_text

    def test_real_dataset_loads(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "input_requests.csv"
        if not path.exists():  # pragma: no cover - dataset is committed
            pytest.skip("dataset not present")
        result = load_requests(path)
        assert len(result.rows) == 18
        assert result.skipped == []


class TestBadInput:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(InputError, match="not found"):
            load_requests(tmp_path / "nope.csv")

    def test_missing_required_column(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "Slack,текст\n", header="channel,raw_text\n")
        with pytest.raises(InputError, match="missing required column"):
            load_requests(path)

    def test_no_usable_rows(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "REQ-001,Slack,,\n")
        with pytest.raises(InputError, match="no usable rows"):
            load_requests(path)

    def test_rows_are_skipped_with_a_reason_not_dropped_silently(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path,
            'REQ-001,Slack,,"нормальний запит"\n'
            ",Telegram,,текст без id\n"
            "REQ-003,Email,,\n"
            'REQ-001,Slack,,"той самий id"\n',
        )
        result = load_requests(path)
        assert len(result.rows) == 1
        assert result.skipped_count == 3
        reasons = " ".join(reason for _, reason in result.skipped)
        assert "missing id" in reasons
        assert "empty raw_text" in reasons
        assert "duplicate id" in reasons
