"""Reading the inbox export.

The input is described as "as in real life", so the loader does not assume the
file is clean: it tolerates a BOM, missing optional columns and blank lines, and
it reports what it dropped instead of silently shrinking the dataset.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

from .models import InboxRow

REQUIRED_COLUMNS = {"id", "raw_text"}

# utf-8 first, then what Excel writes on a Ukrainian Windows when you pick
# "Save as CSV". The task says the data is as in real life, and that is what
# real life hands you.
ENCODINGS = ("utf-8-sig", "utf-16", "cp1251")


class InputError(Exception):
    """The file itself is unusable (missing, empty, no id column)."""


@dataclass
class LoadResult:
    rows: list[InboxRow]
    skipped: list[tuple[int, str]] = field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def _read_text(path: Path) -> str:
    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            # Most often the file is open in Excel, which is exactly the kind of
            # thing that happens to whoever runs this.
            raise InputError(
                f"cannot read {path}: {exc}. Close it if it is open elsewhere."
            ) from exc
    raise InputError(
        f"cannot decode {path}: tried {', '.join(ENCODINGS)}. Re-save the file as UTF-8."
    )


def load_requests(path: Path) -> LoadResult:
    if not path.exists():
        raise InputError(f"input file not found: {path}")
    if not path.is_file():
        raise InputError(f"{path} is a directory, not a file")

    reader = csv.DictReader(io.StringIO(_read_text(path), newline=""))
    if reader.fieldnames is None:
        raise InputError(f"input file has no header row: {path}")

    header = {(name or "").strip().lower() for name in reader.fieldnames}
    missing = REQUIRED_COLUMNS - header
    if missing:
        hint = ""
        if len(reader.fieldnames) == 1 and ";" in (reader.fieldnames[0] or ""):
            hint = " The header looks semicolon-separated; re-save it with commas."
        raise InputError(
            f"input file is missing required column(s): {', '.join(sorted(missing))}.{hint}"
        )

    rows: list[InboxRow] = []
    skipped: list[tuple[int, str]] = []
    seen_ids: set[str] = set()

    for raw in reader:
        # line_num, not a counter: a quoted field can span several physical
        # lines, and this number exists so a human can open the file and fix it.
        line_no = reader.line_num
        record = {(k or "").strip().lower(): (v or "") for k, v in raw.items() if k is not None}
        row_id = record.get("id", "").strip()
        text = record.get("raw_text", "").strip()

        if not row_id:
            skipped.append((line_no, "missing id"))
            continue
        if row_id in seen_ids:
            skipped.append((line_no, f"duplicate id {row_id}"))
            continue
        if not text:
            skipped.append((line_no, f"{row_id}: empty raw_text"))
            continue

        seen_ids.add(row_id)
        rows.append(
            InboxRow(
                id=row_id,
                channel=record.get("channel", "").strip(),
                timestamp=record.get("timestamp", "").strip(),
                raw_text=text,
            )
        )

    if not rows:
        raise InputError(f"no usable rows in {path}")

    return LoadResult(rows=rows, skipped=skipped)
