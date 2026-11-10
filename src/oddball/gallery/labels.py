"""Human labels for gallery candidates (docs/ML_DESIGN.md §7): validation and an append-only CSV.

The label file is committed to the repo (``labels/labels_v1.csv``). It is **append-only**: a new
label for a star is a new row, and the latest row per ``(tic, sector)`` wins. Every write takes
an exclusive ``flock``, appends one complete line, flushes and fsyncs, so a crash or restart can
at worst leave one truncated last line, which the reader skips (and the next write starts on a
fresh line).
"""

from __future__ import annotations

import csv
import datetime as dt
import fcntl
import io
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from oddball.common.paths import REPO_ROOT

LABELS = ("interesting", "systematic", "known-boring-variable", "unsure")
COLUMNS = ("tic", "sector", "scorer", "rank", "split", "reason", "label", "note", "utc", "gallery")
DEFAULT_LABELS_CSV = REPO_ROOT / "labels" / "labels_v1.csv"

NOTE_MAX = 500
MAX_TIC = 10**10  # TIC v8 ids are < 2e9; anything this large is garbage
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")  # scorer / gallery ids
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


class LabelError(ValueError):
    """An input failed validation (the message is safe to show to the user)."""


def validate_label(label) -> str:
    if not isinstance(label, str) or label not in LABELS:
        raise LabelError(f"label must be one of {', '.join(LABELS)}")
    return label


def validate_tic(tic) -> int:
    """A positive int, or a string of digits. Rejects bools, floats and anything else."""
    if isinstance(tic, bool):
        raise LabelError("tic must be an integer")
    if isinstance(tic, int):
        v = tic
    elif isinstance(tic, str) and tic.isascii() and tic.isdigit() and len(tic) <= 11:
        v = int(tic)
    else:
        raise LabelError("tic must be an integer")
    if not 0 < v < MAX_TIC:
        raise LabelError("tic out of range")
    return v


def validate_sector(sector) -> int:
    try:
        v = validate_tic(sector)
    except LabelError:
        raise LabelError("sector must be an integer") from None
    if not 1 <= v <= 999:
        raise LabelError("sector out of range")
    return v


def validate_name(name, what: str = "name") -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise LabelError(f"bad {what}")
    return name


def clean_note(note) -> str:
    """Single line, no control characters, at most NOTE_MAX characters, spreadsheet-safe.

    CSV quoting itself is left to the ``csv`` module. A note that starts like a spreadsheet
    formula (``=``, ``+``, ``-``, ``@``) gets a leading apostrophe so it can't run as one.
    """
    if note is None:
        return ""
    if not isinstance(note, str):
        raise LabelError("note must be text")
    if len(note) > 4 * NOTE_MAX:  # reject absurd input before doing any work on it
        raise LabelError(f"note longer than {NOTE_MAX} characters")
    note = unicodedata.normalize("NFC", note)
    note = "".join(" " if unicodedata.category(c)[0] in "CZ" and c != " " else c for c in note)
    note = " ".join(note.split())
    if len(note) > NOTE_MAX:
        raise LabelError(f"note longer than {NOTE_MAX} characters")
    if note.startswith(_FORMULA_START):
        note = "'" + note
    return note


@dataclass(frozen=True)
class LabelRow:
    tic: int
    sector: int
    scorer: str
    rank: int
    split: str
    reason: str
    label: str
    note: str
    utc: str
    gallery: str

    def as_list(self) -> list[str]:
        return [str(getattr(self, c)) for c in COLUMNS]


def make_row(*, tic, sector, scorer, rank, split, reason, label, note, gallery,
             utc: str | None = None) -> LabelRow:  # fmt: skip
    """Validate everything and build a row. ``reason``/``split`` come from the gallery manifest."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise LabelError("rank must be a non-negative integer")
    return LabelRow(
        tic=validate_tic(tic),
        sector=validate_sector(sector),
        scorer=validate_name(scorer, "scorer"),
        rank=rank,
        split=validate_name(split, "split") if split else "",
        reason=clean_note(reason)[:200],
        label=validate_label(label),
        note=clean_note(note),
        utc=utc or dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        gallery=validate_name(gallery, "gallery"),
    )


class LabelStore:
    """Append-only CSV of labels. Safe for one server process plus concurrent readers."""

    def __init__(self, path: Path = DEFAULT_LABELS_CSV):
        self.path = Path(path)

    def append(self, row: LabelRow) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow(row.as_list())
        line = buf.getvalue()
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            size = os.fstat(fd).st_size
            head = ""
            if size == 0:
                hb = io.StringIO()
                csv.writer(hb, lineterminator="\n").writerow(COLUMNS)
                head = hb.getvalue()
            elif not self._ends_with_newline(fd, size):
                head = "\n"  # a previous write was cut off: start a fresh line
            os.write(fd, (head + line).encode("utf-8"))
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _ends_with_newline(fd: int, size: int) -> bool:
        return os.pread(fd, 1, size - 1) == b"\n"

    def rows(self) -> list[LabelRow]:
        """Every valid row in file order. Malformed or truncated lines are skipped."""
        if not self.path.exists():
            return []
        out = []
        with self.path.open(newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.reader(fh)
            header = next(rd, None)
            if header is None or tuple(header) != COLUMNS:
                if header is not None:
                    raise ValueError(f"{self.path}: unexpected header {header}")
                return []
            for rec in rd:
                if len(rec) != len(COLUMNS):
                    continue
                d = dict(zip(COLUMNS, rec, strict=True))
                try:
                    out.append(LabelRow(
                        tic=validate_tic(d["tic"]), sector=validate_sector(d["sector"]),
                        scorer=d["scorer"], rank=int(d["rank"]), split=d["split"],
                        reason=d["reason"], label=validate_label(d["label"]), note=d["note"],
                        utc=d["utc"], gallery=d["gallery"]))  # fmt: skip
                except (LabelError, ValueError):
                    continue
        return out

    def latest(self) -> dict[tuple[int, int], LabelRow]:
        """Latest label per ``(tic, sector)`` (whatever scorer or gallery it came from)."""
        out: dict[tuple[int, int], LabelRow] = {}
        for r in self.rows():
            out[(r.tic, r.sector)] = r
        return out
