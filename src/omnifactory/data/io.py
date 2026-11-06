"""Streaming JSONL input, dataset checks and transactional local outputs."""

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from omnifactory.data.schema import CanonicalSample, Record
from omnifactory.errors import OmniFactoryError


class DatasetReport(Record):
    records: int
    valid: int
    invalid: int
    errors: tuple[str, ...]
    errors_truncated: bool


def iter_samples(path: Path) -> Iterator[CanonicalSample]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield CanonicalSample.model_validate_json(line)
            except ValidationError as exc:
                raise OmniFactoryError("E_DATASET", f"{path}:{line_number}: {exc}") from exc


def validate_dataset(path: Path, *, max_errors: int = 20) -> DatasetReport:
    if max_errors < 1:
        raise OmniFactoryError("E_VALIDATION", "max_errors must be positive")
    records = valid = invalid = 0
    errors = []
    # Temporary on-disk B-trees keep duplicate and split checks bounded in RAM.
    with (
        tempfile.TemporaryDirectory(prefix="omnifactory-validate-") as temporary,
        sqlite3.connect(Path(temporary) / "seen.sqlite") as db,
    ):
        db.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
        db.execute(
            "CREATE TABLE splits (dataset TEXT, name TEXT, split TEXT, PRIMARY KEY(dataset,name)) WITHOUT ROWID"
        )
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                records += 1
                try:
                    sample = CanonicalSample.model_validate_json(line)
                    if db.execute("SELECT 1 FROM ids WHERE id=?", (sample.id,)).fetchone():
                        raise OmniFactoryError("E_DUPLICATE_SAMPLE", sample.id)
                    group = (sample.provenance.dataset, sample.provenance.group_id)
                    previous = db.execute(
                        "SELECT split FROM splits WHERE dataset=? AND name=?", group
                    ).fetchone()
                    if previous and previous[0] != sample.provenance.split:
                        raise OmniFactoryError(
                            "E_SPLIT_LEAK", f"group {group} occurs in different splits"
                        )
                    db.execute("INSERT INTO ids VALUES (?)", (sample.id,))
                    db.execute(
                        "INSERT OR IGNORE INTO splits VALUES (?,?,?)",
                        (*group, sample.provenance.split),
                    )
                    valid += 1
                except (ValidationError, OmniFactoryError) as exc:
                    invalid += 1
                    if len(errors) < max_errors:
                        errors.append(f"line {number}: {exc}")
    if records == 0:
        errors.append("E_EMPTY_DATASET: no records found")
    return DatasetReport(
        records=records,
        valid=valid,
        invalid=invalid,
        errors=tuple(errors),
        errors_truncated=invalid > max_errors,
    )


@contextmanager
def atomic_writer(path: Path):
    """A late failed sample must not publish a partial preparation or batch plan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OmniFactoryError("E_JSON", f"{path}: {exc}") from exc
