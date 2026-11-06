"""Hash immutable local assets once; invalidate on filesystem identity changes."""

import hashlib
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from omnifactory.errors import OmniFactoryError


def stamp(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def cached_hash(key, signature, compute, *, refresh=False):
    root = Path(os.environ.get("OMNIFACTORY_HASH_CACHE", Path.home() / ".cache/omnifactory"))
    root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(root / "hashes.sqlite", timeout=30)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, stamp TEXT, hash TEXT)"
        )
        row = db.execute("SELECT stamp, hash FROM hashes WHERE path=?", (key,)).fetchone()
        if not refresh and row and row[0] == signature:
            return row[1]
        value = compute()
        db.execute("INSERT OR REPLACE INTO hashes VALUES (?,?,?)", (key, signature, value))
    return value


def file_fingerprint(path, *, refresh=False):
    path = Path(path).resolve()
    before = stamp(path)

    def compute():
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        if stamp(path) != before:
            raise OmniFactoryError("E_SOURCE_CHANGED", f"file changed while hashing: {path}")
        return value

    return cached_hash(str(path), str(before), compute, refresh=refresh)
