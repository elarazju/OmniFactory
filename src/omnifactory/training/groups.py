"""Disk-backed source-group membership shared by prepared data and checkpoints."""

import sqlite3
from contextlib import closing
from pathlib import Path

from omnifactory.errors import OmniFactoryError
from omnifactory.training.fingerprints import file_fingerprint

GROUP_SCHEMA = "CREATE TABLE IF NOT EXISTS groups (dataset TEXT, name TEXT, PRIMARY KEY(dataset,name)) WITHOUT ROWID"


class GroupIndex:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def connect(self):
        return sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)

    def __contains__(self, key):
        with closing(self.connect()) as db:
            return (
                db.execute("SELECT 1 FROM groups WHERE dataset=? AND name=?", key).fetchone()
                is not None
            )

    def __iter__(self):
        with closing(self.connect()) as db:
            yield from db.execute("SELECT dataset,name FROM groups ORDER BY dataset,name")

    def overlaps(self, other):
        with closing(self.connect()) as db:
            db.execute("ATTACH DATABASE ? AS other", (f"{other.path.as_uri()}?mode=ro",))
            return (
                db.execute(
                    "SELECT 1 FROM groups g JOIN other.groups o ON g.dataset=o.dataset AND g.name=o.name LIMIT 1"
                ).fetchone()
                is not None
            )

    def copy_to(self, path):
        with closing(self.connect()) as source, sqlite3.connect(path) as target:
            target.execute(GROUP_SCHEMA)
            target.executemany(
                "INSERT INTO groups VALUES (?,?)", source.execute("SELECT dataset,name FROM groups")
            )
        return file_fingerprint(path)


def checkpoint_groups(root, metadata):
    if "groups_sha256" not in metadata:
        raise OmniFactoryError(
            "E_CHECKPOINT_VERSION", "legacy group metadata; train a new checkpoint"
        )
    path = Path(root) / "groups.sqlite"
    if file_fingerprint(path) != metadata["groups_sha256"]:
        raise OmniFactoryError("E_CHECKPOINT_GROUPS", "checkpoint source groups changed")
    return GroupIndex(path)
