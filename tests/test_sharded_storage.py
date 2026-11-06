import sqlite3

import pytest

pytest.importorskip("torch")
import torch

from omnifactory.errors import OmniFactoryError
from omnifactory.training.storage import ShardWriter, read_record


def test_shard_boundaries_selective_reads_and_oversized_records(tmp_path):
    committed = []
    writer = ShardWriter(
        tmp_path, max_records=3, max_bytes=64, commit=lambda rows: committed.extend(rows)
    )
    for i, size in enumerate((4, 4, 4, 20, 4)):
        writer.add(
            i, str(i), {"tensor_bytes": size * 4}, {"nested/x": torch.full((size,), float(i))}
        )
    writer.flush()
    assert len({e["file"] for _, _, e in committed}) == 3
    assert writer.oversized == 1
    assert writer.peak_bytes == 80
    first = committed[0][2]
    payload = read_record(tmp_path, first)
    payload["nested/x"].fill_(99)
    assert read_record(tmp_path, first)["nested/x"].sum() == 0
    for i, _, entry in committed:
        assert torch.all(read_record(tmp_path, entry)["nested/x"] == i)
    with pytest.raises(OmniFactoryError, match="E_PREPARED_PATH"):
        read_record(tmp_path, {**first, "file": "../outside"})
    (tmp_path / first["file"]).write_bytes(b"damaged")
    with pytest.raises(OmniFactoryError, match="E_PREPARED_HASH"):
        read_record(tmp_path, first)


def test_file_published_before_index_commit_and_orphan_can_be_replaced(tmp_path):
    def crash(rows):
        assert read_record(tmp_path, rows[0][2])["x"].item() == 1
        raise RuntimeError("database transaction failed")

    writer = ShardWriter(tmp_path, max_records=1, max_bytes=64, commit=crash)
    with pytest.raises(RuntimeError, match="transaction"):
        writer.add(0, "", {}, {"x": torch.ones(1)})
    result = []
    retry = ShardWriter(tmp_path, max_records=1, max_bytes=64, commit=result.extend)
    retry.add(0, "", {}, {"x": torch.zeros(1)})
    assert read_record(tmp_path, result[0][2])["x"].item() == 0
    assert not list(tmp_path.rglob("*.partial"))


def test_group_index_copy_membership_and_overlap(tmp_path):
    from omnifactory.training.groups import GROUP_SCHEMA, GroupIndex, checkpoint_groups

    for name, rows in (("a", [("d", "x"), ("d", "y")]), ("b", [("d", "y")]), ("c", [("d", "z")])):
        with sqlite3.connect(tmp_path / name) as db:
            db.execute(GROUP_SCHEMA)
            db.executemany("INSERT INTO groups VALUES (?,?)", rows)
    index = GroupIndex(tmp_path / "a")
    assert ("d", "x") in index and ("d", "z") not in index
    assert index.overlaps(GroupIndex(tmp_path / "b"))
    assert not index.overlaps(GroupIndex(tmp_path / "c"))
    checksum = index.copy_to(tmp_path / "groups.sqlite")
    copied = checkpoint_groups(tmp_path, {"groups_sha256": checksum})
    assert list(copied) == [("d", "x"), ("d", "y")]
    (tmp_path / "groups.sqlite").write_bytes(b"changed")
    with pytest.raises(OmniFactoryError, match="E_CHECKPOINT_GROUPS"):
        checkpoint_groups(tmp_path, {"groups_sha256": checksum})


def test_shard_owns_cached_tensor_storage(tmp_path):
    entries = []
    writer = ShardWriter(tmp_path, max_records=2, max_bytes=64, commit=entries.extend)
    shared = torch.ones(4)
    writer.add(0, "", {}, {"cached_reference": shared})
    shared.fill_(2)
    writer.add(1, "", {}, {"cached_reference": shared})
    assert read_record(tmp_path, entries[0][2])["cached_reference"].tolist() == [1] * 4
    assert read_record(tmp_path, entries[1][2])["cached_reference"].tolist() == [2] * 4
