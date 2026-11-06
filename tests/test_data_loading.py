import hashlib

import pytest

pytest.importorskip("torch")
import torch
from safetensors.torch import load_file, save_file

from omnifactory.errors import OmniFactoryError
from omnifactory.training.data_loading import PreparedPayloadLoader


def records(root):
    rows = []
    for i in range(3):
        path = root / f"{i}.safetensors"
        save_file({"x": torch.full((8,), float(i))}, path)
        rows.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


def load(root, entry):
    path = root / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise OmniFactoryError("E_PREPARED_HASH", "changed tensor")
    return {"nested": [load_file(path)], "marker": entry.get("marker")}


def test_bounded_lru_order_and_independent_returned_trees(tmp_path):
    rows = records(tmp_path)
    loader = PreparedPayloadLoader(tmp_path, max_bytes=64, load_payload=load)
    a = loader(rows[0])
    a["nested"][0]["x"].fill_(99)
    a["nested"].append("mutated")
    loader(rows[1])
    cached = loader(rows[0])
    assert len(cached["nested"]) == 1 and cached["nested"][0]["x"].sum() == 0
    loader(rows[2])  # Evicts 1; the hit made 0 most recent.
    loader(rows[0])
    assert loader.hits == 2 and loader.misses == 3
    loader(rows[1])
    assert loader.misses == 4
    assert loader.report()["peak_tensor_bytes"] == 64


def test_changed_tensor_or_metadata_never_reuses_old_payload(tmp_path):
    row = records(tmp_path)[0]
    loader = PreparedPayloadLoader(tmp_path, max_bytes=64, load_payload=load)
    loader(row)
    changed = {**row, "marker": "new layout"}
    assert loader(changed)["marker"] == "new layout"
    assert loader.misses == 2
    save_file({"x": torch.ones(8)}, tmp_path / row["file"])
    with pytest.raises(OmniFactoryError, match="E_PREPARED_HASH"):
        loader(row)
    with pytest.raises(OmniFactoryError, match="E_PREPARED_PATH"):
        loader({**row, "file": "../outside"})


def test_oversized_bypass_disabled_cache_and_exception_propagation(tmp_path):
    row = records(tmp_path)[0]
    small = PreparedPayloadLoader(tmp_path, max_bytes=31, load_payload=load)
    for _ in range(2):
        small(row)
    assert small.bypasses == 2 and small.resident_bytes == 0 and small.hits == 0
    disabled = PreparedPayloadLoader(tmp_path, max_bytes=0, load_payload=load)
    disabled(row)
    assert disabled.report()["resident_tensor_bytes"] == 0
    (tmp_path / row["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        small(row)


def test_inflight_file_change_is_rejected(tmp_path):
    row = records(tmp_path)[0]

    def mutate(root, entry):
        data = load(root, entry)
        (root / entry["file"]).write_bytes(b"changed after loading")
        return data

    loader = PreparedPayloadLoader(tmp_path, max_bytes=64, load_payload=mutate)
    with pytest.raises(OmniFactoryError, match="E_SOURCE_CHANGED"):
        loader(row)
    assert loader.resident_bytes == 0


def test_prefetch_order_byte_bound_bypass_and_rng_isolation(tmp_path):
    rows = [{**row, "tensor_bytes": 32} for row in records(tmp_path)]
    rows[1]["tensor_bytes"] = 96  # Conservative oversized reservation.
    loader = PreparedPayloadLoader(
        tmp_path, prefetch_records=2, prefetch_bytes=64, load_payload=load
    )
    rng = torch.get_rng_state().clone()
    result = [int(p["nested"][0]["x"][0]) for p in loader.iterate(rows * 3)]
    assert result == [0, 1, 2] * 3
    assert torch.equal(rng, torch.get_rng_state())
    report = loader.report()
    assert report["peak_prefetch_reserved_bytes"] <= 64
    assert report["synchronous_bypasses"] == 3
    assert report["prefetched_records"] == 6


def test_prefetch_failure_and_early_close_join_worker(tmp_path):
    import threading

    rows = [{**row, "tensor_bytes": 32} for row in records(tmp_path)]
    loader = PreparedPayloadLoader(
        tmp_path, prefetch_records=2, prefetch_bytes=64, load_payload=load
    )
    iterator = loader.iterate(rows)
    next(iterator)
    iterator.close()
    assert not any(t.name.startswith("omni-data") for t in threading.enumerate())
    (tmp_path / rows[1]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        list(loader.iterate(rows))
    assert not any(t.name.startswith("omni-data") for t in threading.enumerate())


def test_prefetch_rejects_underreported_payload_size(tmp_path):
    row = {**records(tmp_path)[0], "tensor_bytes": 1}
    loader = PreparedPayloadLoader(
        tmp_path, prefetch_records=2, prefetch_bytes=64, load_payload=load
    )
    with pytest.raises(OmniFactoryError, match="E_DATA_PREFETCH"):
        list(loader.iterate([row]))
