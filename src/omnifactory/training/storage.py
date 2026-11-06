"""Bounded, transactional safetensors shards with record-local tensor namespaces."""

import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from omnifactory.errors import OmniFactoryError
from omnifactory.training.fingerprints import file_fingerprint, stamp


def read_record(root, entry):
    root = Path(root).resolve()
    path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root):
        raise OmniFactoryError("E_PREPARED_PATH", "tensor path leaves artifact")
    before = stamp(path)
    if file_fingerprint(path) != entry["sha256"]:
        raise OmniFactoryError("E_PREPARED_HASH", f"prepared tensor changed: {path}")
    # Copy only this record; returned tensors do not keep an entire shard mapped.
    with safe_open(path, framework="pt", device="cpu") as handle:
        names = entry.get("tensor_names")
        if not names:
            raise OmniFactoryError("E_PREPARED_VERSION", "reprepare data with the current version")
        tensors = {key: handle.get_tensor(name).clone() for key, name in names.items()}
    if stamp(path) != before:
        raise OmniFactoryError("E_SOURCE_CHANGED", "prepared shard changed while reading")
    return tensors


def encoded_metadata(encoded):
    from omnifactory.training.data_loading import tensor_bytes

    encoded.validate()
    return {
        "sample_id": encoded.cost.sample_id,
        "prediction_id": encoded.cost.prediction_id,
        "prompt_length": encoded.prompt_length,
        "answer": encoded.answer,
        "layout": encoded.layout,
        "supervision": encoded.supervision.model_dump(mode="json"),
        "normalization_counts": encoded.normalization_counts,
        "metadata": encoded.metadata,
        "cost": encoded.cost.model_dump(mode="json"),
        # Also reserve supervision tensors even if the adapter's payload omits them.
        "tensor_bytes": max(tensor_bytes(encoded.tensors), tensor_bytes(encoded.payload)),
    }


class ShardWriter:
    """One pending shard; publish its file before committing all index rows together.

    A single oversized record occupies its own shard. Byte limits cover tensor data,
    excluding encoder workspace, serialization overhead and one incoming record.
    """

    def __init__(self, root, *, max_records, max_bytes, commit):
        self.root = Path(root)
        self.max_records, self.max_bytes, self.commit = max_records, max_bytes, commit
        self.pending, self.tensors = [], {}
        self.size = self.peak_bytes = self.shards = self.oversized = 0

    def add(self, index, cache_key, entry, tensors):
        from omnifactory.training.data_loading import tensor_bytes

        size = tensor_bytes(tensors)
        if self.pending and (
            len(self.pending) >= self.max_records or self.size + size > self.max_bytes
        ):
            self.flush()
        entry = dict(entry)
        entry["tensor_names"] = {k: f"{index:08d}/{k}" for k in tensors}
        self.tensors.update(
            {
                entry["tensor_names"][k]: v.detach().clone(memory_format=torch.contiguous_format)
                for k, v in tensors.items()
            }
        )
        self.pending.append((index, cache_key, entry))
        self.size += size
        self.peak_bytes = max(self.peak_bytes, self.size)
        if size > self.max_bytes:
            self.oversized += 1
        if len(self.pending) >= self.max_records or self.size >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        file = f"tensors/shard-{self.pending[0][0]:08d}.safetensors"
        path = self.root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        try:
            save_file(self.tensors, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            checksum = file_fingerprint(path, refresh=True)
            rows = []
            for index, key, entry in self.pending:
                entry.update(file=file, sha256=checksum)
                rows.append((index, key, entry))
            self.commit(rows)
        finally:
            temporary.unlink(missing_ok=True)
        self.shards += 1
        self.pending.clear()
        self.tensors.clear()
        self.size = 0
