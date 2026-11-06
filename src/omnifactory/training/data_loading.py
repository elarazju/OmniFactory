"""Opt-in CPU tensor cache for immutable prepared records with a byte-bounded LRU.

The limit covers retained tensor storage, not the current batch or Python metadata.
Returned trees own their tensors: adapter collation cannot mutate future batches.
"""

import hashlib
import json
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from omnifactory.errors import OmniFactoryError
from omnifactory.training.fingerprints import stamp


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise OmniFactoryError("E_DATA_CACHE", "prepared cache requires CPU tensors")
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.contiguous_format)
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    return value


class PreparedPayloadLoader:
    def __init__(
        self, root, *, max_bytes=0, prefetch_records=0, prefetch_bytes=0, load_payload=None
    ):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("cache max_bytes must be a nonnegative integer")
        if min(prefetch_records, prefetch_bytes) < 0:
            raise ValueError("prefetch limits must be nonnegative")
        self.prefetch_records, self.prefetch_bytes = prefetch_records, prefetch_bytes
        self.prefetch_peak = self.prefetch_count = self.prefetch_bypasses = 0
        self.root, self.max_bytes = Path(root).resolve(), max_bytes
        if load_payload is None:
            from omnifactory.training.engine import load_tensors

            load_payload = load_tensors
        self.load_payload = load_payload
        self.cache = OrderedDict()
        self.resident_bytes = self.peak_bytes = 0
        self.hits = self.misses = self.bypasses = 0

    def __call__(self, entry):
        if not self.max_bytes:
            return self.load_payload(self.root, entry)
        path = (self.root / entry["file"]).resolve()
        if not path.is_relative_to(self.root):
            raise OmniFactoryError("E_PREPARED_PATH", "tensor path leaves artifact")
        # Bind layout, labels and measures as well as the tensor file's hash.
        key = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        signature = stamp(path)
        cached = self.cache.pop(key, None)
        if cached is not None:
            previous, size, payload = cached
            if signature == previous:
                self.cache[key] = cached
                self.hits += 1
                return clone_tree(payload)
            self.resident_bytes -= size
        self.misses += 1
        payload = self.load_payload(self.root, entry)
        if stamp(path) != signature:
            raise OmniFactoryError("E_SOURCE_CHANGED", "prepared file changed while loading")
        size = tensor_bytes(payload)
        if size > self.max_bytes:
            self.bypasses += 1
            return payload
        while self.resident_bytes + size > self.max_bytes:
            _, (_, old_size, _) = self.cache.popitem(last=False)
            self.resident_bytes -= old_size
        self.cache[key] = (signature, size, clone_tree(payload))
        self.resident_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        return payload

    def iterate(self, entries):
        """Ordered lookahead within an accumulation window; never advances training state.

        The reservation covers queued + running loads, excluding yielded payloads and
        the separately bounded LRU. Oversized/legacy records load synchronously.
        Closing the iterator joins the worker, including after an adapter failure.
        """
        if not self.prefetch_records or not self.prefetch_bytes:
            yield from (self(e) for e in entries)
            return
        iterator = iter(entries)
        queue, reserved = deque(), 0
        pending = next(iterator, None)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-data") as worker:
            try:
                while pending is not None or queue:
                    while pending is not None and len(queue) < self.prefetch_records:
                        size = pending.get("tensor_bytes")
                        if size is None or size > self.prefetch_bytes:
                            break
                        if size < 0:
                            raise OmniFactoryError("E_DATA_PREFETCH", "negative tensor byte count")
                        if reserved + size > self.prefetch_bytes:
                            break
                        queue.append((worker.submit(self, pending), size))
                        reserved += size
                        self.prefetch_peak = max(self.prefetch_peak, reserved)
                        self.prefetch_count += 1
                        pending = next(iterator, None)
                    if queue:
                        future, size = queue.popleft()
                        payload = future.result()
                        if tensor_bytes(payload) > size:
                            raise OmniFactoryError(
                                "E_DATA_PREFETCH", "payload exceeds reserved bytes"
                            )
                        reserved -= size
                        yield payload
                        del payload, future
                    else:
                        self.prefetch_bypasses += 1
                        yield self(pending)
                        pending = next(iterator, None)
            finally:
                for future, _ in queue:
                    future.cancel()

    def report(self):
        return dict(
            prefetch_record_limit=self.prefetch_records,
            prefetch_byte_limit=self.prefetch_bytes,
            peak_prefetch_reserved_bytes=self.prefetch_peak,
            prefetched_records=self.prefetch_count,
            synchronous_bypasses=self.prefetch_bypasses,
            tensor_byte_limit=self.max_bytes,
            resident_tensor_bytes=self.resident_bytes,
            peak_tensor_bytes=self.peak_bytes,
            hits=self.hits,
            misses=self.misses,
            oversized_bypasses=self.bypasses,
        )
