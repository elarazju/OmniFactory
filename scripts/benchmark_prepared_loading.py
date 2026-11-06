"""Compare uncached and bounded CPU-cache reads of the same prepared records.

This measures host loading only, not end-to-end GPU training speed. Warm every
record once for both arms, alternate arm order, and bind all observed tensor bytes.
"""

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path
from time import perf_counter

import torch

from omnifactory.artifacts import sha256
from omnifactory.training.artifacts import write_json
from omnifactory.training.data_loading import PreparedPayloadLoader
from omnifactory.training.prepare import PreparedEntries


def payload_digest(value):
    digest = hashlib.sha256()

    def visit(node):
        if isinstance(node, torch.Tensor):
            digest.update(str((node.dtype, tuple(node.shape))).encode())
            digest.update(node.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(node, dict):
            for key in sorted(node):
                digest.update(json.dumps(key).encode())
                visit(node[key])
        elif isinstance(node, (list, tuple)):
            digest.update(str(len(node)).encode())
            for item in node:
                visit(item)
        else:
            digest.update(json.dumps(node).encode())

    visit(value)
    return digest.hexdigest()


def run(prepared, *, cache_bytes, repeats, limit=None):
    prepared = Path(prepared)
    rows = list(islice(PreparedEntries(prepared, "loader-benchmark"), limit))
    if not rows or repeats < 1 or cache_bytes < 1:
        raise ValueError("nonempty data, positive repeats and cache bytes required")
    loaders = {
        "uncached": PreparedPayloadLoader(prepared),
        "cached": PreparedPayloadLoader(prepared, max_bytes=cache_bytes),
    }
    fingerprints = []
    warmup = {}
    for label, loader in loaders.items():
        started = perf_counter()
        digests = [payload_digest(loader(row)) for row in rows]
        warmup[label] = perf_counter() - started
        if label == "uncached":
            fingerprints = digests
        elif digests != fingerprints:
            raise ValueError("warmup payload mismatch")
    measurements = []
    for repeat in range(repeats):
        for label in list(loaders) if repeat % 2 == 0 else list(reversed(loaders)):
            elapsed = 0.0
            for row, expected in zip(rows, fingerprints, strict=True):
                started = perf_counter()
                payload = loaders[label](row)
                elapsed += perf_counter() - started
                if payload_digest(payload) != expected:
                    raise ValueError("measured payload mismatch")
            measurements.append(dict(arm=label, repeat=repeat, seconds=elapsed, records=len(rows)))
    times = {
        label: sum(r["seconds"] for r in measurements if r["arm"] == label) for label in loaders
    }
    return dict(
        status="passed",
        scope="warm prepared host loading; not total GPU training speedup",
        prepared=str(prepared.resolve()),
        manifest_sha256=sha256(prepared / "manifest.json"),
        index_sha256=sha256(prepared / "index.sqlite"),
        records=len(rows),
        repeats=repeats,
        cases=[
            dict(file=row["file"], sha256=row["sha256"], payload_sha256=digest)
            for row, digest in zip(rows, fingerprints, strict=True)
        ],
        warmup_including_digest_seconds=warmup,
        measurements=measurements,
        loading_seconds=times,
        loading_speed_ratio=times["uncached"] / times["cached"],
        cache=loaders["cached"].report(),
        payload_parity="byte_exact",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-bytes", type=int, default=128 * 1024**2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output file")
    torch.set_num_threads(1)
    report = run(
        args.prepared, cache_bytes=args.cache_bytes, repeats=args.repeats, limit=args.limit
    )
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
