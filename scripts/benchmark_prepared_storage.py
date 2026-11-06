"""Storage-only scale check; synthetic tensors, not a model or quality benchmark.

Run from an installed OmniFactory environment. Output contains its own disk index,
shards and report; it is deliberately not a trainable prepared dataset.
"""

import argparse
import json
import resource
import sqlite3
import sys
import time
from pathlib import Path

import torch

from omnifactory.training.data_loading import PreparedPayloadLoader
from omnifactory.training.prepare import PreparedEntries
from omnifactory.training.storage import ShardWriter, read_record


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def benchmark(output, *, records=100000, read_records=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    baseline = peak_rss()
    start = time.perf_counter()
    with sqlite3.connect(output / "index.sqlite") as db:
        db.execute(
            "CREATE TABLE entries (id INTEGER PRIMARY KEY, sample TEXT, prediction TEXT, cache_key TEXT, body TEXT, UNIQUE(sample,prediction))"
        )

        def commit(rows):
            with db:
                db.executemany(
                    "INSERT INTO entries VALUES (?,?,?,?,?)",
                    (
                        (idx, e["sample_id"], e["prediction_id"], key, json.dumps(e))
                        for idx, key, e in rows
                    ),
                )

        writer = ShardWriter(output, max_records=256, max_bytes=8 * 1024**2, commit=commit)
        for i in range(records):
            # Vary tensor kinds/shapes to exercise namespaces without a native codec.
            tensors = {
                "tokens": torch.full((128,), i, dtype=torch.int64),
                "latents": torch.full((16 + i % 4, 32), float(i)),
                "stream/valid": torch.ones(128, dtype=torch.bool),
            }
            size = sum(t.numel() * t.element_size() for t in tensors.values())
            entry = dict(sample_id=str(i), prediction_id="p0", tensor_bytes=size, cost={})
            writer.add(i, str(i), entry, tensors)
        writer.flush()
    write_seconds = time.perf_counter() - start
    entries = PreparedEntries(output, "storage-benchmark")
    loader = PreparedPayloadLoader(
        output, prefetch_records=4, prefetch_bytes=1024**2, load_payload=read_record
    )
    from itertools import islice

    count = 0
    start = time.perf_counter()
    for i, payload in enumerate(loader.iterate(islice(entries, read_records or records))):
        assert payload["tokens"][0].item() == i
        assert payload["latents"][0, 0].item() == i
        assert payload["stream/valid"].all()
        count += 1
    seconds = time.perf_counter() - start
    report = dict(
        scope="synthetic storage/index/read correctness; no native encoding or GPU throughput claim",
        records=records,
        read_records=count,
        shards=writer.shards,
        tensor_files_without_sharding=records,
        write_seconds=write_seconds,
        read_seconds=seconds,
        records_per_second=count / seconds,
        peak_shard_tensor_bytes=writer.peak_bytes,
        process_peak_rss_bytes=peak_rss(),
        baseline_peak_rss_bytes=baseline,
        peak_rss_growth_bytes=peak_rss() - baseline,
        storage_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        loader=loader.report(),
        status="passed",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=100000)
    parser.add_argument("--read-records", type=int)
    args = parser.parse_args()
    if args.records < 1 or (args.read_records is not None and args.read_records < 1):
        parser.error("record counts must be positive")
    print(
        json.dumps(
            benchmark(args.output, records=args.records, read_records=args.read_records), indent=2
        )
    )
