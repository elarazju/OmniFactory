"""Reusable preparation with transactional shard commits and an on-disk index."""

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from omnifactory.adapters.runtime import adapter_class
from omnifactory.data.io import atomic_writer, iter_samples, validate_dataset
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.processor import OmniDataProcessor
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import digest, write_json
from omnifactory.training.fingerprints import file_fingerprint
from omnifactory.training.groups import GROUP_SCHEMA, GroupIndex
from omnifactory.training.storage import ShardWriter, encoded_metadata, read_record


def source_fingerprint(model: str, revision: str | None, *, refresh=False):
    root = Path(model)
    if root.is_dir():
        from omnifactory.training.fingerprints import cached_hash, stamp

        files = [p for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith(".")]
        signature = [(str(p.relative_to(root)), stamp(p)) for p in files]

        def compute():
            hasher = hashlib.sha256()
            for path in files:
                hasher.update(str(path.relative_to(root)).encode())
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        hasher.update(chunk)
            if signature != [(str(p.relative_to(root)), stamp(p)) for p in files]:
                raise OmniFactoryError("E_SOURCE_CHANGED", "model changed while fingerprinting")
            return hasher.hexdigest()

        return cached_hash(
            "directory:" + str(root.resolve()), str(signature), compute, refresh=refresh
        )
    if (
        revision is None
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise OmniFactoryError(
            "E_REVISION", "remote training requires a pinned 40-character Hub commit revision"
        )
    return digest({"model": model, "revision": revision})


def dataset_fingerprint(config):
    resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
    h = hashlib.sha256()
    for sample in iter_samples(config.dataset):
        h.update(sample.model_dump_json().encode())
        for key, asset in sorted(sample.assets.items()):
            # Resolve and validate declared hashes, without decoding media.
            resolved = resolver.resolve(asset)
            h.update(key.encode())
            h.update(resolved.sha256.encode())
    return h.hexdigest()


def encoder_fingerprint(adapter):
    """Invalidate prepared targets after encoder/schema/vendor implementation changes."""
    import inspect

    from omnifactory.adapters.runtime import adapter_spec

    package = Path(__file__).resolve().parents[1]
    paths = {Path(__file__).resolve(), Path(__file__).with_name("storage.py").resolve()}
    for folder in ("data", "adapters", "_vendor"):
        paths.update((package / folder).rglob("*.py"))
    # Third-party direct implementation changes are covered; its dependency
    # semantics must be versioned by the plugin's encoding_version.
    implementation = inspect.getsourcefile(adapter_class(adapter))
    if implementation:
        paths.add(Path(implementation))
    return digest(
        {
            "version": adapter_spec(adapter).encoding_version,
            "sources": sorted(
                (
                    str(p.relative_to(package)) if p.is_relative_to(package) else p.name,
                    file_fingerprint(p),
                )
                for p in paths
            ),
        }
    )


def contracts(config, source):
    # Policy belongs to the execution contract, not cached encoder targets.
    encoding = digest(
        {
            "format": 4,
            "encoder": encoder_fingerprint(config.adapter),
            "source": source,
            "config": config.encoding_contract(),
            "aux": {
                str(p): source_fingerprint(str(p), None) if p.is_dir() else file_fingerprint(p)
                for p in (
                    config.duplex_schedule,
                    config.duplex_text_alignment,
                    config.speech_codes,
                    config.speech_tokenizer,
                    config.speech_assets,
                    config.speech_reference,
                )
                if p is not None
            },
        }
    )
    execution = digest(
        {
            "source": source,
            "execution": config.execution_contract(),
            "encoding": encoding,
            "profile": file_fingerprint(config.memory_profile) if config.memory_profile else None,
        }
    )
    return encoding, execution


class PreparedEntries(Sequence):
    """Bounded-memory random/streaming access; tuple keys serve the batch planner."""

    def __init__(self, root, contract):
        self.root, self.contract = Path(root), contract

    def _db(self):
        return sqlite3.connect(
            f"{(self.root / 'index.sqlite').resolve().as_uri()}?mode=ro", uri=True
        )

    def __len__(self):
        with closing(self._db()) as db:
            return db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def _decode(self, value):
        row = json.loads(value)
        row["cost"]["contract_id"] = self.contract
        return row

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        with closing(self._db()) as db:
            if isinstance(key, tuple):
                row = db.execute(
                    "SELECT body FROM entries WHERE sample=? AND prediction=?", key
                ).fetchone()
            else:
                if key < 0:
                    key += len(self)
                row = db.execute("SELECT body FROM entries WHERE id=?", (key,)).fetchone()
        if row is None:
            raise IndexError(key)
        return self._decode(row[0])

    def __iter__(self):
        with closing(self._db()) as db:
            for row in db.execute("SELECT body FROM entries ORDER BY id"):
                yield self._decode(row[0])


def load_prepared(config, root, *, check_inputs=True):
    root = Path(root)
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata.get("format") != "omnifactory-prepared-v4":
        raise OmniFactoryError(
            "E_PREPARED_VERSION", "regenerate prepared data with the current version"
        )
    source = source_fingerprint(config.model, config.revision)
    encoding, execution = contracts(config, source)
    if encoding != metadata["encoding_contract"] or source != metadata["source_fingerprint"]:
        raise OmniFactoryError(
            "E_PREPARED_CONTRACT", "model/encoder settings differ; prepare again"
        )
    if check_inputs and dataset_fingerprint(config) != metadata["input_fingerprint"]:
        raise OmniFactoryError("E_PREPARED_INPUT", "dataset or media changed; prepare again")
    if file_fingerprint(root / "index.sqlite") != metadata["index_sha256"]:
        raise OmniFactoryError("E_PREPARED_HASH", "prepared index changed")
    if source_fingerprint(str(root / "processor"), None) != metadata["processor_sha256"]:
        raise OmniFactoryError("E_PREPARED_HASH", "prepared processor changed")
    metadata["contract_id"] = execution
    metadata["entries"] = PreparedEntries(root, execution)
    metadata["groups"] = GroupIndex(root / "index.sqlite")
    from omnifactory.planning.batching import SampleCost
    from omnifactory.training.calibration import configured_planner

    planner = configured_planner(config, execution, source)
    for entry in metadata["entries"]:
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise OmniFactoryError("E_PREPARED_PATH", "tensor path leaves artifact")
        if file_fingerprint(path) != entry["sha256"]:
            raise OmniFactoryError("E_PREPARED_HASH", f"prepared tensor changed: {path}")
        cost = SampleCost.model_validate(entry["cost"])
        failures = planner.exceeded(planner.cost([cost]))
        if failures:
            raise OmniFactoryError("E_SAMPLE_OVER_BUDGET", f"{cost.sample_id}: {failures}")
    return metadata


def _prepare(config, output: Path, *, split="train", resume=False, reuse_from=None):
    import fcntl

    output = output.resolve()
    if output.exists():
        raise OmniFactoryError("E_OUTPUT_EXISTS", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name("." + output.name + ".partial")
    # Lock persists outside the renamed directory, so concurrent writers cannot race.
    with output.with_name("." + output.name + ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OmniFactoryError("E_PREPARE_BUSY", str(output)) from None
        if output.exists():
            raise OmniFactoryError("E_OUTPUT_EXISTS", str(output))
        if partial.exists() and not resume:
            raise OmniFactoryError(
                "E_PREPARE_PARTIAL", "interrupted preparation exists; use --resume"
            )
        report = validate_dataset(config.dataset)
        if report.invalid or not report.records:
            raise OmniFactoryError("E_DATASET", "; ".join(report.errors))
        source = source_fingerprint(config.model, config.revision)
        encoding, execution = contracts(config, source)
        inputs = dataset_fingerprint(config)
        identity = dict(
            encoding_contract=encoding,
            source_fingerprint=source,
            input_fingerprint=inputs,
            split=split,
            storage={
                "max_records": config.prepared_shard_records,
                "max_bytes": config.prepared_shard_bytes,
            },
        )
        partial.mkdir(exist_ok=True)
        state = partial / "state.json"
        if state.exists() and json.loads(state.read_text()) != identity:
            raise OmniFactoryError(
                "E_PREPARE_CONTRACT", "interrupted preparation inputs/settings changed"
            )
        write_json(state, identity)
        old = None
        if reuse_from:
            old_root = Path(reuse_from)
            old_meta = json.loads((old_root / "manifest.json").read_text())
            if old_meta.get("encoding_contract") != encoding:
                raise OmniFactoryError(
                    "E_PREPARED_CONTRACT", "incremental source uses a different encoder"
                )
            if file_fingerprint(old_root / "index.sqlite") != old_meta["index_sha256"]:
                raise OmniFactoryError("E_PREPARED_HASH", "incremental source index changed")
            if (
                source_fingerprint(str(old_root / "processor"), None)
                != old_meta["processor_sha256"]
            ):
                raise OmniFactoryError("E_PREPARED_HASH", "incremental processor changed")
            old = sqlite3.connect(f"file:{old_root / 'index.sqlite'}?mode=ro", uri=True)
        adapter = None
        resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        from omnifactory.training.calibration import configured_planner

        planner = configured_planner(config, execution, source)
        reused, encoded_count = 0, 0
        try:
            with sqlite3.connect(partial / "index.sqlite") as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY, sample TEXT, prediction TEXT, cache_key TEXT, body TEXT, UNIQUE(sample,prediction))"
                )
                db.execute("CREATE INDEX IF NOT EXISTS cached ON entries(cache_key)")
                db.execute(GROUP_SCHEMA)

                def commit(rows):
                    with db:
                        db.executemany(
                            "INSERT INTO entries VALUES (?,?,?,?,?)",
                            (
                                (idx, e["sample_id"], e["prediction_id"], key, json.dumps(e))
                                for idx, key, e in rows
                            ),
                        )

                writer = ShardWriter(
                    partial,
                    max_records=config.prepared_shard_records,
                    max_bytes=config.prepared_shard_bytes,
                    commit=commit,
                )
                i = 0
                for sample in iter_samples(config.dataset):
                    if sample.provenance.split != split:
                        raise OmniFactoryError(
                            "E_SPLIT", f"expected only {split} rows, got {sample.provenance.split}"
                        )
                    db.execute(
                        "INSERT OR IGNORE INTO groups VALUES (?,?)",
                        (sample.provenance.dataset, sample.provenance.group_id),
                    )
                    sample_hash = resolver.manifest(sample).fingerprint
                    for prediction in sample.predictions:
                        key = digest([encoding, sample_hash, prediction.id])
                        existing = db.execute(
                            "SELECT cache_key, body FROM entries WHERE id=?", (i,)
                        ).fetchone()
                        if existing:
                            entry = json.loads(existing[1])
                            if (
                                existing[0] != key
                                or file_fingerprint(partial / entry["file"]) != entry["sha256"]
                            ):
                                raise OmniFactoryError(
                                    "E_PREPARED_HASH", "committed preparation record changed"
                                )
                            reused += 1
                        else:
                            cached = (
                                old.execute(
                                    "SELECT body FROM entries WHERE cache_key=?", (key,)
                                ).fetchone()
                                if old
                                else None
                            )
                            if cached:
                                entry = json.loads(cached[0])
                                tensors = read_record(old_root, entry)
                                reused += 1
                            else:
                                if adapter is None:
                                    adapter = adapter_class(config.adapter).from_config(config)
                                encoded = OmniDataProcessor().encode(
                                    sample, prediction.id, adapter=adapter, contract_id=encoding
                                )
                                from omnifactory.adapters.contracts import validate_encoded
                                from omnifactory.adapters.runtime import adapter_spec

                                validate_encoded(adapter_spec(config.adapter), config.task, encoded)
                                cost = encoded.cost.model_copy(update={"contract_id": execution})
                                failures = planner.exceeded(planner.cost([cost]))
                                if failures:
                                    raise OmniFactoryError(
                                        "E_SAMPLE_OVER_BUDGET", f"{sample.id}: {failures}"
                                    )
                                entry = encoded_metadata(encoded)
                                tensors = encoded.tensors
                                encoded_count += 1
                            writer.add(i, key, entry, tensors)
                            del tensors
                        i += 1
                writer.flush()
                db.commit()
                # A processor is small; reuse it when all records were cached.
                processor_marker = partial / "processor.ready"
                if not processor_marker.exists():
                    shutil.rmtree(partial / "processor", ignore_errors=True)
                    if adapter is None and old is not None:
                        shutil.copytree(old_root / "processor", partial / "processor")
                    else:
                        if adapter is None:
                            adapter = adapter_class(config.adapter).from_config(config)
                        adapter.save_processor(partial / "processor")
                    write_json(
                        processor_marker,
                        {"sha256": source_fingerprint(str(partial / "processor"), None)},
                    )
                elif json.loads(processor_marker.read_text())["sha256"] != source_fingerprint(
                    str(partial / "processor"), None
                ):
                    raise OmniFactoryError("E_PREPARED_HASH", "interrupted processor changed")
                fingerprint = hashlib.sha256()
                with (partial / "costs.jsonl").open("w") as costs:
                    for (body,) in db.execute("SELECT body FROM entries ORDER BY id"):
                        fingerprint.update(body.encode())
                        costs.write(json.dumps(json.loads(body)["cost"]) + "\n")
            if dataset_fingerprint(config) != inputs:
                raise OmniFactoryError("E_PREPARE_CONTRACT", "dataset changed during preparation")
            metadata = dict(
                format="omnifactory-prepared-v4",
                **identity,
                contract_id=execution,
                fingerprint=fingerprint.hexdigest(),
                records=i,
                index_sha256=file_fingerprint(partial / "index.sqlite", refresh=True),
                processor_sha256=source_fingerprint(str(partial / "processor"), None),
                preparation=dict(
                    encoded=encoded_count,
                    reused=reused,
                    new_shards=writer.shards,
                    peak_shard_tensor_bytes=writer.peak_bytes,
                    oversized_records=writer.oversized,
                ),
            )
            # Small fixtures remain inspectable; large manifests never inline their index.
            if i <= 256:
                metadata["entries"] = list(PreparedEntries(partial, execution))
            with atomic_writer(partial / "manifest.json") as stream:
                stream.write(json.dumps(metadata, indent=2))
            partial.rename(output)
        finally:
            if old:
                old.close()
    return load_prepared(config, output, check_inputs=False)


def save_encoded(root, encoded, index):
    """Persist one standalone record (also used by adapter contract tests)."""
    result = []
    writer = ShardWriter(
        root,
        max_records=1,
        max_bytes=256 * 1024**2,
        commit=lambda rows: result.extend(e for _, _, e in rows),
    )
    writer.add(index, "", encoded_metadata(encoded), encoded.tensors)
    return result[0]


def prepare(config, output: Path, *, split="train", resume=False, reuse_from=None):
    import torch

    devices = [torch.cuda.current_device()] if config.device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        return _prepare(config, output, split=split, resume=resume, reuse_from=reuse_from)
