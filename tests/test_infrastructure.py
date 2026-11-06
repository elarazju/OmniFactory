"""Prepared reuse, failure recovery and long-run controls have behavioral guarantees."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
import torch
from safetensors.torch import load_file

from omnifactory.adapters.qwen2_5 import Qwen25Adapter
from omnifactory.datasets.demo import create_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train
from omnifactory.training.prepare import load_prepared, prepare
from omnifactory.training.tiny import create_tiny


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(2)
    create_tiny(tmp_path / "model")
    create_demo(tmp_path / "data")
    c = TrainConfig(
        model=str(tmp_path / "model"),
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "run",
        max_steps=4,
        save_steps=1,
        max_length=512,
        components={"language_model": {"mode": "lora", "rank": 2, "alpha": 4, "lr": 0.001}},
        budget=dict(
            max_examples=2,
            max_llm_tokens=1024,
            max_vision_patches=128,
            max_audio_frames=400,
            max_codec_frames=1,
            max_decoded_media_bytes=1000000,
        ),
    )
    return tmp_path, c


def assert_weights_equal(left, right):
    a, b = (load_file(Path(p) / "trainable.safetensors") for p in (left, right))
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_prepared_reuse_avoids_encoding_and_preserves_training(setup, monkeypatch):
    root, c = setup
    prepare(c, root / "prepared")
    fresh = train(c)

    def forbidden(*args, **kwargs):
        raise AssertionError("reused data must not encode")

    monkeypatch.setattr(Qwen25Adapter, "encode", forbidden)
    reused = train(
        c.model_copy(update={"output_dir": root / "reused", "prepared_dataset": root / "prepared"})
    )
    assert_weights_equal(fresh["checkpoint"], reused["checkpoint"])
    changed_lr = c.model_copy(
        update={
            "components": {
                "language_model": c.components["language_model"].model_copy(update={"lr": 0.002})
            }
        }
    )
    load_prepared(changed_lr, root / "prepared")
    with pytest.raises(OmniFactoryError, match="E_PREPARED_CONTRACT"):
        load_prepared(c.model_copy(update={"max_length": 1024}), root / "prepared")
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    (root / "prepared" / manifest["entries"][0]["file"]).write_bytes(b"damaged")
    with pytest.raises(OmniFactoryError, match="E_PREPARED_HASH"):
        load_prepared(c, root / "prepared")


def test_prepared_cpu_cache_preserves_updates_and_resume_cursor(setup):
    root, cfg = setup
    cfg = cfg.model_copy(update={"max_steps": 4, "gradient_accumulation_steps": 4})
    prepare(cfg, root / "prepared")
    cfg = cfg.model_copy(update={"prepared_dataset": root / "prepared"})
    reference = train(cfg)
    cached = cfg.model_copy(
        update={"output_dir": root / "cached", "prepared_cache_bytes": 16 * 1024**2}
    )
    assert cached.encoding_contract() == cfg.encoding_contract()
    assert cached.execution_contract() == cfg.execution_contract()
    result = train(cached)
    assert_weights_equal(reference["checkpoint"], result["checkpoint"])
    rows = [
        json.loads(row) for row in (cached.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["prepared_cache"]["hits"] > 0
    resumed = train(
        cached.model_copy(
            update={
                "output_dir": root / "resumed-cache",
                "resume_from": root / "run/checkpoint-00000002",
            }
        )
    )
    assert_weights_equal(reference["checkpoint"], resumed["checkpoint"])


def test_prepare_resume_and_incremental_reuse(setup, monkeypatch):
    root, c = setup
    c = c.model_copy(update={"prepared_shard_records": 2})
    original = Qwen25Adapter.encode
    count = 0

    def interrupted(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("interrupted")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Qwen25Adapter, "encode", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare(c, root / "prepared")
    assert not (root / "prepared").exists()
    monkeypatch.setattr(Qwen25Adapter, "encode", original)
    result = prepare(c, root / "prepared", resume=True)
    assert result["preparation"]["reused"] == 2

    def forbidden(*args, **kwargs):
        raise AssertionError("incremental reuse must not encode")

    monkeypatch.setattr(Qwen25Adapter, "encode", forbidden)
    copy = prepare(c, root / "copy", reuse_from=root / "prepared")
    assert copy["preparation"]["encoded"] == 0
    assert copy["fingerprint"] == result["fingerprint"]


def test_scheduler_eval_resume_and_retention(setup):
    root, c = setup
    c = c.model_copy(
        update=dict(
            lr_scheduler="cosine",
            scheduler_steps=6,
            warmup_steps=2,
            max_steps=6,
            validation_dataset=root / "data/validation.jsonl",
            eval_steps=2,
        )
    )
    result = train(c)
    resumed = train(
        c.model_copy(
            update=dict(
                output_dir=root / "resume",
                resume_from=root / "run/checkpoint-00000002",
                save_total_limit=2,
            )
        )
    )
    assert_weights_equal(result["checkpoint"], resumed["checkpoint"])
    assert result["best_metric"] == resumed["best_metric"]
    assert len(list((root / "resume").glob("checkpoint-*"))) <= 2
    records = [json.loads(x) for x in (root / "run/metrics.jsonl").read_text().splitlines()]
    assert records[0]["learning_rates"][0] == 0.0005
    assert records[-1]["learning_rates"][0] == 0
    assert len((root / "run/evaluation.jsonl").read_text().splitlines()) == 3


def test_shared_acceptance_and_sealed_export(setup, monkeypatch):
    from omnifactory.acceptance import attach_gate, run_acceptance
    from omnifactory.artifacts import verify_export

    root, c = setup
    c = c.model_copy(update={"validation_dataset": root / "data/validation.jsonl"})
    report = run_acceptance(c, root / "acceptance")
    assert all(
        report.gates[n].status == "passed"
        for n in ("prepare", "train", "resume", "export", "inference")
    )
    assert report.gates["http"].status == "not_run"
    assert report.gates["quality"].status == "not_run"
    assert verify_export(root / "acceptance/export")["artifact_id"] == report.artifact_id
    import omnifactory.serving as serving

    def respond(url, body):
        assert url.endswith("/v1/chat/completions")
        assert body["modalities"] == ["text"]
        return 200, json.dumps({"choices": [{"message": {"content": "generated"}}]}).encode(), {}

    monkeypatch.setattr(serving, "http_request", respond)
    http = serving.probe(
        root / "acceptance/export",
        root / "data/validation.jsonl",
        root / "http-probe",
        "http://localhost",
        "fixture",
    )
    assert http["artifact_id"] == report.artifact_id
    assert http["modality"] == "text"
    assert http["requests"] > 0
    gate = root / "http.json"
    gate.write_text(json.dumps({"status": "passed", "artifact_id": "wrong"}))
    with pytest.raises(OmniFactoryError, match="E_ACCEPTANCE_ARTIFACT"):
        attach_gate(root / "acceptance/acceptance.json", gate, name="http")
    config = root / "acceptance/export/config.json"
    config.write_text(config.read_text() + " ")
    with pytest.raises(OmniFactoryError, match="E_EXPORT_HASH"):
        verify_export(root / "acceptance/export")


def test_prepare_recovers_interrupted_processor(setup, monkeypatch):
    root, c = setup
    original = Qwen25Adapter.save_processor

    def fail(self, path):
        path.mkdir()
        (path / "incomplete").write_text("partial")
        raise RuntimeError("processor interrupted")

    monkeypatch.setattr(Qwen25Adapter, "save_processor", fail)
    with pytest.raises(RuntimeError, match="processor interrupted"):
        prepare(c, root / "prepared")
    monkeypatch.setattr(Qwen25Adapter, "save_processor", original)
    result = prepare(c, root / "prepared", resume=True)
    assert result["preparation"]["encoded"] == 0
    assert not (root / "prepared/processor/incomplete").exists()
    (root / "prepared/processor/modified").write_text("changed")
    with pytest.raises(OmniFactoryError, match="E_PREPARED_HASH"):
        prepare(c, root / "copy", reuse_from=root / "prepared")


def test_acceptance_failure_keeps_gate_evidence(setup, monkeypatch):
    from omnifactory.acceptance import AcceptanceReport, run_acceptance

    root, c = setup

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate preparation failure")

    monkeypatch.setattr(Qwen25Adapter, "encode", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        run_acceptance(c, root / "failed")
    report = AcceptanceReport.model_validate_json((root / "failed/acceptance.json").read_text())
    assert report.gates["prepare"].status == "failed"
    assert report.gates["train"].status == "not_run"


def test_external_adapter_entrypoint(setup, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import omnifactory.adapters.runtime as runtime
    from omnifactory.training.export import export_checkpoint

    root, cfg = setup
    # Reuse a known adapter implementation under a plugin-owned name. The engine
    # must dispatch loader/policy/export without introducing a family branch.
    spec = replace(runtime.adapter_spec("qwen2_5_omni"), name="test_plugin", validator=None)
    monkeypatch.setattr(runtime, "_REGISTRY", dict(runtime._REGISTRY))
    monkeypatch.setattr(runtime, "_PLUGINS_LOADED", False)
    monkeypatch.setattr(
        runtime,
        "entry_points",
        lambda **kw: [SimpleNamespace(name="test_plugin", load=lambda: lambda: spec)],
    )
    cfg = TrainConfig.model_validate({**cfg.model_dump(), "adapter": "test_plugin"})
    assert runtime.adapter_class("test_plugin") is Qwen25Adapter
    result = train(cfg)
    # A third-party exporter gets the same checkpoint and final integrity seal.
    from omnifactory.training.export import export_builtin

    def plugin_export(checkpoint, output, **kwargs):
        return export_builtin(checkpoint, output, **kwargs)

    runtime._REGISTRY["test_plugin"] = replace(spec, exporter=plugin_export)
    export_checkpoint(Path(result["checkpoint"]), root / "plugin-export", format="merged")
    assert (root / "plugin-export/artifact-manifest.json").exists()


def test_export_header_rejects_shape_byte_mismatch(tmp_path):
    import struct

    from omnifactory.artifacts import tensor_header

    path = tmp_path / "invalid.safetensors"
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    with pytest.raises(OmniFactoryError, match="E_EXPORT_TENSOR"):
        tensor_header(path)


def test_quality_comparison_binds_cases_references_and_waveforms(tmp_path):
    import numpy as np
    import soundfile as sf

    from omnifactory.artifacts import sha256
    from omnifactory.quality import compare_transcripts

    wave = tmp_path / "generated.wav"
    sf.write(wave, np.sin(np.arange(1000) * 0.1), 16000)
    row = dict(
        sample_id="one",
        reference="hello world",
        prediction="hello",
        audio=str(wave),
        audio_sha256=sha256(wave),
    )
    a, b = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    a.write_text(json.dumps(row) + "\n")
    b.write_text(json.dumps({**row, "prediction": "hello world"}) + "\n")
    result = compare_transcripts(
        a, b, tmp_path / "quality.json", artifact_id="export", scorer="fixture"
    )
    assert result["baseline"]["wer"] == 0.5
    assert result["tuned"]["wer"] == 0
    b.write_text(json.dumps({**row, "sample_id": "another"}) + "\n")
    with pytest.raises(OmniFactoryError, match="E_QUALITY_PAIR"):
        compare_transcripts(a, b, tmp_path / "bad.json", artifact_id="export", scorer="fixture")
    wave.write_bytes(b"tampered")
    with pytest.raises(OmniFactoryError, match="E_QUALITY_HASH"):
        compare_transcripts(a, b, tmp_path / "bad.json", artifact_id="export", scorer="fixture")


def test_large_preparation_streams_index_and_rejects_encoder_revision(setup, monkeypatch):
    from dataclasses import replace

    import omnifactory.adapters.runtime as runtime
    from omnifactory.training.prepare import PreparedEntries

    root, c = setup
    sample = json.loads(c.dataset.read_text().splitlines()[0])
    rows = [{**sample, "id": f"sample-{i}"} for i in range(257)]
    c.dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = prepare(c, root / "large")
    disk = json.loads((root / "large/manifest.json").read_text())
    assert "entries" not in disk
    entries = manifest["entries"]
    assert isinstance(entries, PreparedEntries)
    assert len(entries) == 257
    assert entries[-1]["sample_id"] == "sample-256"
    assert entries[("sample-256", entries[-1]["prediction_id"])] == entries[-1]
    monkeypatch.setattr(runtime, "_REGISTRY", dict(runtime._REGISTRY))
    spec = runtime.adapter_spec(c.adapter)
    runtime._REGISTRY[c.adapter] = replace(spec, encoding_version="next")
    with pytest.raises(OmniFactoryError, match="E_PREPARED_CONTRACT"):
        load_prepared(c, root / "large")


def test_asr_cache_requires_identical_generated_bytes_and_scorer(tmp_path):
    import runpy
    from importlib.metadata import version
    from types import SimpleNamespace

    from omnifactory.artifacts import sha256
    from omnifactory.training.benchmark import runtime_hashes

    helper = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/validate_infrastructure_gpu.py")
    )["reuse_transcripts"]
    cache, output = tmp_path / "cache", tmp_path / "new"
    (cache / "acceptance").mkdir(parents=True)
    (cache / "environment").mkdir()
    output.mkdir()
    wave = tmp_path / "wave.wav"
    wave.write_bytes(b"exact waveform fixture")
    (cache / "quality.json").write_text(json.dumps({"scorer": "pinned"}))
    (cache / "acceptance/tested-code.json").write_text(json.dumps(runtime_hashes()))
    packages = {name: version(name) for name in ("torch", "transformers", "tokenizers", "librosa")}
    (cache / "environment/environment.json").write_text(json.dumps({"packages": packages}))
    row = dict(
        sample_id="one",
        audio=str(wave),
        audio_sha256=sha256(wave),
        reference="hello",
        prediction="hello",
    )
    for label in ("base", "tuned"):
        (cache / f"{label}-transcripts.jsonl").write_text(json.dumps(row) + "\n")
    sample = SimpleNamespace(
        id="one", messages=[SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])]
    )
    sets = {label: [row] for label in ("base", "tuned")}
    assert helper(cache, "other-scorer", [sample], sets, output) is None
    assert len(helper(cache, "pinned", [sample], sets, output)) == 2
    wave.write_bytes(b"changed")
    assert helper(cache, "pinned", [sample], sets, output) is None


def test_deterministic_run_scopes_backend_flags_and_contract(setup, monkeypatch):
    from omnifactory.training.control import deterministic_training

    _, cfg = setup
    enabled = cfg.model_copy(update={"deterministic": True})
    assert enabled.encoding_contract() == cfg.encoding_contract()
    assert enabled.execution_contract() != cfg.execution_contract()
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    with pytest.raises(RuntimeError, match="intentional"):
        with deterministic_training(enabled):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.backends.cudnn.benchmark
            raise RuntimeError("intentional")
    assert (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark) == before
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(OmniFactoryError, match="E_DETERMINISTIC_CUDA"):
        with deterministic_training(enabled.model_copy(update={"device": "cuda"})):
            pytest.fail("CUDA execution requires an explicit deterministic workspace")


def test_incremental_changed_record_repacked_shards_without_reencoding_rest(setup, monkeypatch):
    from omnifactory.training.engine import load_encoded

    root, cfg = setup
    cfg = cfg.model_copy(update={"prepared_shard_records": 5})
    original = prepare(cfg, root / "prepared")
    assert len({e["file"] for e in original["entries"]}) == 3
    rows = [json.loads(line) for line in cfg.dataset.read_text().splitlines()]
    rows[0]["messages"][-1]["content"][0]["text"] = "Changed response"
    cfg.dataset.write_text("\n".join(map(json.dumps, rows)) + "\n")
    encode = Qwen25Adapter.encode
    calls = []

    def track(self, sample, *args, **kwargs):
        calls.append(sample.id)
        return encode(self, sample, *args, **kwargs)

    monkeypatch.setattr(Qwen25Adapter, "encode", track)
    updated = prepare(
        cfg.model_copy(update={"prepared_shard_records": 4}),
        root / "updated",
        reuse_from=root / "prepared",
    )
    assert calls == ["train-000"]
    assert updated["preparation"]["reused"] == 11
    assert load_encoded(root / "updated", updated["entries"][0]).answer == "Changed response"
    for before, after in zip(list(original["entries"])[1:], list(updated["entries"])[1:]):
        a, b = load_encoded(root / "prepared", before), load_encoded(root / "updated", after)
        assert a.supervision == b.supervision
        for key in a.tensors:
            torch.testing.assert_close(a.tensors[key], b.tensors[key], atol=0, rtol=0)


def test_synchronous_and_prefetch_training_have_exact_same_weights(setup):
    root, cfg = setup
    cfg = cfg.model_copy(update={"gradient_accumulation_steps": 3})
    a = train(cfg.model_copy(update={"prepared_prefetch_records": 0}))
    b = train(cfg.model_copy(update={"output_dir": root / "prefetched"}))
    assert_weights_equal(a["checkpoint"], b["checkpoint"])
