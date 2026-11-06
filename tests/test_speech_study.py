import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("soundfile")
import numpy as np
import soundfile as sf
import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"


def module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def upstream_rows():
    result = []
    for speaker in (101, 202, 303):
        for i in range(3):
            stream = io.BytesIO()
            sf.write(stream, np.full(160, 0.01 * (i + 1)), 16000, format="FLAC")
            result.append(
                dict(
                    id=f"{speaker}-1-{i}",
                    speaker_id=speaker,
                    text="HELLO WORLD",
                    audio=dict(bytes=stream.getvalue()),
                )
            )
    return result


def test_test_selection_is_order_independent_and_rejects_leakage(monkeypatch):
    select = module("prepare_librispeech_test", monkeypatch).select_rows
    rows = upstream_rows()
    a = select(rows, [], speakers=2, per_speaker=2)
    b = select(list(reversed(rows)), [], speakers=2, per_speaker=2)
    assert a == b
    for _, targets, reference in a:
        assert reference["id"] not in {r["id"] for r in targets}
    excluded = SimpleNamespace(
        id="old", provenance=SimpleNamespace(group_id="speaker-101"), assets={}
    )
    with pytest.raises(ValueError, match="overlaps"):
        select(rows, [excluded], speakers=2, per_speaker=2)
    excluded.provenance.group_id = "speaker-999"
    digest = hashlib.sha256(rows[0]["audio"]["bytes"]).hexdigest()
    excluded.assets = {"x": SimpleNamespace(sha256=digest)}
    with pytest.raises(ValueError, match="waveform overlaps"):
        select(rows, [excluded], speakers=2, per_speaker=2)


def test_generation_rng_is_independent_of_training_seed(tmp_path, monkeypatch):
    mod = module("run_real_data_baseline", monkeypatch)
    observed = []
    cfg = SimpleNamespace(seed=123, adapter="voxcpm2", validation_root=tmp_path)
    model = SimpleNamespace(eval=lambda: None)
    adapter = SimpleNamespace(
        resolver=SimpleNamespace(root=None),
        speech_decoder=lambda model: lambda latent: (np.zeros(160), 16000),
        encode=lambda *a, **kw: SimpleNamespace(validate=lambda **kw: SimpleNamespace(payload={})),
        to_device=lambda *a: {},
        generate_speech=lambda *a, **kw: (
            observed.append(torch.initial_seed()) or dict(audio_latents=[], speech_finished=True)
        ),
    )
    monkeypatch.setattr(mod, "load_checkpoint", lambda checkpoint: (model, adapter, cfg, None))
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    pred = SimpleNamespace(id="p0", target_message_id="answer", target_block_ids=["text"])
    sample = SimpleNamespace(
        id="case",
        predictions=[pred],
        messages=[
            SimpleNamespace(
                id="answer", content=[SimpleNamespace(id="text", type="text", text="Hello")]
            )
        ],
    )
    for generation_seed in (42, None):
        out = tmp_path / str(generation_seed)
        out.mkdir()
        mod.generate(
            cfg,
            [sample],
            out,
            "adapter",
            max_new_tokens=4,
            max_audio_units=4,
            checkpoint=tmp_path / "checkpoint",
            generation_seed=generation_seed,
        )
        row = json.loads((out / "adapter-predictions.jsonl").read_text())
        assert row["seed"] == mod.case_seed(generation_seed or 123, "case", "p0")
    assert observed == [mod.case_seed(42, "case", "p0"), mod.case_seed(123, "case", "p0")]


def test_speaker_bootstrap_does_not_count_seeds_as_new_examples(monkeypatch):
    paired = module("analyze_speech_study", monkeypatch).paired
    samples = [dict(id=str(i), provenance=dict(group_id=str(i // 2))) for i in range(4)]
    baseline = [
        dict(sample_id=str(i), reference="hello world", prediction="hello word") for i in range(4)
    ]
    improved = [dict(r, prediction="hello world") for r in baseline]
    one = paired([baseline], [improved], samples, repeats=200)
    three = paired([baseline] * 3, [improved] * 3, samples, repeats=200)
    assert one["delta_wer"] == three["delta_wer"] == -0.5
    assert (
        one["speaker_bootstrap_95_percentile"]
        == three["speaker_bootstrap_95_percentile"]
        == [-0.5, -0.5]
    )
    assert three["cases"] == 4 and three["speakers"] == 2
    with pytest.raises(ValueError, match="unpaired"):
        paired([baseline], [list(reversed(improved))], samples, repeats=200)


def test_audio_tail_metric_covers_partial_window(tmp_path, monkeypatch):
    measure = module("evaluate_speech_recipe", monkeypatch).audio_diagnostics
    path = tmp_path / "wave.wav"
    sf.write(path, np.r_[np.ones(320) * 0.1, np.zeros(327)], 16000, subtype="FLOAT")
    result = measure(path)
    assert result["seconds"] == 647 / 16000
    assert result["trailing_low_energy_seconds"] == {"-50": 327 / 16000, "-60": 327 / 16000}


def test_selection_freezes_before_paired_confirmation(tmp_path, monkeypatch):
    mod = module("analyze_speech_study", monkeypatch)
    samples = [
        dict(
            id=str(i),
            provenance=dict(group_id=str(i // 2), split="test"),
            predictions=[dict(conditions=[dict(kind="reference_audio")] if i % 2 else [])],
        )
        for i in range(4)
    ]
    dataset = tmp_path / "test.jsonl"
    dataset.write_text("".join(json.dumps(s) + "\n" for s in samples))

    def scores(folder, names, improved):
        folder.mkdir()
        energy, evidence, metrics, termination = {}, {}, {}, {}
        for name in names:
            records = [
                dict(
                    sample_id=s["id"],
                    reference="hello world",
                    prediction="hello world" if name in improved else "hello word",
                )
                for s in samples
            ]
            path = folder / f"{name}-transcripts.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            evidence[path.name] = mod.sha256(path)
            metrics[name] = mod.corpus_wer(records)
            termination[name] = dict(total=4, finished=4)
            energy[name] = [
                dict(sample_id=s["id"], trailing_low_energy_seconds={"-50": 0.1, "-60": 0.1})
                for s in samples
            ]
        for name, value in [
            ("audio-diagnostics.json", energy),
            ("identity.json", dict(dataset_sha256=mod.sha256(dataset))),
        ]:
            path = folder / name
            path.write_text(json.dumps(value))
            evidence[name] = mod.sha256(path)
        (folder / "report.json").write_text(
            json.dumps(
                dict(
                    status="completed", metrics=metrics, evidence=evidence, termination=termination
                )
            )
        )

    configs = {}
    for name in ("control", "stop_low"):
        path = tmp_path / (name + ".json")
        path.write_text("{}")
        configs[name] = dict(config=str(path), config_sha256=mod.sha256(path))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            dict(
                candidates=configs,
                development=dict(dataset_sha256=mod.sha256(dataset)),
                confirmation=dict(training_seeds=[42, 123, 456], cases=4, speakers=2),
            )
        )
    )
    dev = tmp_path / "dev"
    scores(dev, configs, {"stop_low"})
    selection = tmp_path / "selection.json"
    mod.select(protocol, dev, selection)
    assert mod.read(selection)["selected"] == "stop_low"
    with pytest.raises(ValueError, match="immutable"):
        mod.select(protocol, dev, selection)
    names = [f"{k}_s{s}" for s in (42, 123, 456) for k in ("control", "selected")] + [
        "base",
        "native_reference",
        "selected_export",
        "groundtruth",
    ]
    test = tmp_path / "scores"
    scores(test, names, {k for k in names if k.startswith("selected") or k == "groundtruth"})
    output = tmp_path / "confirmed.json"
    mod.confirm(protocol, selection, test, dataset, output)
    assert mod.read(output)["stable_improvement_supported"]
    protocol.write_text(protocol.read_text() + "\n")
    with pytest.raises(ValueError, match="mismatch"):
        mod.confirm(protocol, selection, test, dataset, tmp_path / "changed.json")
