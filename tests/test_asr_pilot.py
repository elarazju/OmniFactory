"""Data preparation must reject changed sources and keep speakers disjoint."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("soundfile")
pytest.importorskip("pyarrow")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from omnifactory.data.io import iter_samples


def script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_preparation_and_split_guards(tmp_path, monkeypatch):
    prepare = script("prepare_librispeech")
    baseline = script("run_asr_baseline")
    rows = []
    for speaker in range(10):
        for utterance in range(2):
            stream = io.BytesIO()
            wave = 0.1 * np.sin(np.arange(1600) * (speaker + 1 + utterance / 10) / 20)
            sf.write(stream, wave, 16000, format="FLAC")
            rows.append(
                dict(
                    id=f"{speaker}-0-{utterance}",
                    speaker_id=speaker,
                    text="A SHORT UTTERANCE",
                    audio=dict(bytes=stream.getvalue()),
                )
            )
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    with pytest.raises(ValueError, match="pinned"):
        prepare.prepare(source, tmp_path / "bad", train_count=4, validation_count=2)
    monkeypatch.setattr(prepare, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    report = prepare.prepare(source, tmp_path / "data", train_count=4, validation_count=2)
    train = list(iter_samples(tmp_path / "data/train.jsonl"))
    validation = list(iter_samples(tmp_path / "data/validation.jsonl"))
    baseline.check_splits(train, validation)
    assert report["counts"] == {"train": 4, "validation": 2}
    assert not report["official_test_benchmark"]
    for sample in train + validation:
        asset = sample.assets["recording"]
        assert (
            hashlib.sha256((tmp_path / "data" / asset.uri).read_bytes()).hexdigest() == asset.sha256
        )
    leaked = validation[0].model_copy(
        update={
            "provenance": validation[0].provenance.model_copy(
                update={"group_id": train[0].provenance.group_id}
            )
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        baseline.check_splits(train, [leaked])


def test_real_baseline_checks_waveform_bytes_and_reference_leakage(tmp_path):
    from omnifactory.datasets.demo import create_demo
    from omnifactory.training.config import TrainConfig

    module = script("run_real_data_baseline")
    create_demo(tmp_path / "data")
    # Keep actual audio-containing examples; the runner intentionally rejects text-only rows.
    for split in ("train", "validation"):
        path = tmp_path / f"data/{split}.jsonl"
        rows = [s for s in iter_samples(path) if s.assets]
        path.write_text("".join(s.model_dump_json() + "\n" for s in rows))
    cfg = TrainConfig(
        model="unused",
        dataset=tmp_path / "data/train.jsonl",
        validation_dataset=tmp_path / "data/validation.jsonl",
        output_dir=tmp_path / "unused",
        components={"language_model": {"mode": "full", "lr": 1e-4}},
        budget=dict(
            max_llm_tokens=4096,
            max_vision_patches=4096,
            max_audio_frames=2000,
            max_codec_frames=1,
            max_decoded_media_bytes=67108864,
        ),
    )
    rows = module.check_dataset(cfg)
    assert rows["train"] and rows["validation"]
    assert module.case_seed(42, "a", "p") == module.case_seed(42, "a", "p")
    assert module.case_seed(42, "a", "p") != module.case_seed(42, "b", "p")
    sample = rows["validation"][0]
    asset = next(iter(sample.assets.values()))
    path = tmp_path / "data" / asset.uri
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.check_dataset(cfg)
