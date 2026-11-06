import importlib.util
import json
from pathlib import Path

import pytest

from omnifactory.data.schema import CanonicalSample
from omnifactory.training.asr_metrics import words


def script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for split, group in [("train", "a"), ("validation", "b")]:
        rows = []
        for i in range(2):
            sample = CanonicalSample.model_validate(
                dict(
                    schema_version="0.1",
                    id=f"{group}-{i}",
                    provenance=dict(
                        dataset="fixture", source_id=f"{group}-{i}", split=split, group_id=group
                    ),
                    assets={
                        "recording": dict(
                            uri=f"{group}-{i}.flac", modality="audio", sha256=str(i) * 64
                        )
                    },
                    messages=[
                        dict(
                            id="user",
                            role="user",
                            content=[
                                dict(id="audio", type="audio", media=dict(asset_id="recording"))
                            ],
                        ),
                        dict(
                            id="answer",
                            role="assistant",
                            content=[dict(id="text", type="text", text="HELLO MARY AND NASA")],
                        ),
                    ],
                    predictions=[
                        dict(id="p0", target_message_id="answer", target_block_ids=["text"])
                    ],
                )
            )
            rows.append(sample.model_dump_json())
        (root / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
    return root


def test_casing_is_explicit_and_preserves_pairs_and_source_text(tmp_path):
    prepare = script("prepare_librispeech_tts").prepare
    root = source(tmp_path)
    for mode in ("original", "sentence"):
        output = tmp_path / mode
        prepare(root, output, count=2, text_case=mode)
        changes = json.loads((output / "text-normalization.json").read_text())
        assert len(changes) == 4
        assert all(words(r["original"]) == words(r["transformed"]) for r in changes)
        for split in ("train", "validation"):
            rows = [json.loads(x) for x in (output / f"{split}.jsonl").read_text().splitlines()]
            for row in rows:
                assert (
                    row["messages"][0]["content"][0]["text"]
                    == row["messages"][1]["content"][0]["text"]
                )
                assert row["messages"][0]["content"][0]["text"] == (
                    "HELLO MARY AND NASA" if mode == "original" else "Hello mary and nasa"
                )
            assert rows[0]["predictions"][0]["conditions"] == []
            assert rows[1]["assets"]["reference"]["uri"] != rows[1]["assets"]["recording"]["uri"]
            assert rows[1]["assets"]["reference"]["uri"] == rows[0]["assets"]["recording"]["uri"]
    path = root / "train.jsonl"
    path.write_text(path.read_text().replace("HELLO MARY AND NASA", "Hello Mary and NASA"))
    with pytest.raises(ValueError, match="uppercase"):
        prepare(root, tmp_path / "bad", count=2, text_case="sentence")
    assert not (tmp_path / "bad").exists()


def test_paired_analysis_refuses_different_references_and_missing_cases(tmp_path):
    root = source(tmp_path)
    dataset = root / "validation.jsonl"
    prepare = script("prepare_librispeech_tts").prepare
    prepare(root, tmp_path / "tts", count=2)
    dataset = tmp_path / "tts/validation.jsonl"
    output = tmp_path / "baseline"
    output.mkdir()
    records = [
        dict(sample_id=f"b-{i}", reference="HELLO MARY AND NASA", prediction="hello mary and nasa")
        for i in range(2)
    ]
    for arm in ("base", "adapter", "export"):
        (output / f"{arm}-transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    analyze = script("analyze_real_data_baseline").analyze
    assert analyze(output, dataset, repeats=100)["pairs"]["base->adapter"]["delta_wer"] == 0
    changed = [dict(records[0], reference="DIFFERENT"), records[1]]
    path = output / "adapter-transcripts.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in changed))
    with pytest.raises(ValueError, match="identity/reference"):
        analyze(output, dataset, repeats=100)


def test_asr_prompt_variant_keeps_reference_and_media(tmp_path):
    root = source(tmp_path)
    script("prepare_asr_recipe").prepare(root, tmp_path / "asr")
    for split in ("train", "validation"):
        old = [json.loads(x) for x in (root / f"{split}.jsonl").read_text().splitlines()]
        new = [
            json.loads(x) for x in (tmp_path / "asr" / f"{split}.jsonl").read_text().splitlines()
        ]
        for a, b in zip(old, new, strict=True):
            assert b["messages"][0]["role"] == "system"
            assert b["messages"][1:] == a["messages"]
            assert b["provenance"] == a["provenance"] and b["predictions"] == a["predictions"]
            assert b["assets"]["recording"]["sha256"] == a["assets"]["recording"]["sha256"]
            assert Path(b["assets"]["recording"]["uri"]) == root / a["assets"]["recording"]["uri"]


def test_merge_probe_keeps_runtime_rotary_buffers_in_fp32(monkeypatch):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    module = script("diagnose_merge")

    class Native(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=dtype))
            self.register_buffer(
                "frequency", torch.tensor([0.123456789], dtype=torch.float32), persistent=False
            )

    source = Native(torch.float32)
    native = Native(torch.bfloat16)
    monkeypatch.setattr(module, "load_model", lambda cfg: native)
    result = module.set_model_dtype(
        source, SimpleNamespace(adapter="qwen2_5_omni", device="cpu", dtype="bfloat16")
    )
    assert result.weight.dtype == torch.bfloat16
    assert result.frequency.dtype == torch.float32
    assert torch.equal(result.frequency, source.frequency)


@pytest.mark.parametrize("seconds", [2, 30.08, 40.96])
def test_speech_scorer_covers_long_waveforms_through_native_media_loader(
    tmp_path, monkeypatch, seconds
):
    from types import SimpleNamespace

    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    from omnifactory.adapters.qwen3 import Qwen3Adapter
    from omnifactory.training.config import read_config

    module = script("validate_infrastructure_gpu")
    wave = tmp_path / "speech.wav"
    frames = round(seconds * 16000)
    sf.write(wave, np.zeros(frames, dtype=np.float32), 16000)
    observed = []
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(all_special_tokens=[]),
        feature_extractor=SimpleNamespace(chunk_length=30),
    )

    class Adapter(Qwen3Adapter):
        processor_class = SimpleNamespace(from_pretrained=lambda *a, **kw: processor)

        def encode(self, sample, prediction, **kwargs):
            assert self.max_audio_seconds == 30  # Exercise the real constructor's cap.
            _, _, audio, _, _ = self._messages(sample, prediction)
            assert len(audio) == 1 and len(audio[0]) <= 30 * 16000
            observed.append(len(audio[0]))
            return SimpleNamespace(payload={})

        def to_device(self, payload, *args):
            return payload

        def generate(self, *args, **kwargs):
            return ["Hello"]

    monkeypatch.setattr(module, "load_model", lambda cfg: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(module, "adapter_class", lambda name: Adapter)
    monkeypatch.setattr(module, "release", lambda: None)
    sample = SimpleNamespace(
        id="s", messages=[SimpleNamespace(content=[SimpleNamespace(type="text", text="Hello")])]
    )
    cfg = read_config(Path(__file__).parents[1] / "examples/train-qwen3-asr-pilot.yaml")
    cfg = cfg.model_copy(update=dict(dataset_root=tmp_path, device="cpu", dtype="float32"))
    paths = module.transcribe(
        tmp_path / "model",
        cfg,
        [sample],
        {"base": [dict(sample_id="s", audio=str(wave))]},
        tmp_path,
    )
    result = json.loads(paths[0].read_text())
    assert sum(observed) == frames
    segments = result["segments"]
    assert segments[0]["start_frame"] == 0 and segments[-1]["end_frame"] == frames
    assert all(a["end_frame"] == b["start_frame"] for a, b in zip(segments, segments[1:]))
    assert result["prediction"] == " ".join(["Hello"] * len(observed))
    assert result["reference"] == "Hello" and result["audio"] == str(wave)
