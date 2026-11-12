"""Shared workflow tests with small native models and explicitly synthetic audio."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sphn")
pytest.importorskip("sentencepiece")
import sentencepiece
import torch
from safetensors.torch import load_file, save_file

from omnifactory._vendor.personaplex.models.lm import LMGen
from omnifactory._vendor.personaplex.models.loaders import MIMI_NAME, TEXT_TOKENIZER_NAME
from omnifactory._vendor.personaplex.utils.compile import no_compile
from omnifactory.adapters.personaplex import PersonaPlexAdapter
from omnifactory.artifacts import sha256
from omnifactory.data.io import iter_samples
from omnifactory.datasets.duplex_demo import create_duplex_demo
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train
from omnifactory.training.evaluation import infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.tiny_personaplex import tiny_mimi, tiny_personaplex


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import omnifactory.adapters.personaplex as adapter_module
    import omnifactory.adapters.personaplex_features as features
    import omnifactory.adapters.personaplex_model as native

    torch.set_num_threads(2)
    torch.manual_seed(31)
    root = tmp_path / "model"
    root.mkdir()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Hello there. You are a helpful assistant. <system> Synthetic duplex fixture.\n" * 20
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "spm"),
        model_type="bpe",
        vocab_size=64,
        pad_id=3,
        minloglevel=2,
        shuffle_input_sentence=False,
        num_threads=1,
    )
    (root / TEXT_TOKENIZER_NAME).write_bytes((tmp_path / "spm.model").read_bytes())
    (root / "config.json").write_text(json.dumps(dict(model_type="personaplex", version="7b-v1")))
    model = tiny_personaplex(card=2048, context=512).eval()
    save_file(model.state_dict(), root / "model.safetensors")
    codec = tiny_mimi()
    save_file({k: v.contiguous() for k, v in codec.state_dict().items()}, root / MIMI_NAME)
    generator = LMGen(model, "cpu", use_sampling=False, check=True)
    codes = torch.randint(0, 32, (1, 17, 6))
    embeds = []
    with no_compile(), torch.no_grad(), generator.streaming(1):
        for t in range(6):
            out = generator.step(
                codes[:, 9:, t : t + 1],
                codes[:, 1:9, t : t + 1],
                codes[:, 0, t],
                return_embeddings=True,
            )
            if out is not None:
                embeds.append(out[1])
        state = dict(embeddings=torch.stack(embeds), cache=generator._streaming_state.cache.clone())
    (root / "voices").mkdir()
    torch.save(state, root / "voices/NATF2.pt")

    def inspect(path):
        model = tiny_personaplex(card=2048, context=512)
        assert set(model.state_dict()) == set(load_file(path / "model.safetensors"))
        return model, len(model.state_dict())

    def mimi(path, device):
        codec = tiny_mimi()
        codec.load_state_dict(load_file(path), strict=True)
        return codec.to(device)

    monkeypatch.setattr(native, "inspect_native_weights", inspect)
    monkeypatch.setattr(adapter_module, "inspect_native_weights", inspect)
    monkeypatch.setattr(features, "get_mimi", mimi)
    create_duplex_demo(tmp_path / "data", root / TEXT_TOKENIZER_NAME)
    cfg = TrainConfig(
        model=str(root),
        adapter="personaplex",
        task="speech_sft",
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "run",
        duplex_text_alignment=tmp_path / "data/text-alignment.json",
        max_steps=2,
        save_steps=1,
        max_length=512,
        max_codec_tokens=128,
        max_attention_pairs=512**2,
        components={
            "temporal_model": dict(
                mode="lora", rank=2, alpha=4, lr=1e-3, target_modules=["out_proj"]
            )
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=512,
            max_codec_frames=256,
            max_audio_frames=256,
            max_vision_patches=1,
            max_decoded_media_bytes=8 * 1024 * 1024,
            max_stream_events=1536,
        ),
    )
    return tmp_path, cfg


def test_native_encoding_timing_and_target_isolation(runtime):
    root, cfg = runtime
    adapter = PersonaPlexAdapter(cfg)
    sample = next(iter_samples(cfg.dataset))
    encoded = adapter.encode(sample, "p0", contract_id="test")
    assert encoded.payload["codes"].shape[1] == 17
    assert encoded.supervision.streams[-1].direction == "input"
    assert encoded.normalization_counts["user_symbols"] == 0
    before = adapter.encode(sample, "p0", contract_id="test", inference=True)
    (root / "data/agent-0.wav").unlink()
    (root / "data/text-alignment.json").unlink()
    other = PersonaPlexAdapter(cfg.model_copy(update={"duplex_text_alignment": None}))
    after = other.encode(sample, "p0", contract_id="test", inference=True)
    assert after.supervision == before.supervision
    for key, value in before.tensors.items():
        torch.testing.assert_close(value, after.tensors[key], atol=0, rtol=0)


def test_long_session_budget_includes_voice_and_persona(runtime, monkeypatch):
    from omnifactory.errors import OmniFactoryError

    _, cfg = runtime
    cfg = TrainConfig.model_validate(
        {
            **cfg.model_dump(),
            "max_audio_seconds": 180,
            "max_codec_tokens": 2250,
            "max_length": 3072,
            "max_attention_pairs": 3072**2,
            "budget": {**cfg.budget.model_dump(), "max_decoded_media_bytes": 64 * 1024**2},
        }
    )
    adapter = PersonaPlexAdapter(cfg)
    sample = next(iter_samples(cfg.dataset))
    # Exercise session accounting without running 2250 frames of a random Mimi.
    wave = torch.zeros(180 * 24000)
    monkeypatch.setattr(adapter.features, "audio", lambda *args: (wave, wave.numel() * 4))
    monkeypatch.setattr(adapter.features, "encode", lambda wave: torch.zeros(1, 8, 2250).long())
    row = adapter.encode(sample, "p0", contract_id="long", inference=True)
    total = 2250 + row.metadata["voice_frames"] + row.metadata["persona_frames"]
    assert row.cost.llm_tokens == total and total > 2250
    for changes in ({"max_length": total - 1}, {"max_attention_pairs": total**2 - 1}):
        adapter.config = cfg.model_copy(update=changes)
        with pytest.raises(OmniFactoryError, match="E_SEQUENCE_LENGTH"):
            adapter.encode(sample, "p0", contract_id="long", inference=True)


@pytest.mark.parametrize("agent_only", [False, True])
def test_shared_train_resume_export_and_streaming_decode(runtime, agent_only):
    root, cfg = runtime
    cfg = cfg.model_copy(update={"personaplex_agent_only_depth": agent_only})
    result = train(cfg)
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    a, b = (load_file(Path(r["checkpoint"]) / "trainable.safetensors") for r in (result, resumed))
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    exported = root / "export"
    export_checkpoint(Path(result["checkpoint"]), exported)
    export_checkpoint(Path(result["checkpoint"]), root / "sglang-export", format="sglang-omni")
    assert sha256(root / "sglang-export/model.safetensors") == sha256(
        exported / "model.safetensors"
    )
    assert sha256(root / "sglang-export/voices/NATF2.pt") == sha256(exported / "voices/NATF2.pt")
    assert sha256(exported / MIMI_NAME) == sha256(root / "model" / MIMI_NAME)
    assert sha256(exported / "voices/NATF2.pt") == sha256(root / "model/voices/NATF2.pt")
    output = root / "predictions.jsonl"
    report = infer_export(
        exported,
        root / "data/validation.jsonl",
        output=output,
        generate_audio=True,
        max_codec_tokens=128,
    )
    assert report["audio_predictions"] == 2
    for line in output.read_text().splitlines():
        row = json.loads(line)
        assert row["sample_rate"] == 24000 and row["stop_reason"] == "input_session_end"
        assert len(row["codec_tokens"]) == row["input_frames"]


def test_timing_hash_and_unsafe_lora_fail_early(runtime):
    _, cfg = runtime
    from omnifactory.errors import OmniFactoryError

    rows = json.loads(cfg.duplex_text_alignment.read_text())
    rows[0]["text_sha256"] = "0" * 64
    cfg.duplex_text_alignment.write_text(json.dumps(rows))
    with pytest.raises(OmniFactoryError, match="E_TEXT_ALIGNMENT"):
        PersonaPlexAdapter(cfg).encode(next(iter_samples(cfg.dataset)), "p0", contract_id="test")
    values = cfg.model_dump()
    values["components"] = {
        "code_predictor": dict(mode="lora", rank=2, alpha=4, lr=1e-3, target_modules=["out_proj"])
    }
    with pytest.raises(ValueError, match="weight-only"):
        TrainConfig.model_validate(values)


def test_sglang_export_requires_selected_unpacked_voice(runtime):
    import tarfile

    from omnifactory.adapters.personaplex import validate_export
    from omnifactory.adapters.personaplex_sglang_assets import unpack_voices
    from omnifactory.errors import OmniFactoryError

    root, cfg = runtime
    model_root = root / "model"
    with tarfile.open(model_root / "voices.tgz", "w:gz") as archive:
        archive.add(model_root / "voices/NATF2.pt", arcname="voices/NATF2.pt")
    (model_root / "voices/NATF2.pt").unlink()
    (model_root / "voices/OTHER.pt").write_bytes(b"unrelated preset")
    metadata = dict(format="sglang-omni", inference_config=cfg.model_dump(mode="json"))
    with pytest.raises(OmniFactoryError, match="unpacked selected voice"):
        validate_export(model_root, metadata)
    unpack_voices(model_root)
    validate_export(model_root, metadata)
    torch.save({"unexpected": torch.zeros(1)}, model_root / "voices/NATF2.pt")
    with pytest.raises(OmniFactoryError, match="native embeddings and cache"):
        validate_export(model_root, metadata)


def test_sglang_probe_excludes_warmup_from_acceptance(runtime, monkeypatch):
    import omnifactory.adapters.personaplex_sglang_serving as serving

    root, cfg = runtime
    exported = root / "export"
    exported.mkdir()
    (exported / "omnifactory_export.json").write_text(
        json.dumps(
            dict(
                inference_config=cfg.model_dump(mode="json"),
            )
        )
    )
    monkeypatch.setattr(serving, "verify_export", lambda _: dict(artifact_id="fixture"))
    calls = []

    async def fake_probe(url, model, voice, persona, wave, output, *, require_overlap=True):
        calls.append((output.name, require_overlap, wave.copy()))
        return dict(output_during_input=require_overlap)

    monkeypatch.setattr(serving, "session_probe", fake_probe)
    result = serving.probe_realtime(
        exported,
        root / "data/validation.jsonl",
        root / "probe",
        "http://fixture",
        "model",
    )
    assert result["requests"] == len(result["results"]) == 2
    assert result["warmup"]["counts_as_acceptance"] is False
    assert result["warmup"]["output_during_input"] is False
    assert [call[:2] for call in calls] == [
        ("warmup", False),
        ("session-00000", True),
        ("session-00001", True),
    ]
    assert (calls[0][2] == calls[1][2]).all()
