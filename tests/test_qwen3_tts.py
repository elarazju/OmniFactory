"""Native Qwen3-TTS training alignment, codec assets and standalone export."""

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
import torch
from safetensors.torch import load_file

from omnifactory.adapters.qwen3_tts import Qwen3TTSAdapter, load_qwen3_tts
from omnifactory.artifacts import sha256
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.flow_demo import create_flow_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.tiny_qwen3_tts import create_tiny_qwen3_tts


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("qwen3_tts")
    create_tiny_qwen3_tts(root / "model")
    create_flow_demo(root / "data")
    for split in ("train", "validation"):
        rows = []
        for sample in iter_samples(root / "data" / f"{split}.jsonl"):
            row = sample.model_dump(mode="json")
            text = "Hello world."
            row["messages"][0]["content"][0]["text"] = text
            row["messages"][1]["content"].append({"id": "transcript", "type": "text", "text": text})
            row["predictions"][0]["target_block_ids"].append("transcript")
            row["speech_pairs"] = [{"text_block_id": "transcript", "audio_block_id": "audio"}]
            rows.append(CanonicalSample.model_validate(row).model_dump(mode="json"))
        (root / "data" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    reference = root / "data/audio-0.wav"
    cfg = TrainConfig(
        adapter="qwen3_tts",
        task="speech_sft",
        model=str(root / "model"),
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        speech_reference=reference,
        speech_reference_sha256=sha256(reference),
        speech_speaker="testvoice",
        max_length=256,
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=2,
        components={
            k: dict(mode="lora", rank=2, alpha=4, lr=1e-3)
            for k in ("speech_generator", "code_predictor")
        },
        budget=dict(
            max_examples=2,
            max_llm_tokens=512,
            max_audio_frames=512,
            max_vision_patches=1,
            max_codec_frames=512,
            max_decoded_media_bytes=1048576,
        ),
    )
    adapter = Qwen3TTSAdapter(cfg)
    samples = list(iter_samples(cfg.dataset))
    rows = [adapter.encode(s, "p0", contract_id="test") for s in samples]
    return root, cfg, adapter, samples, rows


def test_native_codec_objectives_and_all_codebook_gradients(runtime):
    _, cfg, adapter, _, rows = runtime
    model = load_qwen3_tts(cfg).eval()
    row = rows[0]
    batch = row.payload
    n = batch["residual_labels"].shape[0]
    assert row.normalization_counts == {"primary_symbols": n + 1, "residual_symbols": n * 15}
    result = adapter.forward(model, batch)
    expected = torch.nn.functional.cross_entropy(
        result.logits.flatten(0, 1), batch["primary_labels"][:, 1:].flatten(), reduction="mean"
    ) + 0.3 * torch.nn.functional.cross_entropy(
        result.residual_logits.flatten(0, 1), batch["residual_labels"].flatten(), reduction="mean"
    )
    torch.testing.assert_close(result.losses.loss, expected)
    result.losses.loss.backward()
    for head in model.talker.code_predictor.lm_head:
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert model.talker.text_projection.linear_fc1.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.speaker_encoder.parameters())


def test_future_frames_and_current_residual_cannot_leak(runtime):
    _, cfg, adapter, _, rows = runtime
    model = load_qwen3_tts(cfg).eval()
    batch = rows[0].payload
    changed = copy.deepcopy(batch)
    positions = batch["frame_mask"][0].nonzero().flatten()
    position = positions[1].item()
    changed["codec_ids"][:, position:] = (changed["codec_ids"][:, position:] + 1) % 2048
    changed["input_ids"][:, position:, 1] = changed["codec_ids"][:, position:, 0]
    with torch.no_grad():
        first = adapter.forward(model, batch)
        second = adapter.forward(model, changed)
    torch.testing.assert_close(
        first.logits[:, :position], second.logits[:, :position], atol=0, rtol=0
    )
    torch.testing.assert_close(first.residual_logits[0], second.residual_logits[0], atol=0, rtol=0)
    # Codebook k may condition on earlier codebooks, never on its own/future labels.
    changed = copy.deepcopy(batch)
    changed["codec_ids"][0, positions[0], 5:] = (
        changed["codec_ids"][0, positions[0], 5:] + 1
    ) % 2048
    with torch.no_grad():
        third = adapter.forward(model, changed)
    torch.testing.assert_close(
        first.residual_logits[0, :5], third.residual_logits[0, :5], atol=0, rtol=0
    )


def test_target_free_input_and_reference_integrity(runtime):
    _, cfg, adapter, samples, _ = runtime
    prompt = adapter.encode(samples[0], "p0", contract_id="infer", inference=True).validate(
        inference=True
    )
    changed = samples[0].model_dump(mode="json")
    changed["messages"][1]["content"][1]["text"] = "secret answer"
    prompt2 = adapter.encode(
        CanonicalSample.model_validate(changed), "p0", contract_id="infer", inference=True
    )
    assert not prompt.supervision.targets
    for key in prompt.tensors:
        torch.testing.assert_close(prompt.tensors[key], prompt2.tensors[key], atol=0, rtol=0)
    with pytest.raises(OmniFactoryError, match="E_TTS_TARGET"):
        adapter.encode(CanonicalSample.model_validate(changed), "p0", contract_id="test")
    wrong = Qwen3TTSAdapter(cfg.model_copy(update={"speech_reference_sha256": "0" * 64}))
    with pytest.raises(OmniFactoryError, match="E_SPEAKER_REFERENCE"):
        wrong.speaker_embedding()


def test_train_resume_export_and_native_generation(runtime, monkeypatch):
    root, cfg, adapter, _, rows = runtime
    run = train(cfg)
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    weights = load_file(Path(run["checkpoint"]) / "trainable.safetensors")
    other = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    assert weights.keys() == other.keys()
    for key in weights:
        torch.testing.assert_close(weights[key], other[key], atol=0, rtol=0)
    trained, _, _, _ = load_checkpoint(Path(run["checkpoint"]))
    before = adapter.forward(trained, rows[0].payload)
    export_checkpoint(Path(run["checkpoint"]), root / "export", format="vllm-omni")
    sgl = export_checkpoint(Path(run["checkpoint"]), root / "sgl-export", format="sglang-omni")
    assert sgl["format"] == "sglang-omni" and sgl["sglang_deployment_verified"] is False
    from omnifactory.artifacts import verify_export

    vllm_files = verify_export(root / "export")["files"]
    sgl_files = verify_export(root / "sgl-export")["files"]
    assert {k: v for k, v in vllm_files.items() if k != "omnifactory_export.json"} == {
        k: v for k, v in sgl_files.items() if k != "omnifactory_export.json"
    }
    loaded = load_qwen3_tts(cfg.model_copy(update={"model": str(root / "export")})).eval()
    after = adapter.forward(loaded, rows[0].payload)
    torch.testing.assert_close(before.logits, after.logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(before.residual_logits, after.residual_logits, atol=2e-6, rtol=2e-5)
    from omnifactory._vendor.qwen3_tts.core.models import Qwen3TTSForConditionalGeneration

    native = Qwen3TTSForConditionalGeneration.from_pretrained(
        root / "export", local_files_only=True
    )
    assert native.speech_tokenizer is not None
    torch.testing.assert_close(
        native.talker.get_input_embeddings().weight, loaded.talker.get_input_embeddings().weight
    )
    assert loaded.speaker_encoder is None
    assert not any(k.startswith("speaker_encoder.") for k in loaded.state_dict())
    assert loaded.config.talker_config.spk_id == {"testvoice": 3000}
    for path in (root / "model/speech_tokenizer").rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes()
                == (
                    root
                    / "export/speech_tokenizer"
                    / path.relative_to(root / "model/speech_tokenizer")
                ).read_bytes()
            )
    result = infer_export(
        root / "export",
        root / "data/validation.jsonl",
        output=root / "generated.jsonl",
        generate_audio=True,
        max_codec_tokens=4,
    )
    assert result["audio_predictions"] == 1
    record = json.loads((root / "generated.jsonl").read_text())
    assert record["sample_rate"] == 24000
    assert record["codec_tokens"] and len(record["codec_tokens"][0]) == 16

    import omnifactory.serving as serving

    def response(url, body):
        assert url.endswith("/v1/audio/speech")
        assert body["voice"] == "testvoice" and body["task_type"] == "CustomVoice"
        assert "ref_audio" not in body
        # This probes request construction; the random codec is not a speech-quality fixture.
        return (
            200,
            cfg.speech_reference.read_bytes(),
            {"Content-Type": "audio/wav", "X-Finish-Reason": "stop"},
        )

    monkeypatch.setattr(serving, "http_request", response)
    report = serving.probe(
        root / "export",
        root / "data/validation.jsonl",
        root / "mock-http",
        "http://localhost",
        "fixture",
    )
    assert report["requests"] == 1 and report["modality"] == "audio"
    report = serving.probe(
        root / "sgl-export",
        root / "data/validation.jsonl",
        root / "mock-sgl-http",
        "http://localhost",
        "fixture",
        backend="sglang-omni",
    )
    assert report["backend"] == "sglang-omni"
    assert report["results"][0]["finish_reason"] == "stop"
    for reason in ("length", "unknown", None):
        monkeypatch.setattr(
            serving,
            "http_request",
            lambda *args: (
                200,
                cfg.speech_reference.read_bytes(),
                {} if reason is None else {"x-finish-reason": reason},
            ),
        )
        with pytest.raises(OmniFactoryError, match="E_SERVE_INCOMPLETE"):
            serving.probe(
                root / "sgl-export",
                root / "data/validation.jsonl",
                root / f"mock-sgl-{reason}",
                "http://localhost",
                "fixture",
                backend="sglang-omni",
            )


def test_training_prefix_matches_native_generation(runtime, monkeypatch):
    _, cfg, adapter, samples, rows = runtime
    model = load_qwen3_tts(cfg).eval()
    model.config.talker_config.spk_id = {"testvoice": 3000}
    model.config.talker_config.spk_is_dialect = {"testvoice": False}
    with torch.no_grad():
        model.talker.get_input_embeddings().weight[3000].copy_(adapter.speaker_embedding()[0])
    captured = {}

    class Captured(Exception):
        pass

    def capture_native(**kwargs):
        captured["native"] = kwargs["inputs_embeds"].detach().clone()
        raise Captured()

    monkeypatch.setattr(model.talker, "generate", capture_native)
    prompt = adapter.encode(samples[0], "p0", contract_id="infer", inference=True).payload
    with pytest.raises(Captured):
        model.generate(
            input_ids=[prompt["text_ids"]],
            languages=["Auto"],
            speakers=["testvoice"],
            non_streaming_mode=True,
        )

    def capture_training(*args, **kwargs):
        captured["training"] = kwargs["inputs_embeds"].detach().clone()
        raise Captured()

    monkeypatch.setattr(model.talker, "forward", capture_training)
    with pytest.raises(Captured):
        adapter.forward(model, rows[0].payload)
    n = rows[0].payload["frame_mask"][0].nonzero()[0].item()
    assert captured["native"].shape[1] == n
    torch.testing.assert_close(
        captured["native"], captured["training"][:, :n], atol=1e-7, rtol=1e-6
    )


def test_customvoice_serving_keeps_embedded_speaker(runtime):
    from omnifactory.adapters.qwen3_tts import build_speech_request
    from omnifactory.data.schema import inference_view

    _, cfg, adapter, samples, _ = runtime
    view = inference_view(samples[0], "p0", ("audio",))
    metadata = {"speaker": "testvoice", "speaker_reference_sha256": cfg.speech_reference_sha256}
    payload = build_speech_request(
        {"input": "Hello world."}, view=view, resolver=adapter.resolver, metadata=metadata
    )
    assert payload["task_type"] == "CustomVoice"
    assert payload["voice"] == "testvoice"
    assert "ref_audio" not in payload
    with pytest.raises(OmniFactoryError, match="E_TTS_CONDITION"):
        build_speech_request(
            payload,
            view=view,
            resolver=adapter.resolver,
            metadata={**metadata, "speaker_reference_sha256": "0" * 64},
        )


@pytest.mark.parametrize(
    "component", ["text_projector", "audio_embedding", "speech_generator", "code_predictor"]
)
def test_component_policy_preserves_gradient_paths(runtime, component):
    from omnifactory.planning.adaptation import ComponentPolicy
    from omnifactory.training.policy import apply_policy

    _, cfg, adapter, _, rows = runtime
    local = cfg.model_copy(
        update={"components": {component: ComponentPolicy(mode="full", lr=1e-3)}}
    )
    model, _, _ = apply_policy(load_qwen3_tts(local), local)
    model.train()
    adapter.forward(model, rows[0].payload).losses.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
    )
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(not p.requires_grad for p in model.speaker_encoder.parameters())


def test_acceptance_records_missing_eos_as_failure(tmp_path):
    import importlib.util

    import numpy as np
    import soundfile as sf

    from omnifactory.acceptance import AcceptanceReport, Gate

    spec = importlib.util.spec_from_file_location(
        "qwen3_tts_acceptance", Path(__file__).parents[1] / "scripts/validate_gpu_qwen3_tts.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    root = tmp_path
    (root / "acceptance").mkdir()
    path = root / "acceptance/acceptance.json"
    report = AcceptanceReport(
        adapter="qwen3_tts", task="speech_sft", model="fixture", revision=None
    )
    report.gates["inference"] = Gate(status="passed", details={"predictions": 4})
    report.save(path)
    audio = root / "wave.wav"
    sf.write(audio, np.sin(np.arange(2400) * 0.1) * 0.1, 24000)
    records = [dict(sample_id=str(i), speech_finished=i != 2, audio=str(audio)) for i in range(4)]
    predictions = root / "acceptance/predictions.jsonl"
    predictions.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = script.validate_waveforms(root)
    assert result["status"] == "failed" and "before EOS" in result["error"]
    report = AcceptanceReport.model_validate_json(path.read_text())
    assert report.gates["inference"].status == "failed"
    assert report.gates["native_reference"].status == "not_run"
    records[2]["speech_finished"] = True
    predictions.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = script.validate_waveforms(root)
    assert result["status"] == "passed" and result["records"] == 4
