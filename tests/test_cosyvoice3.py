"""Shared CosyVoice3 training contracts with synthetic frozen frontend features.

Native pretrained ONNX features and speech quality require separate GPU evidence.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("omegaconf")
pytest.importorskip("x_transformers")
import torch
from safetensors.torch import load_file

from omnifactory.adapters.cosyvoice3 import CosyVoice3Adapter
from omnifactory.adapters.cosyvoice3_features import CosyVoice3Features, pack_sequence
from omnifactory.adapters.cosyvoice3_model import load_cosyvoice3
from omnifactory.artifacts import sha256
from omnifactory.data.io import iter_samples
from omnifactory.datasets.flow_demo import create_flow_demo
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train
from omnifactory.training.evaluation import infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.tiny_cosyvoice3 import create_tiny_cosyvoice3


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    create_tiny_cosyvoice3(tmp_path / "model")
    create_flow_demo(tmp_path / "data")
    for split in ("train", "validation"):
        path = tmp_path / "data" / f"{split}.jsonl"
        rows = []
        for sample in iter_samples(path):
            row = sample.model_dump(mode="json")
            text = "Hello world."
            row["messages"][0]["content"][0]["text"] = text
            row["messages"][1]["content"].append(dict(id="transcript", type="text", text=text))
            row["predictions"][0]["target_block_ids"].append("transcript")
            row["predictions"][0]["conditions"] = []
            row["speech_pairs"] = [dict(text_block_id="transcript", audio_block_id="audio")]
            rows.append(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    # Deliberately synthetic frozen frontend; no claim of native ONNX parity here.
    def features(self, resolver, asset, media, *, flow=True):
        seed = int(resolver.resolve(asset).sha256[:8], 16)
        rng = torch.Generator().manual_seed(seed)
        result = {"speech_token": torch.randint(0, 32, (1, 30), generator=rng)}
        if flow:
            result.update(
                speech_feat=torch.randn(1, 60, 80, generator=rng),
                embedding=torch.randn(1, 192, generator=rng),
            )
        return result, 100000

    monkeypatch.setattr(CosyVoice3Features, "extract", features)
    for name in ("campplus.onnx", "speech_tokenizer_v3.onnx"):
        (tmp_path / "model" / name).write_bytes(b"synthetic frontend fixture; not native ONNX")
    reference = tmp_path / "data/audio-0.wav"
    cfg = TrainConfig(
        model=str(tmp_path / "model"),
        adapter="cosyvoice3",
        task="speech_sft",
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "run",
        speech_reference=reference,
        speech_reference_sha256=sha256(reference),
        max_steps=2,
        save_steps=1,
        max_length=512,
        max_codec_tokens=128,
        cosyvoice_sequence_mode="bistream",
        components={"speech_generator": dict(mode="lora", rank=2, alpha=4, lr=1e-3)},
        budget=dict(
            max_examples=1,
            max_llm_tokens=512,
            max_codec_frames=128,
            max_audio_frames=256,
            max_vision_patches=1,
            max_decoded_media_bytes=1048576,
            max_latent_elements=40960,
        ),
    )
    return tmp_path, cfg


@pytest.mark.parametrize("bistream", [False, True])
def test_packed_embeddings_and_labels_match_native(runtime, monkeypatch, bistream):
    import importlib

    _, cfg = runtime
    model = load_cosyvoice3(cfg).eval().talker
    native = importlib.import_module("omnifactory._vendor.cosyvoice3.llm.llm")
    monkeypatch.setattr(native.random, "random", lambda: 0.0 if bistream else 1.0)
    text = torch.tensor([[2, 3, 4, 5, 6, 7]])
    instruction, speech = torch.tensor([[8, 9]]), torch.arange(30)[None]
    native_labels, native_embeds, _ = model.prepare_lm_input_target(
        model.speech_embedding.weight[model.sos][None, None],
        text,
        model.llm.model.model.embed_tokens(text),
        torch.tensor([6]),
        model.speech_embedding.weight[model.task_id][None, None],
        speech,
        model.speech_embedding(speech),
        torch.tensor([30]),
        instruction,
        model.llm.model.model.embed_tokens(instruction),
        torch.tensor([2]),
    )
    packed = pack_sequence(text, instruction, speech, vocabulary_size=32, bistream=bistream)
    mask, ids = packed["text_mask"], packed["input_ids"]
    actual = torch.where(
        mask[..., None],
        model.llm.model.model.embed_tokens(ids.masked_fill(~mask, 0)),
        model.speech_embedding(ids.masked_fill(mask, 0)),
    )
    assert torch.equal(native_labels, packed["labels"])
    torch.testing.assert_close(actual, native_embeds, atol=0, rtol=0)


@pytest.mark.parametrize("task", ["speech_sft", "flow_sft"])
def test_shared_training_resume_export_and_native_audio(runtime, task, monkeypatch):
    root, cfg = runtime
    component = "speech_generator" if task == "speech_sft" else "audio_decoder"
    cfg = cfg.model_copy(
        update={
            "task": task,
            "components": {
                component: cfg.components["speech_generator"].model_copy(
                    update={"target_modules": ("to_q", "to_v")} if task == "flow_sft" else {}
                )
            },
        }
    )
    adapter = CosyVoice3Adapter(cfg)
    sample = next(iter_samples(cfg.dataset))
    encoded = adapter.encode(sample, "p0", contract_id="test")
    assert encoded.supervision.targets[0].representation.kind == (
        "discrete" if task == "speech_sft" else "continuous"
    )
    run = train(cfg)
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    a, b = (
        load_file(Path(run["checkpoint"]) / "trainable.safetensors"),
        load_file(Path(resumed["checkpoint"]) / "trainable.safetensors"),
    )
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    export_checkpoint(Path(run["checkpoint"]), root / "export", format="vllm-omni")
    export_checkpoint(Path(run["checkpoint"]), root / "sglang-export", format="sglang-omni")
    for name in ("llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx"):
        assert sha256(root / "sglang-export" / name) == sha256(root / "export" / name)
    assert (
        json.loads((root / "sglang-export/omnifactory_export.json").read_text())["format"]
        == "sglang-omni"
    )
    frozen = "flow.pt" if task == "speech_sft" else "llm.pt"
    assert sha256(root / "export" / frozen) == sha256(root / "model" / frozen)
    assert sha256(root / "export/hift.pt") == sha256(root / "model/hift.pt")
    # Random logits frequently select one of 200 native stop symbols immediately.
    # Use a known speech prefix to exercise the real flow/vocoder and EOS plumbing.
    from omnifactory._vendor.cosyvoice3.llm.llm import CosyVoice3LM

    tokens = iter([1, 2, 3, 33])
    monkeypatch.setattr(CosyVoice3LM, "sampling_ids", lambda *args, **kwargs: next(tokens))
    result = infer_export(
        root / "export",
        root / "data/validation.jsonl",
        output=root / "predictions.jsonl",
        generate_audio=True,
        max_codec_tokens=4,
    )
    assert result["audio_predictions"] == 1
    row = json.loads((root / "predictions.jsonl").read_text())
    assert row["sample_rate"] == 24000 and row["codec_tokens"]
    assert row["audio"]


@pytest.mark.parametrize("system_instruction", [None, "Speak warmly."])
def test_native_speech_request_uses_only_reference_and_user_text(runtime, system_instruction):
    from omnifactory.adapters.cosyvoice3 import build_speech_request
    from omnifactory.data.media import LocalMediaResolver
    from omnifactory.data.schema import inference_view
    from omnifactory.errors import OmniFactoryError

    root, cfg = runtime
    sample = next(iter_samples(cfg.dataset))
    view = inference_view(sample, "p0", ("audio",))
    if system_instruction is not None:
        from omnifactory.data.schema import Message, TextBlock

        view = view.model_copy(
            update={
                "messages": (
                    Message(
                        id="system",
                        role="system",
                        content=(
                            TextBlock(id="instruction", type="text", text=system_instruction),
                        ),
                    ),
                    *view.messages,
                )
            }
        )
    instruction = system_instruction or "You are a helpful assistant."
    metadata = {
        "inference_config": {**cfg.model_dump(mode="json"), "speech_reference_text": "Reference."},
        "reference_file": "audio-0.wav",
        "_export_root": str(root / "data"),
    }
    result = build_speech_request(
        {"input": "must be overwritten", "max_new_tokens": 128},
        view=view,
        resolver=LocalMediaResolver(root / "data"),
        metadata=metadata,
    )
    assert result["input"] == "Hello world."
    assert result["ref_text"] == instruction + "<|endofprompt|>Reference."
    assert result["ref_audio"].startswith("data:audio/wav;base64,")
    assert result["max_new_tokens"] == 128
    sglang = build_speech_request(
        {"instructions": "must be replaced by inference-visible conditioning"},
        view=view,
        resolver=LocalMediaResolver(root / "data"),
        metadata={**metadata, "_serving_backend": "sglang-omni"},
    )
    assert sglang["ref_text"] == result["ref_text"]
    assert "instructions" not in sglang
    assert sglang["ref_audio"] == result["ref_audio"]
    metadata["inference_config"]["speech_reference_text"] = ""
    with pytest.raises(OmniFactoryError, match="literal transcript"):
        build_speech_request({}, view=view, resolver=None, metadata=metadata)
    metadata["reference_file"] = "../outside.wav"
    with pytest.raises(OmniFactoryError, match="bundled reference"):
        build_speech_request({}, view=view, resolver=None, metadata=metadata)


def test_flow_objective_and_gradients_match_native(runtime, monkeypatch):
    import importlib

    _, cfg = runtime
    cfg = cfg.model_copy(update={"task": "flow_sft"})
    adapter, model = CosyVoice3Adapter(cfg), load_cosyvoice3(cfg).train()
    batch = dict(
        speech_token=torch.randint(0, 32, (1, 30)),
        speech_token_len=torch.tensor([30]),
        speech_feat=torch.randn(1, 60, 80),
        speech_feat_len=torch.tensor([60]),
        embedding=torch.randn(1, 192),
    )
    native = importlib.import_module("omnifactory._vendor.cosyvoice3.flow.flow")
    # Match random draws while leaving native conditioning/masking/loss untouched.
    monkeypatch.setattr(native.random, "random", lambda: float(torch.rand(())))
    monkeypatch.setattr(native.random, "randint", lambda a, b: int(torch.randint(a, b + 1, ())))
    torch.manual_seed(123)
    expected = model.flow(batch, torch.device("cpu"))["loss"]
    expected.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    result = adapter.forward(model, batch).losses
    actual = result.terms[0].numerator / batch["speech_feat"].numel()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    actual.backward()
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], atol=1e-7, rtol=1e-6)


def test_cached_generation_matches_full_context(runtime):
    _, cfg = runtime
    encoder = load_cosyvoice3(cfg).eval().talker.llm
    torch.manual_seed(17)
    embeddings = torch.randn(1, 9, 32)
    with torch.no_grad():
        full, _ = encoder(embeddings, torch.tensor([9]))
        _, cache = encoder.forward_one_step(
            embeddings[:, :6], torch.ones(1, 6, 6, dtype=torch.bool).tril()
        )
        for i in range(6, 9):
            hidden, cache = encoder.forward_one_step(
                embeddings[:, i : i + 1], torch.ones(1, 1, 1, dtype=torch.bool), cache
            )
            torch.testing.assert_close(hidden[:, -1], full[:, i], atol=1e-6, rtol=1e-5)
