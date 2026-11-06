"""Qwen3 uses its own native MoE, audio reduction and deepstack vision paths."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from omnifactory.adapters.qwen3 import Qwen3Adapter, load_qwen3_thinker
from omnifactory.data.io import iter_samples
from omnifactory.datasets.demo import create_demo
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train
from omnifactory.training.evaluation import infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import apply_policy
from omnifactory.training.tiny_qwen3 import create_tiny_qwen3


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("qwen3")
    model = create_tiny_qwen3(root / "model")
    create_demo(root / "data")
    config = TrainConfig(
        adapter="qwen3_omni",
        model=str(model),
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        max_steps=2,
        save_steps=1,
        max_length=1024,
        max_audio_seconds=1,
        components={"language_model": dict(mode="lora", rank=2, alpha=4, lr=0.001)},
        budget=dict(
            max_examples=2,
            max_llm_tokens=2048,
            max_vision_patches=4096,
            max_audio_frames=1000,
            max_codec_frames=1,
            max_decoded_media_bytes=67108864,
        ),
    )
    return root, config, Qwen3Adapter.from_config(config), list(iter_samples(config.dataset))


def test_native_modalities_and_supervision(runtime):
    _, config, adapter, samples = runtime
    model = load_qwen3_thinker(config).eval()
    rows = [adapter.encode(s, "p0", contract_id="test") for s in samples[:3]]
    with torch.no_grad():
        native = [model(**r.payload, use_cache=False) for r in rows]
        output = adapter.forward(model, adapter.collate([r.payload for r in rows]))
    counts = [r.target_tokens for r in rows]
    expected = sum(x.loss * n for x, n in zip(native, counts)) / sum(counts)
    torch.testing.assert_close(output.loss, expected, atol=3e-5, rtol=3e-5)
    for sample, row in zip(samples, rows):
        inference = adapter.encode(sample, "p0", contract_id="test", inference=True)
        assert not inference.supervision.targets
        assert "labels" not in inference.payload
        assert (row.payload["labels"][0, : row.prompt_length] == -100).all()
        assert torch.equal(
            inference.payload["input_ids"], row.payload["input_ids"][:, : row.prompt_length]
        )


@pytest.mark.parametrize(
    "component,targets,index",
    [
        ("vision_projector", (), 1),
        ("audio_projector", (), 2),
        ("language_experts", ("up_proj", "down_proj"), 0),
    ],
)
def test_component_gradient_paths(runtime, component, targets, index):
    _, config, adapter, samples = runtime
    raw = config.model_dump()
    raw["components"] = {
        component: dict(mode="lora", lr=0.001, rank=2, alpha=4, target_modules=targets)
        if targets
        else dict(mode="full", lr=0.001)
    }
    cfg = TrainConfig.model_validate(raw)
    model, _, plan = apply_policy(load_qwen3_thinker(cfg), cfg)
    row = adapter.encode(samples[index], "p0", contract_id="test")
    adapter.forward(model, row.payload).loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for n, p in model.named_parameters() if ".mlp.gate." in n)
    assert "language_model" in plan.gradient_components


def test_resume_export_reload(runtime):
    root, config, _, _ = runtime
    result = train(config)
    resumed = train(
        config.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    from safetensors.torch import load_file

    a = load_file(str(root / "run/checkpoint-00000002/trainable.safetensors"))
    b = load_file(str(root / "resumed/checkpoint-00000002/trainable.safetensors"))
    assert result["steps"] == resumed["steps"] == 2
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    export_checkpoint(root / "run/checkpoint-00000002", root / "export")
    report = infer_export(
        root / "export",
        root / "data/validation.jsonl",
        output=root / "predictions.jsonl",
        max_new_tokens=2,
    )
    assert report["predictions"] > 0
    assert (
        json.loads((root / "export/config.json").read_text())["model_type"]
        == "qwen3_omni_moe_thinker"
    )


def test_video_with_audio_native_forward(runtime, tmp_path):
    from omnifactory.data.schema import CanonicalSample
    from omnifactory.datasets.omni_demo import create_omni_demo

    _, config, _, _ = runtime
    create_omni_demo(tmp_path / "data")
    cfg = config.model_copy(update={"dataset": tmp_path / "data/train.jsonl"})
    adapter = Qwen3Adapter.from_config(cfg)
    sample = list(iter_samples(cfg.dataset))[-1]
    raw = sample.model_dump(mode="json")
    target = raw["messages"][-1]
    target["content"] = [b for b in target["content"] if b["type"] == "text"]
    raw["predictions"][0]["target_block_ids"] = [b["id"] for b in target["content"]]
    raw["assets"].pop("target-wave")
    raw["speech_pairs"] = []
    row = adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="video")
    assert "pixel_values_videos" in row.payload and "input_features" in row.payload
    model = load_qwen3_thinker(cfg).eval()
    with torch.no_grad():
        expected = model(**row.payload, use_cache=False).loss
        actual = adapter.forward(model, adapter.collate([row.payload])).loss
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_native_export_preserves_qwen3_stages(runtime, tmp_path, monkeypatch, storage_dtype):
    from safetensors.torch import load_file, save_file

    from omnifactory.training.export import assemble_native

    _, config, adapter, samples = runtime
    model = load_qwen3_thinker(config).eval()
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    out.mkdir()
    # Sentinel stages test structural preservation, not speech inference.
    weights = {"thinker." + k: v.to(storage_dtype) for k, v in model.state_dict().items()}
    weights.update({"talker.sentinel": torch.arange(3.0), "code2wav.sentinel": torch.arange(4.0)})
    save_file(weights, base / "model.safetensors")
    (base / "config.json").write_text(
        json.dumps({"model_type": "qwen3_omni_moe", "thinker_config": model.config.to_dict()})
    )
    with torch.no_grad():
        model.lm_head.weight.add_(0.02)
    assemble_native(base, model, out, adapter="qwen3_omni", shard_bytes=100000)
    mapping = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    for key in ("talker.sentinel", "code2wav.sentinel"):
        torch.testing.assert_close(load_file(out / mapping[key])[key], weights[key])
    loaded = load_qwen3_thinker(config.model_copy(update={"model": str(out)})).eval()
    if storage_dtype == torch.bfloat16:
        from pathlib import Path

        monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
        from evaluate_asr_merge import rounded_native_state

        frequencies = model.model.rotary_emb.inv_freq.clone()
        model.load_state_dict(rounded_native_state(model), strict=True)
        # The in-memory precision reference must match a real native export
        # while retaining transient rotary buffers at their original precision.
        torch.testing.assert_close(model.model.rotary_emb.inv_freq, frequencies, rtol=0, atol=0)
    row = adapter.encode(samples[0], "p0", contract_id="export")
    with torch.no_grad():
        torch.testing.assert_close(model(**row.payload).logits, loaded(**row.payload).logits)


@pytest.mark.parametrize(
    "update",
    [
        {"task": "speech_sft"},
        {"quantization": "nf4"},
        {"components": {"language_router": {"mode": "full", "lr": 0.001}}},
    ],
)
def test_unsupported_qwen3_capabilities_fail_early(runtime, update):
    _, config, _, _ = runtime
    with pytest.raises(ValueError):
        TrainConfig.model_validate({**config.model_dump(), **update})
