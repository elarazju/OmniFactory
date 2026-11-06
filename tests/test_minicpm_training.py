"""Second-family acceptance using real pinned OpenBMB code at tiny dimensions."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
from safetensors.torch import load_file, save_file

from omnifactory.adapters.base import unpack_tree
from omnifactory.adapters.minicpm import MiniCPMAdapter, load_minicpm
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.demo import create_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, load_tensors, train
from omnifactory.training.evaluation import evaluate, infer_export
from omnifactory.training.export import assemble_native, export_checkpoint
from omnifactory.training.policy import apply_policy
from omnifactory.training.prepare import prepare
from omnifactory.training.tiny_minicpm import create_tiny_minicpm


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("minicpm")
    create_tiny_minicpm(root / "model")
    create_demo(root / "data")
    config = TrainConfig(
        model=str(root / "model"),
        adapter="minicpm_o_4_5",
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        max_steps=4,
        save_steps=2,
        gradient_accumulation_steps=2,
        max_length=512,
        components={"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.01}},
        budget={
            "max_examples": 2,
            "max_llm_tokens": 1024,
            "max_vision_patches": 128,
            "max_audio_frames": 400,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 1000000,
        },
    )
    return root, config, MiniCPMAdapter.from_config(config), list(iter_samples(config.dataset))


@pytest.fixture(scope="module")
def trained(runtime):
    return Path(train(runtime[1])["checkpoint"])


def test_native_singleton_parity_and_backward(runtime):
    _, config, adapter, samples = runtime
    model = load_minicpm(config).eval()
    for sample in samples[:3]:
        encoded = adapter.encode(sample, "p0", contract_id="test")
        batch = encoded.payload
        assert (batch["labels"][:, : encoded.prompt_length] == -100).all()
        data = {**batch, "position_ids": batch["attention_mask"].long().cumsum(-1) - 1}
        with torch.no_grad():
            native = model(
                data=data, labels=batch["labels"], attention_mask=batch["attention_mask"]
            )
        result = adapter.forward(model, batch)
        torch.testing.assert_close(result.logits, native.logits, atol=1e-6, rtol=1e-6)
        result.loss.backward()
        assert model.llm.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
        model.zero_grad()


def multi_audio(sample):
    raw = sample.model_dump(mode="json")
    audio = next(b for b in raw["messages"][0]["content"] if b["type"] == "audio")
    second = {**audio, "id": "second-audio", "media": {**audio["media"], "end_us": 150000}}
    raw["messages"][0]["content"].append(second)
    return CanonicalSample.model_validate(raw)


def test_mixed_batch_matches_singletons_with_two_audio_clips(runtime):
    _, config, adapter, samples = runtime
    model = load_minicpm(config).eval()
    rows = [
        adapter.encode(s, "p0", contract_id="test")
        for s in [samples[0], samples[1], samples[2], multi_audio(samples[2])]
    ]
    batch = adapter.collate([r.payload for r in rows])
    counts = [int((r.tensors["labels"][:, 1:] != -100).sum()) for r in rows]
    with torch.no_grad():
        singles = [adapter.forward(model, r.payload) for r in rows]
        combined = adapter.forward(model, batch)
    expected = sum(x.loss * n for x, n in zip(singles, counts)) / sum(counts)
    torch.testing.assert_close(combined.loss, expected, atol=2e-5, rtol=2e-5)
    for i, (row, single) in enumerate(zip(rows, singles)):
        torch.testing.assert_close(
            combined.logits[i, -row.cost.llm_tokens :], single.logits[0], atol=2e-5, rtol=2e-5
        )
    assert len(batch["audio_feature_lens"][-1]) == 2


def test_projector_gradient_through_frozen_llm(runtime):
    _, config, adapter, samples = runtime
    for component, index, prefix in [
        ("audio_projector", 2, "audio_projection_layer"),
        ("vision_projector", 1, "resampler"),
    ]:
        c = TrainConfig.model_validate(
            {**config.model_dump(), "components": {component: {"mode": "full", "lr": 0.001}}}
        )
        model, groups, plan = apply_policy(load_minicpm(c), c)
        batch = adapter.encode(samples[index], "p0", contract_id="test").payload
        adapter.forward(model, batch).loss.backward()
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
        assert model.llm.model.layers[0].self_attn.q_proj.weight.grad is None
        assert "language_model" in plan.gradient_components
        assert groups[0]["component"] == component


def test_prepared_nested_tensors_and_saved_processor(runtime, tmp_path):
    _, config, adapter, samples = runtime
    manifest = prepare(config, tmp_path / "prepared")
    reload = MiniCPMAdapter.from_config(
        config.model_copy(update={"model": str(tmp_path / "prepared/processor")})
    )
    for entry, sample in zip(manifest["entries"], samples):
        batch = load_tensors(tmp_path / "prepared", entry)
        expected = adapter.encode(sample, "p0", contract_id="test")
        actual = reload.encode(sample, "p0", contract_id="test")
        assert actual.layout == expected.layout
        for key, value in expected.tensors.items():
            torch.testing.assert_close(actual.tensors[key], value, rtol=0, atol=0)
        torch.testing.assert_close(batch["input_ids"], expected.payload["input_ids"])
    assert manifest["entries"][1]["layout"]["dict"]["pixel_values"]["list"]
    entry = manifest["entries"][0]
    path = tmp_path / "prepared" / entry["file"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(OmniFactoryError, match="E_PREPARED_HASH"):
        load_tensors(tmp_path / "prepared", entry)


def test_no_target_leakage_and_reject_short_audio(runtime):
    _, _, adapter, samples = runtime
    original = samples[2]
    raw = original.model_dump(mode="json")
    raw["messages"][-1]["content"][0]["text"] = "completely different answer"
    changed = CanonicalSample.model_validate(raw)
    a = adapter.encode(original, "p0", contract_id="test", inference=True)
    b = adapter.encode(changed, "p0", contract_id="test", inference=True)
    assert "labels" not in a.tensors
    for key in a.tensors:
        torch.testing.assert_close(a.tensors[key], b.tensors[key])
    audio = next(b for b in raw["messages"][0]["content"] if b["type"] == "audio")
    audio["media"]["end_us"] = 40000
    with pytest.raises(OmniFactoryError, match="E_AUDIO"):
        adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="test")


def test_resume_export_evaluate_infer(runtime, trained, tmp_path):
    root, config, adapter, samples = runtime
    resumed = train(
        config.model_copy(
            update={
                "output_dir": tmp_path / "resume",
                "resume_from": root / "run/checkpoint-00000002",
            }
        )
    )
    a, b = (
        load_file(trained / "trainable.safetensors"),
        load_file(Path(resumed["checkpoint"]) / "trainable.safetensors"),
    )
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    metrics = evaluate(trained, root / "data/validation.jsonl", generate=True, max_new_tokens=3)
    assert metrics["predictions"] == 6 and metrics["loss"] > 0
    result = export_checkpoint(trained, tmp_path / "export")
    assert result["adapter"] == "minicpm_o_4_5" and result["format"] == "understanding"
    assert result["vllm_deployment_verified"] is False
    before, adapter, _, _ = load_checkpoint(trained)
    after = load_minicpm(config.model_copy(update={"model": str(tmp_path / "export")})).eval()
    with torch.no_grad():
        for sample in samples[:3]:
            batch = adapter.encode(sample, "p0", contract_id="test").payload
            torch.testing.assert_close(
                adapter.forward(before, batch).logits,
                adapter.forward(after, batch).logits,
                atol=2e-5,
                rtol=2e-5,
            )
    assert not any("lora" in n for n, _ in after.named_parameters())
    info = infer_export(
        tmp_path / "export",
        root / "data/validation.jsonl",
        output=tmp_path / "predictions.jsonl",
        max_new_tokens=3,
    )
    assert info["predictions"] == 6
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == 6
    assert (tmp_path / "export/provenance.json").is_file()


def test_native_assembly_preserves_speech_and_decoder_assets(runtime, tmp_path):
    _, config, _, _ = runtime
    model = load_minicpm(config)
    base = tmp_path / "base"
    base.mkdir()
    cfg = model.config.to_dict()
    cfg["init_tts"] = True
    (base / "config.json").write_text(json.dumps(cfg))
    weights = {k: v.contiguous() for k, v in model.state_dict().items()}
    weights["tts.test.weight"] = torch.arange(7.0)
    save_file(weights, base / "model.safetensors")
    (base / "decoder").mkdir()
    (base / "decoder/config.json").write_text('{"preserve":true}')
    with torch.no_grad():
        model.audio_projection_layer.linear1.weight.add_(0.5)
    out = tmp_path / "native"
    out.mkdir()
    assemble_native(base, model, out, adapter="minicpm_o_4_5", shard_bytes=100000)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    for key in ("tts.test.weight", "audio_projection_layer.linear1.weight"):
        actual = load_file(out / index[key])[key]
        expected = weights[key] if key.startswith("tts") else model.state_dict()[key]
        torch.testing.assert_close(actual, expected)
    assert (out / "decoder/config.json").read_text() == '{"preserve":true}'


def test_nf4_rejected_explicitly(runtime):
    with pytest.raises(OmniFactoryError, match="E_QUANTIZATION"):
        load_minicpm(runtime[1].model_copy(update={"quantization": "nf4"}))


def test_tensor_tree_keeps_empty_lists_and_scalars():
    from omnifactory.adapters.base import pack_tree

    payload = {"labels": torch.ones(1, 2), "nested": [[], [torch.arange(3)], 7, None]}
    tensors, layout = pack_tree(payload)
    restored = unpack_tree(tensors, layout)
    assert restored["nested"][0] == [] and restored["nested"][2:] == [7, None]
    torch.testing.assert_close(restored["nested"][1][0], torch.arange(3))


def test_component_lora_ranks_and_learning_rates(runtime):
    _, config, adapter, samples = runtime
    config = TrainConfig.model_validate(
        {
            **config.model_dump(),
            "components": {
                "language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.001},
                "audio_projector": {
                    "mode": "lora",
                    "rank": 2,
                    "alpha": 4,
                    "lr": 0.0002,
                    "target_modules": ["linear1", "linear2"],
                },
            },
        }
    )
    model, groups, plan = apply_policy(load_minicpm(config), config)
    assert {g["component"]: g["lr"] for g in groups} == {
        "language_model": 0.001,
        "audio_projector": 0.0002,
    }
    assert model.audio_projection_layer.linear1.lora_A["default"].weight.shape[0] == 2
    assert model.llm.model.layers[0].self_attn.q_proj.lora_A["default"].weight.shape[0] == 4
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    adapter.forward(
        model, adapter.encode(samples[2], "p0", contract_id="test").payload
    ).loss.backward()
    assert model.audio_projection_layer.linear1.lora_B["default"].weight.grad.abs().sum() > 0
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == plan.trainable_numel


def test_combined_image_audio_and_image_slicing(runtime, tmp_path):
    import hashlib

    from PIL import Image

    _, config, adapter, samples = runtime
    raw = samples[2].model_dump(mode="json")
    # Same ordered conversation now carries both an image and audio.
    image = next(
        b
        for b in samples[1].model_dump(mode="json")["messages"][0]["content"]
        if b["type"] == "image"
    )
    raw["assets"]["combined-photo"] = samples[1].model_dump(mode="json")["assets"][
        image["asset_id"]
    ]
    image["asset_id"] = "combined-photo"
    image["id"] = "combined-image"
    raw["messages"][0]["content"].insert(0, image)
    path = tmp_path / "large.png"
    Image.new("RGB", (112, 112), "red").save(path)
    raw["assets"][image["asset_id"]]["uri"] = str(path)
    raw["assets"][image["asset_id"]]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = MiniCPMAdapter.from_config(config)
    adapter.processor.image_processor.max_slice_nums = 4
    from omnifactory.data.processor import OmniDataProcessor

    sample = CanonicalSample.model_validate(raw)
    plan = OmniDataProcessor().plan(sample, model="minicpm_o_4_5")
    assert plan.predictions[0].input_modalities == ("audio", "image", "text")
    encoded = adapter.encode(sample, "p0", contract_id="test")
    payload = encoded.payload
    assert len(payload["pixel_values"][0]) > 1
    assert len(payload["image_bound"][0]) == len(payload["pixel_values"][0])
    assert encoded.cost.modalities == ("audio", "image", "text")
    model = load_minicpm(config).eval()
    expected = adapter.forward(model, payload)
    other = adapter.encode(samples[0], "p0", contract_id="test")
    batch = adapter.collate([payload, other.payload])
    actual = adapter.forward(model, batch)
    torch.testing.assert_close(
        actual.logits[0, -encoded.cost.llm_tokens :], expected.logits[0], atol=2e-5, rtol=2e-5
    )


def test_vendor_sources_match_pinned_provenance():
    import hashlib

    import omnifactory._vendor.minicpm_o45 as vendor

    root = Path(vendor.__file__).parent
    info = json.loads((root / "provenance.json").read_text())
    assert len(info["revision"]) == 40
    for name, digest in info["sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
