"""Real tiny Qwen architecture tests; skipped when the optional runtime is absent."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from omnifactory.adapters.qwen2_5 import Qwen25Adapter, load_thinker
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.demo import create_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import ModalBatchPlanner, SampleCost
from omnifactory.training.artifacts import new_directory, write_json
from omnifactory.training.config import TrainConfig, read_config
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import evaluate, infer_export
from omnifactory.training.export import assemble_native, export_checkpoint
from omnifactory.training.policy import apply_policy
from omnifactory.training.tiny import create_tiny


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("runtime")
    create_tiny(root / "model")
    create_demo(root / "data")
    config = TrainConfig(
        model=str(root / "model"),
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
    adapter = Qwen25Adapter.from_config(config)
    samples = list(iter_samples(config.dataset))
    return root, config, adapter, samples


@pytest.fixture(scope="module")
def trained(runtime):
    root, config, _, _ = runtime
    result = train(config)
    return Path(result["checkpoint"])


def test_real_modalities_forward_backward_and_label_boundary(runtime):
    _, config, adapter, samples = runtime
    model = load_thinker(config)
    for sample in samples[:3]:
        encoded = adapter.encode(sample, "p0", contract_id="test")
        labels = encoded.tensors["labels"]
        assert (labels[:, : encoded.prompt_length] == -100).all()
        assert torch.equal(
            labels[:, encoded.prompt_length :],
            encoded.tensors["input_ids"][:, encoded.prompt_length :],
        )
        output = model(**encoded.payload)
        output.loss.backward()
        assert torch.isfinite(output.loss)
        assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
        model.zero_grad()


def test_prompt_encoding_independent_of_targets(runtime):
    _, _, adapter, samples = runtime
    original = samples[2]
    raw = original.model_dump(mode="json")
    raw["messages"][-1]["content"][0]["text"] = "entirely different answer"
    changed = CanonicalSample.model_validate(raw)
    a = adapter.encode(original, "p0", contract_id="test", inference=True)
    b = adapter.encode(changed, "p0", contract_id="test", inference=True)
    for key in a.tensors:
        torch.testing.assert_close(a.tensors[key], b.tensors[key])
    assert "labels" not in a.tensors


def test_padded_batch_loss_matches_independent_losses(runtime):
    _, config, adapter, samples = runtime
    model = load_thinker(config).eval()
    # Exercise different lengths, mixed modalities, left padding and multimodal offsets.
    encoded = [adapter.encode(s, "p0", contract_id="test") for s in samples[:3]]
    counts = [int((e.tensors["labels"][:, 1:] != -100).sum()) for e in encoded]
    with torch.no_grad():
        expected = sum(model(**e.payload).loss * n for e, n in zip(encoded, counts)) / sum(counts)
        actual = model(**adapter.collate([e.payload for e in encoded])).loss
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_projector_gradient_through_frozen_language_model(runtime):
    _, config, adapter, samples = runtime
    config = config.model_copy(
        update={
            "components": {
                "vision_projector": config.components["language_model"].model_validate(
                    {"mode": "full", "lr": 0.001}
                )
            }
        }
    )
    model, _, plan = apply_policy(load_thinker(config), config)
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    encoded = adapter.encode(samples[1], "p0", contract_id="test")
    model(**encoded.payload).loss.backward()
    gradients = [p.grad for name, p in model.named_parameters() if name.startswith("visual.merger")]
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert model.model.layers[0].self_attn.q_proj.weight.grad is None
    assert "language_model" in plan.gradient_components
    torch.testing.assert_close(before, model.model.layers[0].self_attn.q_proj.weight)


def test_lora_updates_only_selected_weights(runtime):
    _, config, adapter, samples = runtime
    model, groups, plan = apply_policy(load_thinker(config), config)
    snapshots = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(groups)
    model(**adapter.encode(samples[0], "p0", contract_id="test").payload).loss.backward()
    optimizer.step()
    changed = {name for name, p in model.named_parameters() if not torch.equal(p, snapshots[name])}
    assert changed and all(".lora_" in name for name in changed)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == plan.trainable_numel


def test_exact_resume_matches_uninterrupted(runtime, trained):
    root, config, _, _ = runtime
    resumed = config.model_copy(
        update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000002"}
    )
    result = train(resumed)
    expected = load_file(trained / "trainable.safetensors")
    actual = load_file(Path(result["checkpoint"]) / "trainable.safetensors")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_resume_rejects_policy_change(runtime, trained):
    root, config, _, _ = runtime
    bad = config.model_copy(
        update={
            "output_dir": root / "bad-resume",
            "resume_from": trained,
            "max_steps": 6,
            "seed": 99,
        }
    )
    with pytest.raises(OmniFactoryError, match="E_RESUME_CONTRACT"):
        train(bad)


def test_export_merge_reload_logits_and_inference(runtime, trained):
    root, config, adapter, samples = runtime
    model, _, _, _ = load_checkpoint(trained)
    exported = root / "export"
    export_checkpoint(trained, exported)
    reloaded = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        exported, attn_implementation="sdpa"
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(exported, use_fast=False)
    assert (
        processor.feature_extractor.feature_size == adapter.processor.feature_extractor.feature_size
    )
    assert processor.image_processor.patch_size == adapter.processor.image_processor.patch_size
    with torch.no_grad():
        for sample in samples[:3]:
            encoded = adapter.encode(sample, "p0", contract_id="test")
            torch.testing.assert_close(
                reloaded(**encoded.payload).logits,
                model(**encoded.payload).logits,
                rtol=2e-5,
                atol=2e-5,
            )
    result = infer_export(
        exported,
        root / "data/validation.jsonl",
        output=root / "predictions.jsonl",
        max_new_tokens=2,
    )
    assert result["predictions"] == 6
    assert len((root / "predictions.jsonl").read_text().splitlines()) == 6


def test_export_without_inference_contract_is_rejected(tmp_path):
    (tmp_path / "omnifactory_export.json").write_text("{}")
    with pytest.raises(OmniFactoryError, match="E_EXPORT_FORMAT"):
        infer_export(tmp_path, tmp_path / "data.jsonl", output=tmp_path / "predictions.jsonl")


def test_evaluation_and_split_leak_protection(runtime, trained):
    root, _, _, _ = runtime
    report = evaluate(trained, root / "data/validation.jsonl")
    assert report["predictions"] == 6 and report["loss"] > 0
    with pytest.raises(OmniFactoryError, match="E_EVAL_SPLIT"):
        evaluate(trained, root / "data/train.jsonl")
    raw = list(iter_samples(root / "data/train.jsonl"))[0].model_dump(mode="json")
    raw["provenance"]["split"] = "validation"
    path = root / "leak.jsonl"
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(OmniFactoryError, match="E_SPLIT_LEAK"):
        evaluate(trained, path, root=root / "data")


def test_execution_rejects_unimplemented_target_and_media(runtime):
    _, _, adapter, samples = runtime
    raw = samples[2].model_dump(mode="json")
    raw["messages"][-1]["content"] = [
        {"id": "answer-text", "type": "audio", "media": {"asset_id": "input"}}
    ]
    with pytest.raises(OmniFactoryError, match="E_RUNTIME_TARGET"):
        adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="test")
    raw = samples[2].model_dump(mode="json")
    raw["messages"][0]["content"][0]["placement"] = {"timeline_id": "t", "offset_us": 0}
    with pytest.raises(OmniFactoryError, match="E_RUNTIME_MODALITY"):
        adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="test")


def test_audio_duration_budget_and_hash(runtime):
    _, config, adapter, samples = runtime
    restricted = Qwen25Adapter(adapter.processor, root=config.dataset.parent, max_audio_seconds=0.1)
    with pytest.raises(OmniFactoryError, match="E_DECODE_BUDGET"):
        restricted.encode(samples[2], "p0", contract_id="test")
    raw = samples[2].model_dump(mode="json")
    raw["assets"]["input"]["sha256"] = "0" * 64
    with pytest.raises(OmniFactoryError, match="E_MEDIA_HASH"):
        adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="test")


def test_audio_allocation_accounts_for_clip_count_and_longest_clip(runtime):
    _, config, _, _ = runtime
    costs = [
        SampleCost(
            sample_id="a",
            prediction_id="p",
            contract_id="c",
            modalities=("audio",),
            llm_tokens=10,
            audio_frames=125,
            audio_sequences=5,
            audio_sequence_length=25,
        ),
        SampleCost(
            sample_id="b",
            prediction_id="p",
            contract_id="c",
            modalities=("audio",),
            llm_tokens=10,
            audio_frames=100,
            audio_sequences=1,
            audio_sequence_length=100,
        ),
    ]
    planner = ModalBatchPlanner(config.budget)
    assert planner.cost(costs).allocated_audio_frames == 600
    assert "allocated_audio_frames" in planner.exceeded(planner.cost(costs))


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_atomic_native_assembly_preserves_other_stage_tensors(runtime, storage_dtype, tmp_path):
    _, config, _, _ = runtime
    root = tmp_path
    model = load_thinker(config)
    base = root / "native-fixture"
    base.mkdir()
    (base / "spk_dict.pt").write_bytes(b"speaker fixture")
    # Structural fixture only. These sentinel stages are not serving models.
    weights = {"thinker." + k: v.to(storage_dtype) for k, v in model.state_dict().items()}
    weights.update({"talker.sentinel": torch.arange(5.0), "token2wav.sentinel": torch.arange(7.0)})
    save_file(weights, base / "model.safetensors")
    write_json(
        base / "config.json",
        {"model_type": "qwen2_5_omni", "thinker_config": model.config.to_dict()},
    )
    with torch.no_grad():
        model.lm_head.weight.add_(0.25)
    out = root / "native-export"
    with new_directory(out) as temporary:
        result = assemble_native(base, model, temporary, shard_bytes=32768)
    assert result["shards"] > 1
    assert (out / "spk_dict.pt").read_bytes() == b"speaker fixture"
    mapping = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(mapping) == set(weights)
    for key in ("talker.sentinel", "token2wav.sentinel"):
        torch.testing.assert_close(load_file(out / mapping[key])[key], weights[key])
    torch.testing.assert_close(
        load_file(out / mapping["thinker.lm_head.weight"])["thinker.lm_head.weight"],
        model.lm_head.weight.to(storage_dtype),
    )
    (base / "spk_dict.pt").unlink()
    with pytest.raises(OmniFactoryError, match="E_NATIVE_ASSETS"):
        assemble_native(base, model, root / "incomplete-native-export")


def test_standalone_thinker_cannot_masquerade_as_native_omni(runtime, trained):
    root, _, _, _ = runtime
    with pytest.raises(OmniFactoryError, match="E_NATIVE_BASE"):
        export_checkpoint(trained, root / "bad-native-export", format="vllm-omni")
    assert not (root / "bad-native-export").exists()


def test_config_paths_resolve_from_yaml_directory(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text((Path(__file__).parents[1] / "examples/train-tiny.yaml").read_text())
    config = read_config(path)
    assert config.dataset == (tmp_path / "../outputs/demo/train.jsonl").resolve()


def test_compact_audio_preserves_native_valid_features(runtime):
    _, _, adapter, samples = runtime
    sample = samples[2]
    encoded = adapter.encode(sample, "p0", contract_id="test")
    _, _, audio, _, _ = adapter._messages(sample, "p0")
    reference = adapter.processor.feature_extractor(
        audio, sampling_rate=16000, return_attention_mask=True, return_tensors="pt"
    )
    n = int(encoded.tensors["feature_attention_mask"].sum())
    assert encoded.tensors["input_features"].shape[-1] == 27
    assert encoded.cost.audio_frames < reference["input_features"].shape[-1]
    torch.testing.assert_close(
        encoded.tensors["input_features"][:, :, :n],
        reference["input_features"][:, :, :n],
        rtol=1e-5,
        atol=1e-5,
    )


def test_audio_collation_with_variable_lengths_matches_independent(runtime):
    _, config, adapter, samples = runtime
    raw = samples[2].model_dump(mode="json")
    raw["id"] = "short"
    raw["messages"][0]["content"][0]["media"]["end_us"] = 120000
    shortened = CanonicalSample.model_validate(raw)
    examples = [adapter.encode(s, "p0", contract_id="test") for s in (samples[2], shortened)]
    model = load_thinker(config).eval()
    counts = [int((e.tensors["labels"][:, 1:] != -100).sum()) for e in examples]
    batch = adapter.collate([e.payload for e in examples])
    assert batch["input_features"].shape[0] == 2
    assert batch["input_features"].shape[-1] == max(e.cost.audio_sequence_length for e in examples)
    with torch.no_grad():
        independent = sum(model(**e.payload).loss * n for e, n in zip(examples, counts)) / sum(
            counts
        )
        actual = model(**batch).loss
    torch.testing.assert_close(actual, independent, rtol=2e-5, atol=2e-5)


def test_native_full_checkpoint_loads_only_the_thinker(runtime):
    root, config, _, _ = runtime
    model = load_thinker(config)
    base = root / "full-prefix-fixture"
    base.mkdir()
    weights = {"thinker." + name: tensor for name, tensor in model.state_dict().items()}
    weights.update({"talker.sentinel": torch.zeros(1), "token2wav.sentinel": torch.zeros(1)})
    save_file(weights, base / "model.safetensors")
    write_json(
        base / "config.json",
        {"model_type": "qwen2_5_omni", "thinker_config": model.config.to_dict()},
    )
    loaded = load_thinker(config.model_copy(update={"model": str(base)}))
    for name, parameter in loaded.state_dict().items():
        torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)


def test_gradient_accumulation_matches_token_weighted_batch(runtime):
    _, config, adapter, samples = runtime
    model, _, _ = apply_policy(load_thinker(config), config)
    model.eval()
    examples = [adapter.encode(s, "p0", contract_id="test") for s in samples[:3]]
    counts = [int((e.tensors["labels"][:, 1:] != -100).sum()) for e in examples]
    for example, count in zip(examples, counts):
        (model(**example.payload).loss * count / sum(counts)).backward()
    independent = {
        n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad()
    model(**adapter.collate([e.payload for e in examples])).loss.backward()
    for name, p in model.named_parameters():
        if name in independent:
            torch.testing.assert_close(p.grad, independent[name], rtol=5e-4, atol=2e-6)


def test_device_transfer_uses_encoder_dtype_not_quantization_compute_dtype():
    from types import SimpleNamespace

    from omnifactory.training.engine import to_device

    model = SimpleNamespace(
        audio_tower=SimpleNamespace(dtype=torch.float32),
        visual=SimpleNamespace(dtype=torch.float64),
    )
    config = SimpleNamespace(adapter="qwen2_5_omni", device="cpu", dtype="bfloat16")
    batch = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "input_features": torch.ones(1, 8, 10),
        "pixel_values": torch.ones(4, 12),
    }
    moved = to_device(batch, model, config)
    assert moved["input_features"].dtype == torch.float32
    assert moved["pixel_values"].dtype == torch.float64
    assert moved["input_ids"].dtype == torch.long
