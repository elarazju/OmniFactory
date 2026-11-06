"""Native continuous/discrete dual-objective TTS, including full engine/export paths."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
from safetensors.torch import load_file

from omnifactory._vendor.voxcpm2.model.voxcpm2 import VoxCPM2Model
from omnifactory.adapters.voxcpm2 import VoxCPM2Adapter
from omnifactory.adapters.voxcpm2_model import load_voxcpm2
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.flow_demo import create_flow_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.adaptation import ComponentPolicy
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import apply_policy
from omnifactory.training.tiny_voxcpm2 import create_tiny_voxcpm2


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("voxcpm2")
    create_tiny_voxcpm2(root / "model")
    create_flow_demo(root / "data")
    for split in ("train", "validation"):
        rows = []
        for sample in iter_samples(root / "data" / f"{split}.jsonl"):
            row = sample.model_dump(mode="json")
            text = "Hello world."
            row["messages"][0]["content"][0]["text"] = text
            row["messages"][1]["content"].append({"id": "transcript", "type": "text", "text": text})
            row["predictions"][0]["target_block_ids"].append("transcript")
            if sample.id == "flow-1":
                row["predictions"][0]["conditions"] = []
            row["speech_pairs"] = [{"text_block_id": "transcript", "audio_block_id": "audio"}]
            rows.append(CanonicalSample.model_validate(row).model_dump(mode="json"))
        (root / "data" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = TrainConfig(
        adapter="voxcpm2",
        task="speech_sft",
        model=str(root / "model"),
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        max_length=256,
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=2,
        components={
            k: dict(mode="lora", rank=2, alpha=4, lr=1e-3)
            for k in ("language_model", "residual_model", "audio_decoder")
        },
        budget=dict(
            max_examples=2,
            max_llm_tokens=512,
            max_audio_frames=512,
            max_vision_patches=1,
            max_codec_frames=1,
            max_decoded_media_bytes=1048576,
            max_latent_elements=65536,
        ),
    )
    return root, cfg, VoxCPM2Adapter.from_config(cfg), list(iter_samples(cfg.dataset))


def test_native_objective_and_supervision(runtime):
    _, cfg, adapter, samples = runtime
    model = load_voxcpm2(cfg).eval()
    for sample in samples:
        row = adapter.encode(sample, "p0", contract_id="test")
        batch = row.payload
        n = int(batch["loss_mask"].sum())
        assert row.normalization_counts == {"latent_frames": 4 * n, "stop_symbols": n}
        assert row.supervision.targets[0].representation.kind == "continuous"
        assert row.supervision.targets[1].representation.kind == "discrete"
        assert batch["labels"][batch["loss_mask"]].sum() == 1
        assert batch["labels"][0, -2] == 1
        assert not batch["loss_mask"][0, -1]
        torch.manual_seed(91)
        native = model(**batch)
        torch.manual_seed(91)
        result = adapter.forward(model, batch)
        torch.testing.assert_close(
            result.losses.loss, native["loss/diff"] + native["loss/stop"], atol=0, rtol=0
        )
        prompt = adapter.encode(sample, "p0", contract_id="infer", inference=True).validate(
            inference=True
        )
        assert not prompt.supervision.targets and "audio_feats" not in prompt.tensors
        # Target edits cannot affect the input representation.
        changed = sample.model_dump(mode="json")
        changed["messages"][1]["content"][1]["text"] = "secret answer"
        prompt2 = adapter.encode(
            CanonicalSample.model_validate(changed), "p0", contract_id="infer", inference=True
        )
        assert prompt.layout == prompt2.layout
        for key in prompt.tensors:
            torch.testing.assert_close(prompt.tensors[key], prompt2.tensors[key], atol=0, rtol=0)


def test_preparation_is_identical_across_cold_and_warm_encoder(runtime):
    _, cfg, _, samples = runtime
    adapter = VoxCPM2Adapter(cfg)
    cold = adapter.encode(samples[0], "p0", contract_id="repeat")
    for _ in range(3):
        warm = adapter.encode(samples[0], "p0", contract_id="repeat")
        for key in cold.tensors:
            torch.testing.assert_close(cold.tensors[key], warm.tensors[key], atol=0, rtol=0)
    assert cold.metadata["encoder_execution"] == "torchscript_unoptimized_fp32"


@pytest.mark.parametrize("component", ["audio_projector", "speech_projection", "audio_decoder"])
def test_gradient_through_frozen_native_stages(runtime, component):
    _, cfg, adapter, samples = runtime
    local = cfg.model_copy(
        update={"components": {component: ComponentPolicy(mode="full", lr=1e-3)}}
    )
    model, _, _ = apply_policy(load_voxcpm2(local), local)
    model.train()
    result = adapter.forward(model, adapter.encode(samples[0], "p0", contract_id="test").payload)
    result.losses.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
    )
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(not p.requires_grad for p in model.audio_vae.parameters())


def test_train_resume_export_native_generation(runtime):
    root, cfg, adapter, samples = runtime
    run = train(cfg)
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    weights, other = (
        load_file(Path(run["checkpoint"]) / "trainable.safetensors"),
        load_file(Path(resumed["checkpoint"]) / "trainable.safetensors"),
    )
    assert weights.keys() == other.keys()
    for key in weights:
        torch.testing.assert_close(weights[key], other[key], atol=0, rtol=0)
    model, _, _, _ = load_checkpoint(Path(run["checkpoint"]))
    batch = adapter.encode(samples[0], "p0", contract_id="test").payload
    torch.manual_seed(44)
    before = adapter.forward(model, batch).losses.loss
    export_checkpoint(Path(run["checkpoint"]), root / "export", format="vllm-omni")
    assert (root / "model/audiovae.safetensors").read_bytes() == (
        root / "export/audiovae.safetensors"
    ).read_bytes()
    reload = VoxCPM2Model.from_local(str(root / "export"), device="cpu", optimize=False)
    torch.manual_seed(44)
    after = adapter.forward(reload, batch).losses.loss
    torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-5)
    result = infer_export(
        root / "export",
        root / "data/validation.jsonl",
        output=root / "generated.jsonl",
        generate_audio=True,
        max_codec_tokens=4,
    )
    assert result["audio_predictions"] == 1
    record = json.loads((root / "generated.jsonl").read_text())
    assert "audio_latents" in record and "codec_tokens" not in record
    assert record["sample_rate"] == 48000


def test_reject_mismatched_tts_transcript(runtime):
    _, _, adapter, samples = runtime
    raw = samples[0].model_dump(mode="json")
    raw["messages"][1]["content"][1]["text"] = "wrong"
    with pytest.raises(OmniFactoryError, match="E_TTS_TARGET"):
        adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="test")


def test_native_diffusion_condition_cannot_read_current_or_future_targets(runtime):
    _, cfg, adapter, samples = runtime
    model = load_voxcpm2(cfg).eval()
    batch = adapter.encode(samples[0], "p0", contract_id="causal").payload
    batch["audio_feats"] = batch["audio_feats"].clone().requires_grad_(True)
    observed = {}
    hook = model.feat_decoder.estimator.register_forward_pre_hook(
        lambda m, args: observed.update(mu=args[1])
    )
    try:
        adapter.forward(model, batch)
    finally:
        hook.remove()
    target = int(batch["loss_mask"][0].nonzero()[1])
    grad = torch.autograd.grad(observed["mu"][target].sum(), batch["audio_feats"])[0]
    assert torch.equal(grad[:, target:], torch.zeros_like(grad[:, target:]))
    assert grad[:, :target].abs().sum() > 0


def test_bf16_runtime_never_rounds_frozen_vae_weights(runtime):
    root, cfg, _, _ = runtime
    # A loader-level CPU test isolates storage precision; CPU BF16 training remains disabled.
    model = load_voxcpm2(cfg.model_copy(update={"dtype": "bfloat16"}))
    original = load_file(root / "model/audiovae.safetensors")
    for key, value in model.audio_vae.state_dict().items():
        assert value.dtype == original[key].dtype
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)
