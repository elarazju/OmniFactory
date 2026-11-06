"""Native BAGEL continuous image supervision, causal isolation, resume and export."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from omnifactory.adapters.bagel import BagelAdapter
from omnifactory.adapters.bagel_model import load_bagel, native_config
from omnifactory.data.schema import CanonicalSample
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train
from omnifactory.training.evaluation import infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import apply_policy
from omnifactory.training.tiny_bagel import create_tiny_bagel


def test_stale_config_uses_checkpoint_position_geometry(tmp_path):
    tmp_path = tmp_path / "model"
    create_tiny_bagel(tmp_path)
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw["max_latent_size"] = 8
    path.write_text(json.dumps(raw))
    assert native_config(tmp_path).max_latent_size == 16


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("bagel")
    create_tiny_bagel(root / "model")
    data = root / "data"
    data.mkdir()
    rows = []
    for i, color in enumerate(["red", "blue", "green"]):
        image = Image.new("RGB", (32, 32), "white")
        ImageDraw.Draw(image).ellipse((5, 5, 27, 27), fill=color)
        image.save(data / f"{i}.png")
        rows.append(
            CanonicalSample.model_validate(
                dict(
                    schema_version="0.1",
                    id=f"image-{i}",
                    provenance=dict(
                        dataset="image-fixture",
                        source_id=str(i),
                        group_id=str(i),
                        split="validation" if i == 2 else "train",
                    ),
                    assets={"target": dict(uri=f"{i}.png", modality="image")},
                    messages=[
                        dict(
                            id="user",
                            role="user",
                            content=[dict(id="prompt", type="text", text=f"A {color} circle.")],
                        ),
                        dict(
                            id="assistant",
                            role="assistant",
                            content=[dict(id="image", type="image", asset_id="target")],
                        ),
                    ],
                    predictions=[
                        dict(id="p0", target_message_id="assistant", target_block_ids=["image"])
                    ],
                )
            )
        )
    for split in ["train", "validation"]:
        (data / f"{split}.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in rows if r.provenance.split == split)
        )
    cfg = TrainConfig(
        adapter="bagel_dit",
        task="image_sft",
        model=str(root / "model"),
        dataset=data / "train.jsonl",
        output_dir=root / "run",
        image_height=32,
        image_width=32,
        max_length=256,
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=2,
        components={
            "image_generator": dict(
                mode="lora",
                rank=2,
                alpha=4,
                lr=1e-3,
                target_modules=["q_proj_moe_gen", "v_proj_moe_gen"],
            )
        },
        budget=dict(
            max_examples=2,
            max_llm_tokens=512,
            max_vision_patches=1,
            max_audio_frames=1,
            max_codec_frames=1,
            max_latent_elements=65536,
            max_decoded_media_bytes=65536,
        ),
    )
    return root, cfg, rows[:2]


def test_native_flow_and_target_free_inference(runtime):
    _, cfg, samples = runtime
    adapter = BagelAdapter(cfg)
    model = load_bagel(cfg)
    encoded = adapter.encode(samples[0], "p0", contract_id="test")
    assert encoded.supervision.targets[0].representation.kind == "continuous"
    count = encoded.tensors["clean_patches"].numel()
    assert encoded.normalization_counts == {"latent_elements": count}
    torch.manual_seed(33)
    output = adapter.forward(model, encoded.payload)
    torch.manual_seed(33)
    batch = encoded.payload
    timestep = torch.randn(()).expand(len(batch["packed_vae_token_indexes"]))
    native = model(**batch, packed_timesteps=timestep)
    torch.testing.assert_close(output.losses.loss, native["mse"].mean(), atol=0, rtol=0)
    again = adapter.encode(samples[0], "p0", contract_id="test")
    torch.testing.assert_close(
        encoded.tensors["clean_patches"], again.tensors["clean_patches"], atol=0, rtol=0
    )
    prompt = adapter.encode(samples[0], "p0", contract_id="test", inference=True).validate(
        inference=True
    )
    changed = samples[0].model_copy(deep=True)
    changed.assets["target"] = changed.assets["target"].model_copy(
        update={"uri": "nonexistent-private-target.png"}
    )
    other = adapter.encode(changed, "p0", contract_id="test", inference=True)
    assert prompt.payload == other.payload and not prompt.supervision.targets


def test_image_attention_is_bidirectional_and_text_cannot_see_target(runtime):
    _, cfg, samples = runtime
    adapter = BagelAdapter(cfg)
    batch = adapter.encode(samples[0], "p0", contract_id="test").payload
    mask = batch["nested_attention_masks"][0]
    image = batch["packed_vae_token_indexes"]
    first = int(image[0])
    last = int(image[-1])
    assert mask[first, last] == 0 and mask[last, first] == 0
    assert torch.isneginf(mask[: first - 1, first : last + 1]).all()
    model = load_bagel(cfg)
    # Native understanding branch remains unchanged when only target image changes.
    observed = []
    hook = model.language_model.register_forward_hook(
        lambda m, a, o: observed.append(o.detach().clone())
    )
    try:
        torch.manual_seed(2)
        adapter.forward(model, batch)
        altered = {**batch, "padded_latent": batch["padded_latent"] + 100}
        torch.manual_seed(2)
        adapter.forward(model, altered)
    finally:
        hook.remove()
    torch.testing.assert_close(observed[0][: first - 1], observed[1][: first - 1], atol=0, rtol=0)
    assert not torch.equal(observed[0][image], observed[1][image])


def test_backward_resume_export_and_image_generation(runtime):
    root, cfg, samples = runtime
    model, _, _ = apply_policy(load_bagel(cfg), cfg)
    adapter = BagelAdapter(cfg)
    adapter.forward(
        model, adapter.encode(samples[0], "p0", contract_id="test").payload
    ).losses.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
    )
    run = train(cfg)
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    a = load_file(Path(run["checkpoint"]) / "trainable.safetensors")
    b = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    export_checkpoint(Path(run["checkpoint"]), root / "export", format="vllm-omni")
    assert (root / "model/ae.safetensors").read_bytes() == (
        root / "export/ae.safetensors"
    ).read_bytes()
    index = json.loads((root / "export/model.safetensors.index.json").read_text())
    expected_map, total_size = {}, 0
    for filename in ["ema.safetensors", "ae.safetensors"]:
        for key, value in load_file(root / "export" / filename).items():
            expected_map[key] = filename
            total_size += value.numel() * value.element_size()
    assert index["weight_map"] == expected_map
    assert index["metadata"]["total_size"] == total_size
    result = infer_export(
        root / "export",
        root / "data/validation.jsonl",
        output=root / "images.jsonl",
        generate_image=True,
        image_steps=3,
    )
    assert result["image_predictions"] == 1
    row = json.loads((root / "images.jsonl").read_text())
    with Image.open(row["image"]) as image:
        assert image.size == (32, 32)
