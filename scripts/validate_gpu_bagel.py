"""BAGEL native pretrained DiT train/resume/export/image gates; synthetic images."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors import safe_open
from safetensors.torch import load_file

from omnifactory._vendor.bagel.modeling.bagel import Bagel, Qwen2ForCausalLM
from omnifactory.adapters.bagel import BagelAdapter
from omnifactory.adapters.bagel_model import load_bagel, native_config, native_training_dispatch
from omnifactory.data.io import iter_samples
from omnifactory.datasets.image_demo import create_image_demo
from omnifactory.training.artifacts import write_json
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.efficiency import environment
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import evaluate, infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import merge_lora


def release():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def snapshots(model, adapter, samples, cfg):
    values = []
    for i, sample in enumerate(samples):
        batch = adapter.to_device(
            adapter.encode(sample, "p0", contract_id="parity").payload, model, cfg
        )
        observed = []
        hook = model.llm2vae.register_forward_hook(
            lambda m, args, output: observed.append(output.detach().cpu())
        )
        try:
            torch.manual_seed(900 + i)
            with native_training_dispatch(model):
                adapter.forward(model, batch)
        finally:
            hook.remove()
        values.append(observed[0])
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--image-steps", type=int, default=16)
    args = p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    create_image_demo(root / "data", size=args.size)
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        adapter="bagel_dit",
        task="image_sft",
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        device="cuda",
        dtype="bfloat16",
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=4,
        max_length=512,
        image_height=args.size,
        image_width=args.size,
        max_image_pixels=args.size**2,
        components={
            "image_generator": dict(
                mode="lora",
                rank=4,
                alpha=8,
                lr=1e-4,
                target_modules=["q_proj_moe_gen", "v_proj_moe_gen"],
            ),
            "latent_projector": dict(mode="full", lr=1e-5),
            "image_head": dict(mode="full", lr=1e-5),
            "timestep_embedder": dict(mode="full", lr=1e-5),
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=512,
            max_vision_patches=1,
            max_audio_frames=1,
            max_codec_frames=1,
            max_latent_elements=131072,
            max_decoded_media_bytes=16777216,
        ),
    )
    write_json(root / "config.json", cfg.model_dump(mode="json"))
    write_json(root / "tested-code.json", runtime_hashes())
    write_json(root / "environment.json", environment("cuda"))
    run = train(cfg, progress=lambda row: print(json.dumps(row), flush=True))
    release()
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    release()
    checkpoint = Path(run["checkpoint"])
    a = load_file(checkpoint / "trainable.safetensors")
    b = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    del a, b
    write_json(root / "training-gate.json", dict(training=run, exact_resume=True))
    evaluation = evaluate(
        checkpoint, root / "data/validation.jsonl", output=root / "evaluation.json"
    )
    release()
    print("Independent native merge/reload parity", flush=True)
    model, _, _, _ = load_checkpoint(checkpoint, for_export=True)
    matrices = {}
    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            rows = torch.linspace(0, module.weight.shape[0] - 1, 16).long()
            expected = (
                (
                    module.get_base_layer().weight[rows].double()
                    + module.scaling["default"]
                    * (
                        module.lora_B["default"].weight[rows].double()
                        @ module.lora_A["default"].weight.double()
                    )
                )
                .detach()
                .float()
            )
            matrices[name] = (rows, expected)
    model = merge_lora(model)
    merge_diff = 0.0
    for name, (rows, expected) in matrices.items():
        actual = model.get_submodule(name).weight[rows].detach()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        merge_diff = max(merge_diff, float((actual - expected).abs().max()))
    rounded = {
        key: value.bfloat16() if value.is_floating_point() else value
        for key, value in model.state_dict().items()
    }
    del model
    release()
    model = load_bagel(cfg).eval()
    model.load_state_dict(rounded, strict=True)
    del rounded
    samples = list(iter_samples(root / "data/validation.jsonl"))
    adapter = BagelAdapter(cfg.model_copy(update={"dataset": root / "data/validation.jsonl"}))
    expected = snapshots(model, adapter, samples, cfg)
    del model, adapter
    release()
    exported = export_checkpoint(checkpoint, root / "export", format="vllm-omni")
    release()
    # Independently construct the native architecture and load its native keys.
    config = native_config(root / "export")
    with init_empty_weights():
        native = Bagel(Qwen2ForCausalLM(config.llm_config), None, config)
    with safe_open(root / "export/ema.safetensors", framework="pt", device="cpu") as source:
        state = {key: source.get_tensor(key).bfloat16() for key in native.state_dict()}
    native.load_state_dict(state, strict=True, assign=True)
    del state
    native.to("cuda").eval()
    adapter = BagelAdapter(
        cfg.model_copy(
            update={"model": str(root / "export"), "dataset": root / "data/validation.jsonl"}
        )
    )
    actual = snapshots(native, adapter, samples, cfg)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    del native, adapter, expected, actual
    release()
    # Validate every original non-updated tensor, including the unloaded vision tower.
    changed = set(exported["updated_parameters"])
    unchanged = 0
    with (
        safe_open(args.model / "ema.safetensors", framework="pt", device="cpu") as source,
        safe_open(root / "export/ema.safetensors", framework="pt", device="cpu") as target,
    ):
        assert set(source.keys()) == set(target.keys())
        for key in source.keys():
            if key not in changed:
                left, right = source.get_tensor(key), target.get_tensor(key)
                assert left.dtype == right.dtype and torch.equal(left, right), key
                unchanged += 1

    def sha(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    assert sha(args.model / "ae.safetensors") == sha(root / "export/ae.safetensors")
    write_json(
        root / "export-gate.json",
        dict(
            native_velocity_max_abs=0.0,
            merge_weight_max_abs=merge_diff,
            matrices=len(matrices),
            unchanged_tensors=unchanged,
            vae_unchanged=True,
        ),
    )
    print("Native image generation from reloaded export", flush=True)
    result = infer_export(
        root / "export",
        root / "data/validation.jsonl",
        output=root / "predictions.jsonl",
        generate_image=True,
        image_steps=args.image_steps,
        device="cuda",
        dtype="bfloat16",
    )
    assert result["image_predictions"] == len(samples)
    assert runtime_hashes() == json.loads((root / "tested-code.json").read_text())
    report = dict(
        adapter="bagel_dit",
        task="image_sft",
        training=run,
        exact_resume=True,
        native_velocity_max_abs=0.0,
        merge_weight_max_abs=merge_diff,
        unchanged_tensors=unchanged,
        vae_unchanged=True,
        evaluation=evaluation,
        product_inference=result,
        native_export=exported,
        pretrained_training_verified=not (args.model / "fixture.json").exists(),
        vllm_serving_verified=False,
        quality_verified=False,
        synthetic_images=True,
    )
    write_json(root / "acceptance.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
