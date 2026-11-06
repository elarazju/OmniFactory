"""Native pretrained VoxCPM2 CFM/stop training, resume, export and waveform gates."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file

from omnifactory._vendor.voxcpm2.model.voxcpm2 import VoxCPM2Model
from omnifactory.adapters.voxcpm2 import VoxCPM2Adapter
from omnifactory.adapters.voxcpm2_model import load_voxcpm2
from omnifactory.data.io import iter_samples
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
        observed = {}
        hooks = [
            model.feat_decoder.estimator.register_forward_hook(
                lambda m, a, o: observed.update(velocity=o.detach().cpu())
            ),
            model.stop_head.register_forward_hook(
                lambda m, a, o: observed.update(stop=o.detach().cpu())
            ),
        ]
        try:
            torch.manual_seed(100 + i)
            adapter.forward(model, batch)
        finally:
            for h in hooks:
                h.remove()
        values.append(observed)
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root, data = args.output.resolve(), args.data.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        adapter="voxcpm2",
        task="speech_sft",
        dataset=data / "train.jsonl",
        output_dir=root / "run",
        device="cuda",
        dtype="bfloat16",
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=4,
        max_length=512,
        max_audio_seconds=8,
        max_codec_tokens=128,
        components={
            **{
                k: dict(mode="lora", rank=4, alpha=8, lr=1e-4)
                for k in ("language_model", "residual_model", "audio_decoder")
            },
            "speech_projection": dict(mode="full", lr=1e-5),
            "stop_predictor": dict(mode="full", lr=1e-5),
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=512,
            max_vision_patches=1,
            max_audio_frames=1024,
            max_codec_frames=1,
            max_latent_elements=131072,
            max_decoded_media_bytes=16777216,
        ),
    )
    write_json(root / "config.json", cfg.model_dump(mode="json"))
    write_json(root / "tested-code.json", runtime_hashes())
    write_json(root / "environment.json", environment("cuda"))
    print("Native continuous CFM + stop training", flush=True)
    run = train(cfg, progress=lambda r: print(json.dumps(r), flush=True))
    release()
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    checkpoint = Path(run["checkpoint"])
    weights = load_file(checkpoint / "trainable.safetensors")
    other = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    assert weights.keys() == other.keys()
    for name in weights:
        torch.testing.assert_close(weights[name], other[name], atol=0, rtol=0)
    write_json(root / "training-gate.json", dict(training=run, exact_resume=True))
    del weights, other
    release()
    evaluation = evaluate(checkpoint, data / "validation.jsonl", output=root / "evaluation.json")
    release()
    print("Independent FP64 merge arithmetic; native BF16 reload parity", flush=True)
    model, adapter, _, _ = load_checkpoint(checkpoint, for_export=True)
    matrices = {}
    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            rows = torch.linspace(0, module.weight.shape[0] - 1, 16).long()
            matrices[name] = (
                rows,
                (
                    module.get_base_layer().weight[rows].double()
                    + module.scaling["default"]
                    * (
                        module.lora_B["default"].weight[rows].double()
                        @ module.lora_A["default"].weight.double()
                    )
                )
                .detach()
                .float(),
            )
    model = merge_lora(model)
    merge_diff = 0.0
    for name, (rows, expected) in matrices.items():
        actual = model.get_submodule(name).weight[rows].detach()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        merge_diff = max(merge_diff, float((actual - expected).abs().max()))
    rounded = {
        k: (v if k.startswith("audio_vae.") or not v.is_floating_point() else v.bfloat16())
        for k, v in model.state_dict().items()
    }
    del model
    release()
    model = load_voxcpm2(cfg).eval()
    model.load_state_dict(rounded, strict=True)
    del rounded
    samples = list(iter_samples(data / "validation.jsonl"))
    adapter = VoxCPM2Adapter.from_config(
        cfg.model_copy(update={"dataset": data / "validation.jsonl"})
    )
    expected = snapshots(model, adapter, samples, cfg)
    del model, adapter
    release()
    exported = export_checkpoint(checkpoint, root / "export", format="vllm-omni")
    release()
    # Load directly through the unmodified native class, independent of our loader.
    native = VoxCPM2Model.from_local(str(root / "export"), optimize=False, device="cuda")
    adapter = VoxCPM2Adapter.from_config(
        cfg.model_copy(update={"model": str(root / "export"), "dataset": data / "validation.jsonl"})
    )
    actual = snapshots(native, adapter, samples, cfg)
    differences = {"velocity": 0.0, "stop": 0.0}
    for left, right in zip(expected, actual):
        for key in differences:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
            differences[key] = max(differences[key], float((left[key] - right[key]).abs().max()))
    del native, adapter, actual, expected
    release()
    vae_name = (
        "audiovae.safetensors" if (args.model / "audiovae.safetensors").exists() else "audiovae.pth"
    )

    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    assert digest(args.model / vae_name) == digest(root / "export" / vae_name)
    write_json(
        root / "export-gate.json",
        dict(
            native_reload_max_abs=differences,
            merge_weight_max_abs=merge_diff,
            matrices=len(matrices),
            vae_unchanged=True,
        ),
    )
    print("Product text/reference → latent → 48 kHz waveform inference", flush=True)
    result = infer_export(
        root / "export",
        data / "validation.jsonl",
        output=root / "predictions.jsonl",
        device="cuda",
        dtype="bfloat16",
        generate_audio=True,
        max_codec_tokens=128,
    )
    rows = [json.loads(s) for s in (root / "predictions.jsonl").read_text().splitlines()]
    assert result["audio_predictions"] == len(samples)
    assert all(r["speech_finished"] and r["sample_rate"] == 48000 for r in rows)
    assert runtime_hashes() == json.loads((root / "tested-code.json").read_text())
    report = dict(
        adapter="voxcpm2",
        task="speech_sft",
        training=run,
        exact_resume=True,
        native_reload_max_abs=differences,
        merge_weight_max_abs=merge_diff,
        vae_unchanged=True,
        evaluation=evaluation,
        product_inference=result,
        native_export=exported,
        pretrained_training_verified=True,
        vllm_serving_verified=False,
        quality_verified=False,
    )
    write_json(root / "acceptance.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
