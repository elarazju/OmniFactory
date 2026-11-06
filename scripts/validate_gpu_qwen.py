"""Pretrained Qwen2.5 Thinker CUDA acceptance across text/image/audio/video inputs.

Uses short synthetic fixtures for execution correctness, not quality. Native full
export can be passed to a separate vLLM-Omni environment; this script does not
claim server compatibility or validate Talker training.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors import safe_open
from safetensors.torch import load_file

from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.omni_demo import create_omni_demo
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import evaluate
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import merge_lora


def release():
    gc.collect()
    torch.cuda.empty_cache()


def create_understanding_data(root):
    create_omni_demo(root)
    for split in ("train", "validation"):
        rows = []
        for sample in iter_samples(root / f"{split}.jsonl"):
            raw = sample.model_dump(mode="json")
            target = raw["messages"][-1]
            target["content"] = [b for b in target["content"] if b["type"] == "text"]
            raw["predictions"][0]["target_block_ids"] = [b["id"] for b in target["content"]]
            raw["assets"].pop("target-wave")
            raw["speech_pairs"] = []
            rows.append(CanonicalSample.model_validate(raw))
        (root / f"{split}.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in rows))

    (root / "dataset_info.json").write_text(
        json.dumps(
            {
                "name": "qwen-understanding-acceptance",
                "records": {"train": 4, "validation": 4},
                "input_modalities": ["text", "image", "audio", "video_with_audio"],
                "supervision": "text",
                "synthetic": True,
                "quality_benchmark": False,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("qwen2_5_omni", "qwen3_omni"), default="qwen2_5_omni")
    parser.add_argument("--model", type=Path, required=True, help="Pinned local full HF snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this validation on the GPU host")
    if args.adapter == "qwen2_5_omni" and not (args.model / "spk_dict.pt").is_file():
        raise RuntimeError("The native Qwen snapshot must include spk_dict.pt")
    # This acceptance targets official BF16 snapshots. Native assembly retains
    # their storage precision; an FP32 LoRA merge must be rounded for comparison.
    for shard in args.model.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            assert all(
                reader.get_slice(k).get_dtype() == "BF16"
                for k in reader.keys()
                if k.startswith("thinker.")
            ), "This acceptance expects BF16 Thinker source weights"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    create_understanding_data(root / "data")
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        adapter=args.adapter,
        device="cuda",
        dtype="bfloat16",
        quantization=args.quantization,
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=4,
        max_length=1024,
        max_audio_seconds=3,
        max_video_seconds=2,
        max_video_frames=4,
        components={"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 1e-4}},
        budget={
            "max_examples": 1,
            "max_llm_tokens": 1024,
            "max_vision_patches": 4096,
            "max_audio_frames": 400,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 67108864,
        },
    )
    (root / "config.json").write_text(cfg.model_dump_json(indent=2))
    print(f"Training four input modalities, quantization={args.quantization}", flush=True)
    result = train(cfg, progress=lambda m: print(json.dumps(m), flush=True))
    release()
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    release()
    checkpoint = Path(result["checkpoint"])
    weights = load_file(checkpoint / "trainable.safetensors")
    resumed_weights = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    resume_max_abs = 0.0
    assert weights.keys() == resumed_weights.keys()
    for key in weights:
        torch.testing.assert_close(weights[key], resumed_weights[key], rtol=1e-5, atol=1e-7)
        resume_max_abs = max(
            resume_max_abs, float((weights[key] - resumed_weights[key]).abs().max())
        )
    metrics = [json.loads(s) for s in (root / "run/metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(m["loss"]) and m["grad_norm"] > 0 for m in metrics)
    assert any("lora_B" in k and v.float().abs().sum() > 0 for k, v in weights.items())
    evaluation = evaluate(checkpoint, root / "data/validation.jsonl", device="cuda")
    (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    release()
    # First check FP32 LoRA merge, then independently round the merged model to
    # native BF16 storage before checking native export/reload in FP32 compute.
    model, adapter, fp_cfg, _ = load_checkpoint(checkpoint, device="cuda", for_export=True)
    reference = []
    samples = list(iter_samples(root / "data/validation.jsonl"))
    with torch.no_grad():
        for sample in samples:
            encoded = adapter.encode(sample, "p0", contract_id="parity")
            batch = adapter.to_device(encoded.payload, model, fp_cfg)
            reference.append(adapter.forward(model, batch).logits.cpu())
    weight_reference = {}
    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            base = module.get_base_layer().weight
            # Sample rows from every adapted matrix; compute W + scale * B @ A
            # independently on CPU in float64 before the GPU FP32 merge.
            rows = torch.linspace(0, base.shape[0] - 1, 16).long()
            expected = base[rows.to(base.device)].detach().cpu().double()
            left = module.lora_B["default"].weight[rows.to(base.device)].detach().cpu().double()
            right = module.lora_A["default"].weight.detach().cpu().double()
            expected += (left @ right) * module.scaling["default"]
            weight_reference[name] = (rows, expected.float())
    model = merge_lora(model).eval()
    weight_max_abs = 0.0
    for name, (rows, expected) in weight_reference.items():
        weight = model.get_submodule(name).weight
        actual = weight[rows.to(weight.device)].detach().cpu()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        weight_max_abs = max(weight_max_abs, float((actual - expected).abs().max()))
    merge_max_abs = 0.0
    with torch.no_grad():
        for sample, expected in zip(samples, reference):
            encoded = adapter.encode(sample, "p0", contract_id="parity")
            actual = adapter.forward(
                model, adapter.to_device(encoded.payload, model, fp_cfg)
            ).logits.cpu()
            # Merged and unmerged matmul orders differ. This check bounds FP32
            # arithmetic drift; native reload below retains its stricter bound.
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)
            merge_max_abs = max(merge_max_abs, float((actual - expected).abs().max()))
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.copy_(tensor.bfloat16().float())
    rounding_max_abs = 0.0
    with torch.no_grad():
        for i, sample in enumerate(samples):
            encoded = adapter.encode(sample, "p0", contract_id="parity")
            actual = adapter.forward(
                model, adapter.to_device(encoded.payload, model, fp_cfg)
            ).logits.cpu()
            rounding_max_abs = max(rounding_max_abs, float((actual - reference[i]).abs().max()))
            reference[i] = actual
    del model, adapter, batch
    release()
    print("Assembling native full checkpoint with untouched speech stages", flush=True)
    exported = export_checkpoint(checkpoint, root / "export", format="vllm-omni")
    if args.adapter == "qwen2_5_omni":
        assert (root / "export/spk_dict.pt").read_bytes() == (
            args.model / "spk_dict.pt"
        ).read_bytes()
    release()
    reload_cfg = cfg.model_copy(
        update={
            "model": str(root / "export"),
            "dtype": "float32",
            "quantization": "none",
            "gradient_checkpointing": False,
        }
    )
    model = load_model(reload_cfg).eval()
    adapter = adapter_class(cfg.adapter).from_config(reload_cfg)
    predictions = []
    max_logit_difference = 0.0
    with torch.no_grad():
        for sample, expected in zip(samples, reference):
            encoded = adapter.encode(sample, "p0", contract_id="parity")
            actual = adapter.forward(
                model, adapter.to_device(encoded.payload, model, reload_cfg)
            ).logits.cpu()
            torch.testing.assert_close(actual, expected, atol=3e-4, rtol=1e-4)
            max_logit_difference = max(max_logit_difference, float((actual - expected).abs().max()))
            prompt = adapter.encode(sample, "p0", contract_id="generate", inference=True)
            text = adapter.generate(
                model, adapter.to_device(prompt.payload, model, reload_cfg), max_new_tokens=8
            )[0]
            predictions.append({"id": sample.id, "prediction": text})
    (root / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (source_root / "src", source_root / "scripts")
        for p in sorted(folder.rglob("*.py"))
    }
    (root / "tested-code.json").write_text(json.dumps(source_hashes, indent=2))
    from importlib.metadata import version

    report = {
        "adapter": args.adapter,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "model": str(args.model.resolve()),
        "gpu": torch.cuda.get_device_name(),
        "training": result,
        "input_modalities": ["text", "image", "audio", "video_with_audio"],
        "supervision": "text",
        "bf16_lora": args.quantization == "none",
        "quantization": args.quantization,
        "gradient_checkpointing": True,
        "cuda_resume_max_abs": resume_max_abs,
        "evaluation_loss": evaluation["loss"],
        "max_training_peak_bytes": max(m["cuda_peak_allocated_bytes"] for m in metrics),
        "native_export": exported,
        "native_speaker_asset_preserved": True if args.adapter == "qwen2_5_omni" else None,
        "fp32_lora_merge_max_logit_difference": merge_max_abs,
        "fp32_lora_merge_tolerance": {"atol": 1e-3, "rtol": 1e-4},
        "merged_weight_sample_max_abs": weight_max_abs,
        "merged_weight_matrices_checked": len(weight_reference),
        "native_bf16_storage_rounding_max_logit_difference": rounding_max_abs,
        "fp32_export_reload_max_logit_difference": max_logit_difference,
        "generated_predictions": len(predictions),
        "speech_training_verified": False,
        "nf4_verified": args.quantization == "nf4",
        "vllm_serving_verified": False,
        "data": "short synthetic execution fixtures; not a quality or general VRAM benchmark",
    }
    (root / "acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
