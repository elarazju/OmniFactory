"""Pretrained Qwen Thinker/Talker SFT, native export and CUDA WAV acceptance."""

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file
from validate_gpu_qwen import release

from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.data.io import iter_samples
from omnifactory.datasets.omni_demo import create_omni_demo
from omnifactory.datasets.qwen_speech import synthesize_dataset
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import evaluate, infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import merge_lora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-codec-tokens", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    create_omni_demo(root / "seed")
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        task="speech_sft",
        adapter="qwen2_5_omni",
        dataset=root / "seed/train.jsonl",
        output_dir=root / "run",
        speech_codes=root / "data/speech_codes.jsonl",
        speech_speaker="Chelsie",
        device="cuda",
        dtype="bfloat16",
        gradient_checkpointing=False,
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=4,
        max_length=1024,
        max_audio_seconds=12,
        max_video_seconds=2,
        max_video_frames=4,
        max_codec_tokens=args.max_codec_tokens,
        components={
            k: {"mode": "lora", "rank": 4, "alpha": 8, "lr": 1e-4}
            for k in ("language_model", "speech_generator")
        },
        budget={
            "max_examples": 1,
            "max_llm_tokens": 1024,
            "max_vision_patches": 4096,
            "max_audio_frames": 1200,
            "max_codec_frames": args.max_codec_tokens,
            "max_decoded_media_bytes": 67108864,
        },
    )
    print("Synthesizing native teacher codes and WAV targets", flush=True)
    teacher = synthesize_dataset(
        cfg,
        root / "data",
        datasets=[root / "seed/train.jsonl", root / "seed/validation.jsonl"],
        max_codec_tokens=args.max_codec_tokens,
    )
    labels = [json.loads(s) for s in (root / "data/speech_codes.jsonl").read_text().splitlines()]
    assert len(labels) == 8 and all(x["speech_finished"] for x in labels)
    release()
    cfg = cfg.model_copy(
        update={"dataset": root / "data/train.jsonl", "gradient_checkpointing": True}
    )
    (root / "config.json").write_text(cfg.model_dump_json(indent=2))
    print("Training Thinker and Talker on four input modalities", flush=True)
    result = train(cfg, progress=lambda m: print(json.dumps(m), flush=True))
    release()
    resumed = train(
        cfg.model_copy(
            update={
                "output_dir": root / "resumed",
                "resume_from": root / "run/checkpoint-00000001",
            }
        )
    )
    release()
    checkpoint = Path(result["checkpoint"])
    weights = load_file(checkpoint / "trainable.safetensors")
    other = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    resume_difference = 0.0
    assert weights.keys() == other.keys()
    for k in weights:
        torch.testing.assert_close(weights[k], other[k], atol=1e-7, rtol=1e-5)
        resume_difference = max(resume_difference, float((weights[k] - other[k]).abs().max()))
    for stage in ("thinker.", "talker."):
        assert any(
            k.startswith(stage) and "lora_B" in k and v.float().abs().sum() > 0
            for k, v in weights.items()
        ), f"{stage} did not update"
    metrics = [json.loads(s) for s in (root / "run/metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(m["loss"]) and m["grad_norm"] > 0 for m in metrics)
    evaluation = evaluate(checkpoint, root / "data/validation.jsonl", device="cuda")
    (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    release()
    print("Checking native BF16 merged checkpoint reload", flush=True)
    model, adapter, fp_cfg, _ = load_checkpoint(checkpoint, device="cuda", for_export=True)
    model = merge_lora(model).eval()
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.copy_(tensor.bfloat16().float())
    samples = list(iter_samples(root / "data/validation.jsonl"))
    reference = []
    with torch.no_grad():
        for sample in samples:
            encoded = adapter.encode(sample, "p0", contract_id="parity")
            forward = adapter.forward(model, adapter.to_device(encoded.payload, model, fp_cfg))
            reference.append((forward.logits.cpu(), float(forward.codec_loss)))
    del model, adapter, forward
    release()
    exported = export_checkpoint(checkpoint, root / "export", format="vllm-omni")
    assert (root / "export/spk_dict.pt").read_bytes() == (args.model / "spk_dict.pt").read_bytes()
    release()
    reload_cfg = cfg.model_copy(
        update={
            "model": str(root / "export"),
            "dtype": "float32",
            "gradient_checkpointing": False,
        }
    )
    model = load_model(reload_cfg).eval()
    adapter = adapter_class(cfg.adapter).from_config(reload_cfg)
    # Labels retain original-source provenance; only the forward uses the export.
    source_adapter = adapter_class(cfg.adapter).from_config(fp_cfg)
    max_logit_difference = max_codec_loss_difference = 0.0
    with torch.no_grad():
        for sample, (logits, codec_loss) in zip(samples, reference):
            encoded = source_adapter.encode(sample, "p0", contract_id="parity")
            forward = adapter.forward(model, adapter.to_device(encoded.payload, model, reload_cfg))
            torch.testing.assert_close(forward.logits.cpu(), logits, atol=3e-4, rtol=1e-4)
            torch.testing.assert_close(float(forward.codec_loss), codec_loss, atol=3e-4, rtol=1e-4)
            max_logit_difference = max(
                max_logit_difference, float((forward.logits.cpu() - logits).abs().max())
            )
            max_codec_loss_difference = max(
                max_codec_loss_difference, abs(float(forward.codec_loss) - codec_loss)
            )
    del model, adapter, source_adapter, forward
    release()
    print("Generating text and native WAV on CUDA from exported weights", flush=True)
    # Keep perception inputs, but bound the requested reply so EOS is exercised
    # rather than treating a token-limit cutoff as complete speech.
    inference_rows = []
    for sample in samples:
        raw = sample.model_dump(mode="json")
        user = next(m for m in raw["messages"] if m["role"] == "user")
        user["content"].append(
            {"id": "brevity", "type": "text", "text": "Answer in five words or fewer."}
        )
        inference_rows.append(raw)
    (root / "inference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inference_rows))
    inference = infer_export(
        root / "export",
        root / "inference.jsonl",
        root=root / "data",
        output=root / "predictions.jsonl",
        device="cuda",
        dtype="bfloat16",
        generate_audio=True,
        max_new_tokens=64,
        max_codec_tokens=750,
        speaker="Chelsie",
    )
    assert inference["audio_predictions"] == 4
    predictions = [json.loads(s) for s in (root / "predictions.jsonl").read_text().splitlines()]
    assert all(r["speech_finished"] and r["text_finished"] for r in predictions), (
        "Generation reached its limit"
    )
    audio = []
    for row in predictions:
        waveform, rate = sf.read(row["audio"])
        assert rate == 24000 and len(waveform) > 0 and torch.isfinite(torch.tensor(waveform)).all()
        audio.append(
            {
                "id": row["sample_id"],
                "frames": len(waveform),
                "sample_rate": rate,
                "speech_finished": row["speech_finished"],
                "text_finished": row["text_finished"],
            }
        )
    source = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (source / "src", source / "scripts")
        for p in sorted(folder.rglob("*.py"))
    }
    (root / "tested-code.json").write_text(json.dumps(hashes, indent=2))
    report = {
        "model": str(args.model.resolve()),
        "gpu": torch.cuda.get_device_name(),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "teacher": teacher,
        "teacher_codec_lengths": [len(x["codes"]) for x in labels],
        "training": result,
        "cuda_resume_max_abs": resume_difference,
        "max_training_peak_bytes": max(m["cuda_peak_allocated_bytes"] for m in metrics),
        "evaluation_loss": evaluation["loss"],
        "native_export": exported,
        "native_export_reload_max_logit_difference": max_logit_difference,
        "native_export_reload_max_codec_loss_difference": max_codec_loss_difference,
        "inference": inference,
        "audio": audio,
        "input_modalities": ["text", "image", "audio", "video_with_audio"],
        "supervision": ["text", "native_audio_codes"],
        "speech_training_verified": True,
        "vllm_serving_verified": False,
        "speech_quality_verified": False,
        "data": "short synthetic prompts and explicit native teacher targets; not a quality benchmark",
    }
    (root / "acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
