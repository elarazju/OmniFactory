"""Qwen3 native 16-codebook GPU train/resume/export/generation acceptance.

CPU FP32 merging avoids putting the 30B full FP32 model on one GPU. Reload
parity compares independently rounded weights in fresh BF16 native runtimes.
Teacher synthesis is execution evidence, never a speech quality benchmark.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors import safe_open
from safetensors.torch import load_file

from omnifactory.adapters.qwen3_speech import CODEC_EOS, talker_condition, talker_logits
from omnifactory.adapters.runtime import load_model
from omnifactory.data.io import iter_samples
from omnifactory.datasets.omni_demo import create_omni_demo
from omnifactory.datasets.qwen_speech import synthesize_dataset
from omnifactory.training.artifacts import write_json
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.efficiency import environment
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import merge_lora


def release():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def snapshots(model, adapter, config, samples):
    result = []
    for sample in samples:
        row = adapter.encode(sample, "p0", contract_id="speech-parity")
        batch = adapter.to_device(row.payload, model, config)
        output = model.thinker(
            **{k: v for k, v in batch.items() if k not in ("codec_labels", "speech_spans")},
            use_cache=False,
            output_hidden_states=True,
        )
        primary, residual = talker_logits(
            model,
            talker_condition(model, batch, output, *map(int, batch["speech_spans"][0])),
            batch["codec_labels"][0],
        )
        result.append(
            dict(text=output.logits.cpu(), primary=primary.cpu(), residual=residual.cpu())
        )
    return result


def decoder_hashes(root):
    """Check every frozen decoder tensor, independent of shard boundaries."""
    result = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("code2wav."):
                    tensor = f.get_tensor(key).contiguous()
                    result[key] = hashlib.sha256(
                        tensor.view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
    assert result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--data", type=Path, help="Reuse an explicitly supplied, provenance-checked teacher dataset"
    )
    p.add_argument("--teacher-only", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error("CUDA BF16 required")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = args.data.resolve() if args.data else root / "data"
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        adapter="qwen3_omni",
        task="speech_sft",
        dataset=data / "train.jsonl",
        output_dir=root / "run",
        speech_codes=data / "speech_codes.jsonl",
        speech_speaker="Ethan",
        device="cuda",
        dtype="bfloat16",
        gradient_checkpointing=True,
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=4,
        max_length=1024,
        max_audio_seconds=12,
        max_video_seconds=2,
        max_video_frames=4,
        max_codec_tokens=750,
        components={
            **{
                k: dict(mode="lora", rank=4, alpha=8, lr=1e-4)
                for k in ("language_model", "speech_generator", "code_predictor")
            },
            "speech_projection": dict(mode="full", lr=1e-5),
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=1024,
            max_vision_patches=4096,
            max_audio_frames=1200,
            max_codec_frames=750,
            max_decoded_media_bytes=67108864,
        ),
    )
    write_json(root / "config.json", cfg.model_dump(mode="json"))
    write_json(root / "tested-code.json", runtime_hashes())
    write_json(root / "environment.json", environment("cuda"))
    if args.data is None:
        create_omni_demo(root / "seed")
        print("Synthesizing eight native 16-codebook teacher targets", flush=True)
        teacher = synthesize_dataset(
            cfg,
            data,
            datasets=[root / "seed/train.jsonl", root / "seed/validation.jsonl"],
            max_codec_tokens=750,
        )
    else:
        teacher = json.loads((data / "dataset_info.json").read_text())
    write_json(root / "teacher.json", teacher)
    labels = [json.loads(line) for line in cfg.speech_codes.read_text().splitlines()]
    assert len(labels) == 8 and all(
        r["speech_finished"] and r["eos_token_id"] == CODEC_EOS for r in labels
    )
    assert all(len(f) == 16 for r in labels for f in r["codes"])
    release()
    if args.teacher_only:
        return
    print("Training Thinker, projection, Talker and residual predictor", flush=True)
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
    other = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    assert weights.keys() == other.keys()
    resume_max = 0.0
    for key in weights:
        torch.testing.assert_close(weights[key], other[key], atol=1e-7, rtol=1e-5)
        resume_max = max(resume_max, float((weights[key] - other[key]).abs().max()))
    for prefix in ("thinker.", "talker.model.", "talker.code_predictor."):
        assert any(
            k.startswith(prefix) and "lora_B" in k and v.float().abs().sum() > 0
            for k, v in weights.items()
        ), prefix
    del weights, other
    samples = list(iter_samples(data / "validation.jsonl"))
    model, adapter, _, _ = load_checkpoint(checkpoint)
    model.eval()
    unmerged = snapshots(model, adapter, cfg, samples)
    del model
    release()
    print("CPU FP32 merge with independent sampled matrix arithmetic", flush=True)
    model, _, _, _ = load_checkpoint(checkpoint, for_export=True)
    reference = {}
    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            weight = module.get_base_layer().weight
            rows = torch.linspace(0, weight.shape[0] - 1, 16).long()
            expected = weight[rows].detach().double() + module.scaling["default"] * (
                module.lora_B["default"].weight[rows].detach().double()
                @ module.lora_A["default"].weight.detach().double()
            )
            reference[name] = (rows, expected.float())
    model = merge_lora(model)
    merge_max = 0.0
    for name, (rows, expected) in reference.items():
        actual = model.get_submodule(name).weight[rows].detach()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        merge_max = max(merge_max, float((actual - expected).abs().max()))
    rounded = {
        k: v.bfloat16() if v.is_floating_point() else v for k, v in model.state_dict().items()
    }
    del model
    release()
    # Fresh construction retains native nonpersistent FP32 rotary buffers.
    model = load_model(cfg).eval()
    model.load_state_dict(rounded, strict=True)
    del rounded
    release()
    reference_outputs = snapshots(model, adapter, cfg, samples)
    drift = {
        key: max(
            float((a[key].float() - b[key].float()).abs().max())
            for a, b in zip(unmerged, reference_outputs)
        )
        for key in ("text", "primary", "residual")
    }
    del model, unmerged
    release()
    print("Assembling native full speech export", flush=True)
    exported = export_checkpoint(checkpoint, root / "export", format="vllm-omni")
    release()
    before_decoder = decoder_hashes(args.model)
    assert before_decoder == decoder_hashes(root / "export")
    reload_cfg = cfg.model_copy(
        update={"model": str(root / "export"), "gradient_checkpointing": False}
    )
    model = load_model(reload_cfg).eval()
    actual_outputs = snapshots(model, adapter, reload_cfg, samples)
    reload_diff = {key: 0.0 for key in ("text", "primary", "residual")}
    for expected, actual in zip(reference_outputs, actual_outputs):
        for key in reload_diff:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            reload_diff[key] = max(
                reload_diff[key], float((actual[key] - expected[key]).abs().max())
            )
    del model, actual_outputs, reference_outputs
    release()
    from omnifactory.training.evaluation import infer_export

    inference_rows = []
    for sample in samples:
        raw = sample.model_dump(mode="json")
        next(m for m in raw["messages"] if m["role"] == "user")["content"].append(
            dict(id="brevity", type="text", text="Answer in five words or fewer.")
        )
        inference_rows.append(raw)
    (root / "inference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inference_rows))
    inference = infer_export(
        root / "export",
        root / "inference.jsonl",
        root=data,
        output=root / "predictions.jsonl",
        device="cuda",
        dtype="bfloat16",
        generate_audio=True,
        max_new_tokens=64,
        max_codec_tokens=750,
        speaker="Ethan",
    )
    product_predictions = [
        json.loads(s) for s in (root / "predictions.jsonl").read_text().splitlines()
    ]
    assert inference["audio_predictions"] == 4
    assert all(r["text_finished"] and r["speech_finished"] for r in product_predictions)
    release()
    print("Independent full native HF generation from exported weights", flush=True)
    from transformers import Qwen3OmniMoeForConditionalGeneration

    from omnifactory.data.schema import CanonicalSample

    native = (
        Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            root / "export", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    talker_generate = native.talker.generate
    primary_outputs = []

    def capture(**kwargs):
        output = talker_generate(**kwargs)
        primary_outputs.append(output.sequences[0].cpu())
        return output

    native.talker.generate = capture
    predictions = []
    inference_rows = []
    for i, sample in enumerate(samples):
        raw = sample.model_dump(mode="json")
        next(m for m in raw["messages"] if m["role"] == "user")["content"].append(
            dict(id="brevity", type="text", text="Answer in five words or fewer.")
        )
        inference_rows.append(raw)
        encoded = adapter.encode(
            CanonicalSample.model_validate(raw),
            "p0",
            contract_id="native-inference",
            inference=True,
        )
        batch = adapter.to_device(encoded.payload, native, reload_cfg)
        text, wave = native.generate(
            **batch,
            speaker="Ethan",
            return_audio=True,
            thinker_max_new_tokens=64,
            thinker_eos_token_id=adapter.processor.tokenizer.eos_token_id,
            talker_max_new_tokens=750,
        )
        assert int(text.sequences[0, -1]) == adapter.processor.tokenizer.eos_token_id
        assert int(primary_outputs[-1][-1]) == CODEC_EOS
        wave = wave.float().cpu().reshape(-1)
        assert wave.numel() > 0 and torch.isfinite(wave).all()
        sf.write(root / f"native-{i}.wav", wave.numpy(), 24000)
        predictions.append(
            dict(
                id=sample.id,
                text=adapter.processor.tokenizer.decode(
                    text.sequences[0, batch["input_ids"].shape[1] :], skip_special_tokens=True
                ),
                audio_frames=len(wave),
                sample_rate=24000,
                text_finished=True,
                speech_finished=True,
            )
        )
        print(json.dumps(predictions[-1]), flush=True)
    (root / "inference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inference_rows))
    del native
    release()
    assert runtime_hashes() == json.loads((root / "tested-code.json").read_text()), (
        "source changed during acceptance"
    )
    metrics = [json.loads(s) for s in (root / "run/metrics.jsonl").read_text().splitlines()]
    report = dict(
        adapter="qwen3_omni",
        task="speech_sft",
        training=result,
        teacher=teacher,
        input_modalities=["text", "image", "audio", "video_with_audio"],
        codebooks=16,
        primary_eos=CODEC_EOS,
        cuda_resume_max_abs=resume_max,
        merged_weight_matrices_checked=len(reference),
        merged_weight_sample_max_abs=merge_max,
        bf16_merge_logit_drift=drift,
        bf16_export_reload_max_abs=reload_diff,
        decoder_tensors_unchanged=len(before_decoder),
        native_export=exported,
        product_inference=inference,
        predictions=predictions,
        max_training_peak_bytes=max(m["cuda_peak_allocated_bytes"] for m in metrics),
        speech_training_verified=True,
        vllm_serving_verified=False,
        quality_verified=False,
        data="native teacher synthesis of short execution fixtures; not arbitrary waveform encoding",
    )
    write_json(root / "acceptance.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
