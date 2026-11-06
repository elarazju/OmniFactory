"""Full pretrained MiniCPM/MiMo speech SFT, resume, native export and reload.

Short synthetic execution fixtures, not a quality benchmark. Live server
acceptance is deliberately a separate script and report.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.data.io import iter_samples
from omnifactory.datasets.mimo_demo import create_mimo_demo
from omnifactory.datasets.omni_demo import create_omni_demo
from omnifactory.training.artifacts import write_json
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.efficiency import environment
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import merge_lora
from omnifactory.training.prepare import source_fingerprint


def release():
    gc.collect()
    torch.cuda.empty_cache()


def snapshot_output(result):
    # Both families expose the shared objective sums, even when native logits
    # have different text/codec layouts. MiniCPM also has native text logits.
    tensors = {"loss": result.loss.detach().float().cpu()}
    if getattr(result, "logits", None) is not None:
        tensors["logits"] = result.logits.detach().float().cpu()
    for term in result.losses.terms:
        tensors[term.id] = term.numerator.detach().float().cpu()
    return tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", choices=("minicpm_o_4_5", "mimo_audio"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--speech-tokenizer", type=Path, required=True)
    p.add_argument("--speech-assets", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--continue-run",
        action="store_true",
        help="Reuse completed training/resume results after an export or inference failure",
    )
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the GPU host")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output.resolve()
    mini = args.adapter == "minicpm_o_4_5"
    if not args.continue_run:
        root.mkdir(parents=True, exist_ok=False)
        (create_omni_demo if mini else create_mimo_demo)(root / "data")
    codec = args.speech_tokenizer.resolve()
    if mini:
        import hashlib

        codec_hash = hashlib.sha256(codec.read_bytes()).hexdigest()
    else:
        codec_hash = source_fingerprint(str(codec), None)
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        adapter=args.adapter,
        task="speech_sft",
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        speech_tokenizer=codec,
        speech_tokenizer_sha256=codec_hash,
        speech_assets=args.speech_assets.resolve() if args.speech_assets else None,
        device="cuda",
        dtype="bfloat16",
        max_steps=2,
        save_steps=1,
        gradient_accumulation_steps=4 if mini else 3,
        max_length=2048,
        max_audio_seconds=3,
        max_video_seconds=2,
        max_video_frames=4,
        max_codec_tokens=256,
        components={
            name: {"mode": "lora", "rank": 4, "alpha": 8, "lr": 1e-4}
            for name in ("language_model", "speech_generator")
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=2048,
            max_vision_patches=16384,
            max_audio_frames=1000,
            max_codec_frames=512,
            max_decoded_media_bytes=67108864,
        ),
    )
    if args.continue_run:
        assert json.loads((root / "config.json").read_text()) == cfg.model_dump(mode="json"), (
            "continue-run requires the original model, codec and training configuration"
        )
        result = json.loads((root / "run/result.json").read_text())
        resumed = json.loads((root / "resumed/result.json").read_text())
    else:
        write_json(root / "config.json", cfg.model_dump(mode="json"))
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
    restored = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    resume_delta = 0.0
    for name in weights:
        torch.testing.assert_close(weights[name], restored[name], atol=1e-7, rtol=1e-5)
        resume_delta = max(resume_delta, float((weights[name] - restored[name]).abs().max()))
    metrics = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(m["loss"]) and m["grad_norm"] > 0 for m in metrics)
    assert any("lora_B" in n and w.abs().sum() > 0 for n, w in weights.items())
    model, adapter, export_cfg, _ = load_checkpoint(checkpoint, device="cuda", for_export=True)
    model = merge_lora(model).eval()
    # Native MiniCPM assembly preserves base storage precision. MiMo saves the
    # merged FP32 model directly, so its independent reference remains FP32.
    if mini:
        with torch.no_grad():
            for tensor in model.state_dict().values():
                if tensor.is_floating_point():
                    tensor.copy_(tensor.bfloat16().float())
    samples = list(iter_samples(root / "data/validation.jsonl"))
    references = []
    with torch.no_grad():
        for sample in samples:
            encoded = adapter.encode(sample, "p0", contract_id="reload-parity")
            references.append(
                snapshot_output(
                    adapter.forward(model, adapter.to_device(encoded.payload, model, export_cfg))
                )
            )
    del model, adapter
    release()
    if args.continue_run and (root / "export/omnifactory_export.json").is_file():
        exported = json.loads((root / "export/omnifactory_export.json").read_text())
        assert Path(exported["source_checkpoint"]).resolve() == checkpoint.resolve()
        assert exported["format"] == "vllm-omni" and exported["lora_merged"]
        # Reuse only the completed atomic export; the independent comparisons
        # below still verify its weights against the current checkpoint.
    else:
        exported = export_checkpoint(checkpoint, root / "export", format="vllm-omni")
    release()
    reload_cfg = export_cfg.model_copy(update={"model": str(root / "export"), "revision": None})
    model = load_model(reload_cfg).eval()
    adapter = adapter_class(args.adapter).from_config(reload_cfg)
    maximum = 0.0
    predictions = []
    torch.manual_seed(cfg.seed)
    with torch.no_grad():
        for sample, reference in zip(samples, references):
            encoded = adapter.encode(sample, "p0", contract_id="reload-parity")
            actual = snapshot_output(
                adapter.forward(model, adapter.to_device(encoded.payload, model, reload_cfg))
            )
            assert actual.keys() == reference.keys()
            for name in actual:
                torch.testing.assert_close(actual[name], reference[name], atol=3e-4, rtol=1e-4)
                maximum = max(maximum, float((actual[name] - reference[name]).abs().max()))
            prompt = adapter.encode(sample, "p0", contract_id="inference", inference=True)
            prediction = adapter.generate_speech(
                model,
                adapter.to_device(prompt.payload, model, reload_cfg),
                max_new_tokens=512,
                max_codec_tokens=750 if mini else 2048,
            )
            predictions.append({"id": sample.id, **prediction})
    write_json(root / "native-predictions.json", predictions)
    # Decode one model-generated code sequence from the independent export.
    # No replacement or forced ground-truth codec labels are used.
    generated = next((p for p in predictions if p["codec_tokens"]), None)
    assert generated is not None, "no model-generated speech codes"
    assert all(p["codec_tokens"] for p in predictions), "missing generated speech codes"
    assert all(
        p["text_finished"] and p["speech_finished"] if mini else p["finished"] for p in predictions
    ), "native generation reached a token limit instead of EOS"
    reference_audio = root / "export/assets/HT_ref_audio.wav" if mini else None
    decoder = adapter.speech_decoder(model, reference_audio=reference_audio)
    waveform, rate = decoder(generated["codec_tokens"])
    import numpy as np
    import soundfile as sf

    assert rate == 24000 and len(waveform) and np.isfinite(waveform).all()
    sf.write(root / "native-generated.wav", waveform, rate)
    write_json(root / "tested-code.json", runtime_hashes())
    report = dict(
        adapter=args.adapter,
        environment=environment("cuda"),
        model=cfg.model,
        training=result,
        resume_max_abs=resume_delta,
        native_export=exported,
        reload_max_abs=maximum,
        generated_predictions=len(predictions),
        predictions_with_codes=sum(bool(p["codec_tokens"]) for p in predictions),
        native_eos_verified=True,
        max_training_peak_bytes=max(m["cuda_peak_allocated_bytes"] for m in metrics),
        native_waveform_frames=len(waveform),
        native_waveform_sample_rate=rate,
        speech_training_verified=True,
        vllm_serving_verified=False,
        quality_verified=False,
    )
    write_json(root / "acceptance.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
