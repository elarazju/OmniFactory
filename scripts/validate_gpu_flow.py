"""Pinned pretrained Token2Wav GPU acceptance, including native export and waveform reload."""

import argparse
import gc
import json
from pathlib import Path

import soundfile as sf
import torch

from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.adapters.token2wav import NativeToken2Wav
from omnifactory.data.io import iter_samples
from omnifactory.datasets.flow_demo import create_flow_demo
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import evaluate
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import apply_policy, merge_lora


def main():
    # Evaluate merge/reload parity without TF32 rounding obscuring small LoRA updates.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-run",
        action="store_true",
        help="recheck completed training without modifying its weights",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.reuse_run:
        root.mkdir(parents=True, exist_ok=False)
        create_flow_demo(root / "data")
    cfg = TrainConfig(
        model=str(args.assets.resolve()),
        adapter="minicpm_token2wav",
        task="flow_sft",
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        device="cuda",
        dtype="float32",
        max_steps=4,
        save_steps=2,
        gradient_accumulation_steps=2,
        flow_cfg_dropout=0.2,
        components={
            "audio_decoder": {
                "mode": "lora",
                "rank": 8,
                "alpha": 16,
                "lr": 1e-4,
                "target_modules": ["to_q", "to_v"],
            }
        },
        budget={
            "max_examples": 1,
            "max_llm_tokens": 1,
            "max_vision_patches": 1,
            "max_audio_frames": 200,
            "max_codec_frames": 100,
            "max_decoded_media_bytes": 1000000,
            "max_latent_elements": 40000,
        },
    )
    if args.reuse_run:
        old_config = TrainConfig.model_validate_json((root / "config.json").read_text())
        if old_config.execution_contract() != cfg.execution_contract():
            raise ValueError("Existing acceptance config differs")
        result = json.loads((root / "run/result.json").read_text())
        resumed = json.loads((root / "resumed/result.json").read_text())
    else:
        (root / "config.json").write_text(cfg.model_dump_json(indent=2))
        result = train(cfg, progress=lambda m: print(json.dumps(m), flush=True))
        resumed = train(
            cfg.model_copy(
                update={
                    "output_dir": root / "resumed",
                    "resume_from": root / "run/checkpoint-00000002",
                }
            )
        )
    checkpoint = Path(result["checkpoint"])
    from safetensors.torch import load_file

    original_weights = load_file(checkpoint / "trainable.safetensors")
    resumed_weights = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    for name in original_weights:
        torch.testing.assert_close(
            original_weights[name], resumed_weights[name], rtol=1e-5, atol=1e-7
        )

    metrics = [json.loads(s) for s in (root / "run/metrics.jsonl").read_text().splitlines()]
    assert all(m["grad_norm"] > 0 for m in metrics)
    evaluation = evaluate(
        checkpoint, root / "data/validation.jsonl", device="cuda", output=root / "evaluation.json"
    )
    model, adapter, config, _ = load_checkpoint(checkpoint, device="cuda")
    sample = next(iter_samples(cfg.dataset))
    encoded = adapter.encode(sample, "p0", contract_id="parity")
    row = adapter.to_device(encoded.payload, model, config)
    with torch.no_grad():
        before = model(row).velocity.cpu()
        merged = merge_lora(model)
        after = merged(row).velocity.cpu()
    torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-4)
    trained_state = {
        k: v.detach().cpu().clone() for k, v in merged.flow.decoder.estimator.state_dict().items()
    }
    del model, merged, adapter, row
    gc.collect()
    torch.cuda.empty_cache()
    exported = (
        json.loads((root / "export/omnifactory_export.json").read_text())
        if args.reuse_run and (root / "export").exists()
        else export_checkpoint(checkpoint, root / "export")
    )
    reload_cfg = cfg.model_copy(update={"model": str(root / "export")})
    reloaded = load_model(reload_cfg).eval()
    export_weights = torch.load(root / "export/flow.pt", map_location="cpu", weights_only=True)
    merge_max_abs = 0.0
    for key, value in trained_state.items():
        actual = reloaded.flow.decoder.estimator.state_dict()[key].cpu()
        # CPU and CUDA matrix products used by LoRA merging have roundoff differences.
        torch.testing.assert_close(actual, value, rtol=1e-6, atol=1e-8)
        merge_max_abs = max(merge_max_abs, float((actual - value).abs().max()))
        torch.testing.assert_close(
            actual, export_weights["decoder.estimator." + key], rtol=0, atol=0
        )
    del export_weights
    reload_adapter = adapter_class(cfg.adapter).from_config(reload_cfg)
    with torch.no_grad():
        reloaded_output = reloaded(
            reload_adapter.to_device(encoded.payload, reloaded, reload_cfg)
        ).velocity.cpu()
    torch.testing.assert_close(before, reloaded_output, atol=2e-5, rtol=2e-4)
    # Exercise BF16 autograd on the same real architecture in an independent policy.
    bf_cfg = cfg.model_copy(update={"dtype": "bfloat16", "gradient_checkpointing": True})
    reloaded = reloaded.to(dtype=torch.bfloat16)
    bf_model, _, _ = apply_policy(reloaded, bf_cfg)
    bf_model.train()
    bf_output = reload_adapter.forward(
        bf_model, reload_adapter.to_device(encoded.payload, bf_model, bf_cfg)
    )
    bf_output.loss.backward()
    assert any(
        p.grad is not None and p.grad.float().abs().sum() > 0
        for p in bf_model.parameters()
        if p.requires_grad
    )
    del bf_model, reloaded, reload_adapter, bf_output
    gc.collect()
    torch.cuda.empty_cache()
    decoder = NativeToken2Wav(root / "export", device="cuda")
    target, _ = sf.read(root / "data/audio-1.wav", dtype="float32")
    import torchaudio

    waveform16 = torchaudio.functional.resample(torch.from_numpy(target), 24000, 16000).numpy()
    codes = decoder.codec.encode(waveform16).tolist()
    (root / "codes.json").write_text(json.dumps(codes))
    waveform, rate = decoder.decode(codes, root / "data/audio-0.wav")
    sf.write(root / "decoded.wav", waveform, rate)
    assert rate == 24000 and len(waveform) > 0
    report = {
        "model": "MiniCPM-o 4.5 pretrained Token2Wav",
        "gpu": torch.cuda.get_device_name(),
        "training": result,
        "evaluation_loss": evaluation["loss"],
        "max_training_peak_bytes": max(m["cuda_peak_allocated_bytes"] for m in metrics),
        "fp32_training": True,
        "cuda_resume_parity": True,
        "tf32_for_parity": False,
        "bf16_backward": True,
        "checkpointing": True,
        "lora_merge_parity": True,
        "cpu_vs_cuda_merged_weight_max_abs": merge_max_abs,
        "native_export_reload_parity": True,
        "export": exported,
        "waveform_frames": len(waveform),
        "sample_rate": rate,
        "data": "synthetic tones; no speech quality claim",
        "vllm_serving_verified": False,
    }
    (root / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
