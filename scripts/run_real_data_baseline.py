"""Paired base / adapter / native-export quality and resource baseline.

Supports Qwen2.5-Omni ASR and VoxCPM2 TTS. Uses every validation prediction,
retains per-case outputs and fixes inference RNG per case across all three arms.
WER is descriptive; a completed run never implies quality improved.
"""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import soundfile as sf
import torch

from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.artifacts import sha256, verify_export
from omnifactory.data.io import iter_samples
from omnifactory.training.artifacts import write_json
from omnifactory.training.asr_metrics import corpus_wer
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import read_config
from omnifactory.training.control import deterministic_training
from omnifactory.training.efficiency import environment, summarize_steps
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.export import export_checkpoint
from omnifactory.training.prepare import prepare, source_fingerprint


def check_dataset(config):
    """Check source bytes, speaker groups and every target/reference asset."""
    splits = {}
    hashes, groups = {}, {}
    for split, path, root in (
        ("train", config.dataset, config.dataset_root or config.dataset.parent),
        (
            "validation",
            config.validation_dataset,
            config.validation_root or config.validation_dataset.parent,
        ),
    ):
        rows = list(iter_samples(path))
        if not rows or len({s.id for s in rows}) != len(rows):
            raise ValueError("empty split or duplicate sample IDs")
        groups[split], hashes[split] = set(), set()
        for sample in rows:
            if sample.provenance.split != split:
                raise ValueError("incorrect declared split")
            groups[split].add((sample.provenance.dataset, sample.provenance.group_id))
            if not sample.assets:
                raise ValueError("real-recording baseline requires audio assets")
            for asset in sample.assets.values():
                path = Path(asset.uri)
                path = path if path.is_absolute() else root / path
                if not asset.sha256 or sha256(path) != asset.sha256:
                    raise ValueError(f"media hash mismatch: {sample.id}")
                hashes[split].add(asset.sha256)
        splits[split] = rows
    if groups["train"] & groups["validation"] or hashes["train"] & hashes["validation"]:
        raise ValueError("train/validation speaker or waveform overlap")
    return splits


def case_seed(seed, sample_id, prediction_id):
    value = hashlib.sha256(f"{seed}:{sample_id}:{prediction_id}".encode()).digest()
    return int.from_bytes(value[:4], "little")


def reference_text(sample, prediction):
    target = next(m for m in sample.messages if m.id == prediction.target_message_id)
    return "\n".join(
        b.text for b in target.content if b.id in prediction.target_block_ids and b.type == "text"
    )


def generate(
    config,
    rows,
    output,
    label,
    *,
    max_new_tokens,
    max_audio_units,
    checkpoint=None,
    export_path=None,
    generation_seed=None,
):
    if label == "adapter":
        if checkpoint is None:
            result = json.loads((output / "train/result.json").read_text())
            checkpoint = Path(result["checkpoint"])
        model, adapter, cfg, _ = load_checkpoint(checkpoint)
    else:
        cfg = config.model_copy(update={"gradient_checkpointing": False, "feature_cache": None})
        if label == "export":
            export_path = export_path or output / "export"
            verify_export(export_path)
            cfg = cfg.model_copy(update={"model": str(export_path), "revision": None})
        model = load_model(cfg)
        adapter = adapter_class(cfg.adapter).from_config(cfg)
    model.eval()
    adapter.resolver.root = config.validation_root or config.validation_dataset.parent
    speech = cfg.adapter == "voxcpm2"
    decoder = adapter.speech_decoder(model) if speech else None
    records = []
    with (output / f"{label}-predictions.jsonl").open("x") as stream, torch.no_grad():
        for sample in rows:
            for prediction in sample.predictions:
                seed = case_seed(
                    cfg.seed if generation_seed is None else generation_seed,
                    sample.id,
                    prediction.id,
                )
                torch.manual_seed(seed)
                encoded = adapter.encode(
                    sample, prediction.id, contract_id="baseline", inference=True
                ).validate(inference=True)
                batch = adapter.to_device(encoded.payload, model, cfg)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
                record = dict(
                    sample_id=sample.id,
                    prediction_id=prediction.id,
                    seed=seed,
                    reference=reference_text(sample, prediction),
                )
                if speech:
                    generated = adapter.generate_speech(
                        model,
                        batch,
                        max_new_tokens=max_new_tokens,
                        max_codec_tokens=max_audio_units,
                    )
                    wave, rate = decoder(generated["audio_latents"])
                    path = output / f"{label}-{len(records):05d}.wav"
                    sf.write(path, wave, rate)
                    record.update(
                        audio=str(path),
                        audio_sha256=sha256(path),
                        audio_seconds=len(wave) / rate,
                        sample_rate=rate,
                        finished=bool(generated.get("speech_finished")),
                    )
                else:
                    ids = adapter.generate_ids(model, batch, max_new_tokens=max_new_tokens)
                    new = ids[0, batch["input_ids"].shape[1] :]
                    record.update(
                        prediction=adapter.processor.tokenizer.decode(
                            new, skip_special_tokens=True
                        ),
                        finished=bool((new == adapter.processor.tokenizer.eos_token_id).any()),
                    )
                torch.cuda.synchronize()
                record.update(
                    generation_seconds=perf_counter() - start,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                records.append(record)
                print(
                    json.dumps(dict(phase=label, case=len(records), finished=record["finished"])),
                    flush=True,
                )
    del model, adapter, decoder
    gc.collect()
    torch.cuda.empty_cache()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(config, samples, output, asr_model):
    arms = {
        label: read_rows(output / f"{label}-predictions.jsonl")
        for label in ("base", "adapter", "export")
    }
    expected = [(s.id, p.id, reference_text(s, p)) for s in samples for p in s.predictions]
    for records in arms.values():
        if [(r["sample_id"], r["prediction_id"], r["reference"]) for r in records] != expected:
            raise ValueError("evaluation identity/reference mismatch")
    scorer = "reference transcript / greedy Qwen2.5-Omni"
    if config.adapter == "voxcpm2":
        if asr_model is None:
            raise ValueError("TTS baseline requires --asr-model (pinned Qwen3-Omni)")
        from validate_infrastructure_gpu import transcribe

        paths = transcribe(asr_model, config, samples, arms, output)
        metrics = {
            label: corpus_wer(read_rows(path)) for label, path in zip(arms, paths, strict=True)
        }
        scorer = "Qwen3-Omni greedy ASR / " + source_fingerprint(str(asr_model), None)
    else:
        metrics = {label: corpus_wer(rows) for label, rows in arms.items()}
    steps = read_rows(output / "train/metrics.jsonl")
    measured = steps[2:]  # Explicitly exclude the first two optimizer steps from throughput.
    report = dict(
        format="omnifactory-real-data-baseline-v1",
        status="completed",
        artifact_id=verify_export(output / "export")["artifact_id"],
        metrics=metrics,
        scorer=scorer,
        cases=len(expected),
        termination={
            k: {"finished": sum(r["finished"] for r in v), "total": len(v)} for k, v in arms.items()
        },
        resources=summarize_steps(measured),
        all_step_peak_allocated_bytes=max(r["cuda_peak_allocated_bytes"] for r in steps),
        host_loading_seconds=sum(r["data_load_host_seconds"] for r in measured),
        host_collation_seconds=sum(r["collate_host_seconds"] for r in measured),
        excluded_warmup_steps=2,
        improvement_measured=metrics["adapter"]["wer"] < metrics["base"]["wer"],
        export_wer_delta=metrics["export"]["wer"] - metrics["adapter"]["wer"],
        evidence={
            str(p.relative_to(output)): sha256(p) for p in output.glob("*-predictions.jsonl")
        },
        scope="speaker-disjoint development re-partition; not official test WER or perceptual speech quality",
    )
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--asr-model", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--max-audio-units", type=int, default=256)
    p.add_argument(
        "--stage", choices=("all", "base", "train", "adapter", "export", "score"), default="all"
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = read_config(args.config)
    if cfg.adapter not in ("qwen2_5_omni", "voxcpm2") or cfg.quantization != "none":
        p.error("baseline requires unquantized Qwen2.5-Omni ASR or VoxCPM2 TTS")
    if cfg.validation_dataset is None or cfg.max_steps < 4 or cfg.device != "cuda":
        p.error("requires validation_dataset, >=4 steps and CUDA")
    if (cfg.adapter == "voxcpm2") != (cfg.task == "speech_sft"):
        p.error("use understanding_sft for Qwen and speech_sft for VoxCPM2")
    rows = check_dataset(cfg)
    output = args.output.resolve()
    cfg = cfg.model_copy(
        update={"output_dir": output / "train", "prepared_dataset": output / "prepared"}
    )
    identity = dict(
        config=cfg.model_dump(mode="json"),
        code=runtime_hashes(),
        source_fingerprint=source_fingerprint(cfg.model, cfg.revision),
        datasets={str(path): sha256(path) for path in (cfg.dataset, cfg.validation_dataset)},
        max_new_tokens=args.max_new_tokens,
        max_audio_units=args.max_audio_units,
        runner_sha256=sha256(Path(__file__)),
        scorer_script_sha256=sha256(Path(__file__).with_name("validate_infrastructure_gpu.py")),
        asr_model=str(args.asr_model.resolve()) if args.asr_model else None,
    )
    if args.stage == "all":
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "identity.json", identity)
        write_json(output / "environment.json", environment(cfg.device))
        # Fresh processes isolate allocator history and release model/codec memory.
        for stage in ("base", "train", "adapter", "export", "score"):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--stage", stage], check=True)
        return
    if json.loads((output / "identity.json").read_text()) != identity:
        raise ValueError("baseline source/config/data/evaluator changed; choose a fresh run")
    started = perf_counter()
    if args.stage == "train":
        prep_start = perf_counter()
        manifest = prepare(cfg, output / "prepared")
        write_json(
            output / "preparation.json",
            dict(
                seconds=perf_counter() - prep_start,
                predictions=len(manifest["entries"]),
                fingerprint=manifest["fingerprint"],
            ),
        )
        train(cfg, progress=lambda r: print(json.dumps(r), flush=True))
    elif args.stage == "score":
        score(cfg, rows["validation"], output, args.asr_model)
    else:
        if args.stage == "export":
            result = json.loads((output / "train/result.json").read_text())
            export_checkpoint(Path(result["checkpoint"]), output / "export", format="vllm-omni")
            gc.collect()
            torch.cuda.empty_cache()
        with deterministic_training(cfg):
            generate(
                cfg,
                rows["validation"],
                output,
                args.stage,
                max_new_tokens=args.max_new_tokens,
                max_audio_units=args.max_audio_units,
            )
    write_json(
        output / f"{args.stage}-stage.json",
        dict(seconds=perf_counter() - started, status="completed"),
    )


if __name__ == "__main__":
    main()
