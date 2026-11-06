"""Real VoxCPM2 training pilot plus shared infrastructure and paired ASR evaluation."""

import argparse
import gc
import json
import math
from pathlib import Path

import soundfile as sf
import torch

from omnifactory.acceptance import Gate, attach_gate, run_acceptance
from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.artifacts import sha256
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.environment import capture_environment
from omnifactory.quality import compare_transcripts
from omnifactory.training.config import TrainConfig
from omnifactory.training.evaluation import infer_export
from omnifactory.training.prepare import source_fingerprint


def release():
    gc.collect()
    torch.cuda.empty_cache()


def generate_base(cfg, samples, output):
    model = load_model(cfg).eval()
    adapter = adapter_class(cfg.adapter).from_config(cfg)
    decoder = adapter.speech_decoder(model)
    records = []
    torch.manual_seed(cfg.seed)
    with torch.no_grad():
        for i, sample in enumerate(samples):
            encoded = adapter.encode(sample, "p0", contract_id="quality", inference=True)
            result = adapter.generate_speech(
                model,
                adapter.to_device(encoded.payload, model, cfg),
                max_new_tokens=32,
                max_codec_tokens=128,
            )
            wave, rate = decoder(result["audio_latents"])
            path = output / f"base-{i:03d}.wav"
            sf.write(path, wave, rate)
            records.append(
                dict(
                    sample_id=sample.id,
                    audio=str(path.resolve()),
                    speech_finished=result.get("speech_finished"),
                )
            )
    (output / "base-generation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    del model, adapter, decoder
    release()
    return records


def speech_segments(path, max_seconds):
    """Cover every source frame, balancing slices to avoid a tiny final tail."""
    info = sf.info(path)
    if not info.frames or max_seconds <= 0:
        raise ValueError("empty audio or invalid scoring duration")
    count = math.ceil(info.duration / max_seconds)
    segments = []
    for i in range(count):
        start, end = info.frames * i // count, info.frames * (i + 1) // count
        media = dict(asset_id="recording")
        if count > 1:
            media.update(
                start_us=round(start * 1e6 / info.samplerate),
                end_us=round(end * 1e6 / info.samplerate),
            )
            if any(
                round(media[k] * info.samplerate / 1e6) != v
                for k, v in (("start_us", start), ("end_us", end))
            ):
                raise ValueError("segment timestamps do not preserve source frames")
        segments.append(dict(start_frame=start, end_frame=end, media=media))
    return segments


def transcribe(asr_model, cfg, samples, sets, output):
    asr_cfg = cfg.model_copy(
        update=dict(
            model=str(asr_model.resolve()),
            adapter="qwen3_omni",
            task="understanding_sft",
            max_length=4096,
            max_audio_seconds=30,
            max_codec_tokens=128,
            speech_speaker="Ethan",
        )
    )
    model = load_model(asr_cfg).eval()
    adapter = adapter_class(asr_cfg.adapter).from_config(asr_cfg)
    references = {
        s.id: "".join(b.text for b in s.messages[0].content if b.type == "text") for s in samples
    }
    paths = []
    with torch.no_grad():
        for label, records in sets.items():
            target = output / f"{label}-transcripts.jsonl"
            with target.open("x") as stream:
                for record in records:
                    path = Path(record["audio"])
                    digest = sha256(path)
                    segments = speech_segments(path, adapter.max_audio_seconds)
                    for segment in segments:
                        sample = CanonicalSample.model_validate(
                            dict(
                                schema_version="0.1",
                                id=record["sample_id"],
                                provenance=dict(
                                    dataset="generated-speech-pilot",
                                    source_id=record["sample_id"],
                                    group_id=record["sample_id"],
                                    split="validation",
                                ),
                                assets=dict(
                                    recording=dict(
                                        uri=str(path.resolve()), modality="audio", sha256=digest
                                    )
                                ),
                                messages=[
                                    dict(
                                        id="user",
                                        role="user",
                                        content=[
                                            dict(id="audio", type="audio", media=segment["media"]),
                                            dict(
                                                id="instruction",
                                                type="text",
                                                text="Transcribe the English speech exactly. Return only the transcription.",
                                            ),
                                        ],
                                    ),
                                    dict(
                                        id="answer",
                                        role="assistant",
                                        content=[
                                            dict(
                                                id="text",
                                                type="text",
                                                text=references[record["sample_id"]],
                                            )
                                        ],
                                    ),
                                ],
                                predictions=[
                                    dict(
                                        id="p0",
                                        target_message_id="answer",
                                        target_block_ids=["text"],
                                    )
                                ],
                            )
                        )
                        encoded = adapter.encode(
                            sample, "p0", contract_id="quality-asr", inference=True
                        )
                        segment["prediction"] = adapter.generate(
                            model,
                            adapter.to_device(encoded.payload, model, asr_cfg),
                            max_new_tokens=128,
                        )[0]
                    row = dict(
                        sample_id=record["sample_id"],
                        reference=references[record["sample_id"]],
                        prediction=" ".join(s["prediction"].strip() for s in segments),
                        audio=str(path.resolve()),
                        audio_sha256=digest,
                        segments=segments,
                        segmentation="balanced-nonoverlap-v1",
                    )
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(json.dumps(dict(phase=label, **row)), flush=True)
            paths.append(target)
    del model, adapter
    release()
    return paths


def reuse_transcripts(cache, scorer, samples, sets, output):
    """Reuse actual ASR only when scorer, every case, reference and WAV bytes match."""
    if cache is None or not (cache / "quality.json").is_file():
        return None
    if json.loads((cache / "quality.json").read_text())["scorer"] != scorer:
        return None
    from importlib.metadata import version

    from omnifactory.training.benchmark import runtime_hashes

    provenance = cache / "acceptance/tested-code.json"
    environment = cache / "environment/environment.json"
    if not provenance.is_file() or not environment.is_file():
        return None
    previous_code = json.loads(provenance.read_text())
    current_code = runtime_hashes()
    relevant = {
        name
        for name in current_code
        if name.startswith(("data/", "adapters/qwen", "adapters/base", "adapters/speech"))
    }
    if any(previous_code.get(name) != current_code[name] for name in relevant):
        return None
    packages = json.loads(environment.read_text())["packages"]
    if any(
        packages.get(name) != version(name)
        for name in ("torch", "transformers", "tokenizers", "librosa")
    ):
        return None
    references = {
        s.id: "".join(b.text for b in s.messages[0].content if b.type == "text") for s in samples
    }
    pending = []
    for label, records in sets.items():
        source = cache / f"{label}-transcripts.jsonl"
        if not source.is_file():
            return None
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        keyed = {row["sample_id"]: row for row in rows}
        if len(keyed) != len(rows) or set(keyed) != {r["sample_id"] for r in records}:
            return None
        updated = []
        for record in records:
            old = keyed[record["sample_id"]]
            if old["reference"] != references[record["sample_id"]] or old["audio_sha256"] != sha256(
                record["audio"]
            ):
                return None
            updated.append({**old, "audio": record["audio"]})
        pending.append((output / f"{label}-transcripts.jsonl", updated, source))
    for path, rows, _ in pending:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "asr-cache.json").write_text(
        json.dumps(
            {
                "scorer": scorer,
                "source_reports": {str(source): sha256(source) for _, _, source in pending},
                "matched_waveforms": sum(len(rows) for _, rows, _ in pending),
            },
            indent=2,
        )
    )
    return [path for path, _, _ in pending]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--asr-model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--transcript-cache", type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture_environment(output / "environment")
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        adapter="voxcpm2",
        task="speech_sft",
        dataset=args.data.resolve() / "train.jsonl",
        validation_dataset=args.data.resolve() / "validation.jsonl",
        output_dir=output / "unused",
        device="cuda",
        dtype="bfloat16",
        max_steps=args.steps,
        save_steps=16,
        eval_steps=16,
        max_length=4096,
        max_audio_seconds=30,
        gradient_accumulation_steps=4,
        lr_scheduler="cosine",
        warmup_steps=4,
        scheduler_steps=args.steps,
        weight_decay=0.01,
        components={
            "language_model": dict(mode="lora", rank=4, alpha=8, lr=1e-4),
            "residual_model": dict(mode="lora", rank=4, alpha=8, lr=1e-4),
            "audio_decoder": dict(mode="lora", rank=4, alpha=8, lr=1e-4),
            "speech_projection": dict(mode="full", lr=1e-5),
            "stop_predictor": dict(mode="full", lr=1e-5),
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=4096,
            max_vision_patches=1,
            max_audio_frames=10000,
            max_codec_frames=10000,
            max_latent_elements=1000000,
            max_decoded_media_bytes=64000000,
        ),
    )
    samples = list(iter_samples(cfg.validation_dataset))[:8]
    base = generate_base(cfg, samples, output)
    report = run_acceptance(cfg, output / "acceptance", generate=False)
    subset = output / "quality-data.jsonl"
    subset.write_text("".join(s.model_dump_json() + "\n" for s in samples))
    export = output / "acceptance/export"
    generated = infer_export(
        export,
        subset,
        output=output / "tuned-generation.jsonl",
        generate_audio=True,
        max_audio_units=128,
        device="cuda",
        dtype="bfloat16",
    )
    tuned = [
        json.loads(line) for line in (output / "tuned-generation.jsonl").read_text().splitlines()
    ]
    report.gates["inference"] = Gate(
        status="passed",
        details=generated,
        evidence={"tuned-generation.jsonl": sha256(output / "tuned-generation.jsonl")},
    )
    report.save(output / "acceptance/acceptance.json")
    (output / "termination.json").write_text(
        json.dumps(
            {
                label: [
                    {"sample_id": row["sample_id"], "speech_finished": row.get("speech_finished")}
                    for row in rows
                ]
                for label, rows in {"base": base, "tuned": tuned}.items()
            },
            indent=2,
        )
    )
    scorer = "Qwen3-Omni greedy ASR / " + source_fingerprint(str(args.asr_model), None)
    sets = {"base": base, "tuned": tuned}
    cached = reuse_transcripts(args.transcript_cache, scorer, samples, sets, output)
    before, after = cached or transcribe(args.asr_model, cfg, samples, sets, output)
    result = compare_transcripts(
        before, after, output / "quality.json", artifact_id=report.artifact_id, scorer=scorer
    )
    attach_gate(output / "acceptance/acceptance.json", output / "quality.json", name="quality")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
