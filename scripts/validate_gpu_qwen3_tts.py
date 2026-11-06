"""Qwen3-TTS single-speaker real-recording acceptance with chapter-held-out data.

The selected LibriSpeech speaker is shared across splits; chapters, recordings
and reference/targets are disjoint. This is execution validation, not a quality
benchmark. Run on CUDA with a pinned 12 Hz 1.7B Base snapshot.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from omnifactory.acceptance import AcceptanceReport, Gate, attach_gate, run_acceptance
from omnifactory.artifacts import sha256, verify_export
from omnifactory.datasets.importers import import_tts
from omnifactory.environment import capture_environment
from omnifactory.training.artifacts import write_json
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint
from omnifactory.training.policy import merge_lora


def prepare_data(corpus, output, train_count=8, validation_count=4):
    import io
    import re

    import pyarrow.parquet as pq

    parquet = corpus if corpus.is_file() else corpus / "clean/validation/0000.parquet"
    expected = "c816e936fed8b83d5e3de28795bff1bcd3b46b3f1a5815bad4395d20186a9770"
    if sha256(parquet) != expected:
        raise ValueError("LibriSpeech bytes do not match the pinned dev-clean snapshot")
    speakers = {}
    selection = None
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if not re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+", row["id"]):
                raise ValueError("unexpected utterance identifier")
            info = sf.info(io.BytesIO(row["audio"]["bytes"]))
            if not 1 <= info.duration <= 8 or info.samplerate != 16000 or info.channels != 1:
                continue
            speaker, chapter, _ = row["id"].split("-")
            chapters = speakers.setdefault(speaker, {})
            candidates = chapters.setdefault(chapter, [])
            if len(candidates) < max(train_count + 1, validation_count):
                candidates.append(row)
            for train_chapter, training in chapters.items():
                for val_chapter, validation in chapters.items():
                    if (
                        train_chapter != val_chapter
                        and len(training) >= train_count + 1
                        and len(validation) >= validation_count
                    ):
                        selection = (
                            speaker,
                            training[:train_count],
                            validation[:validation_count],
                            training[train_count],
                        )
                        break
                if selection:
                    break
            if selection:
                break
        if selection:
            break
    if selection is None:
        raise ValueError("need one speaker with enough short recordings in two chapters")
    speaker, training, validation, reference = selection
    output.mkdir(parents=True, exist_ok=False)
    (output / "media").mkdir()
    hashes = set()
    for row in training + validation + [reference]:
        path = output / "media" / (row["id"] + ".flac")
        path.write_bytes(row["audio"]["bytes"])
        digest = sha256(path)
        if digest in hashes:
            raise ValueError("reference/targets contain duplicate audio")
        hashes.add(digest)
    for split, records in (("train", training), ("validation", validation)):
        raw = output / f"{split}-source.jsonl"
        raw.write_text(
            "".join(
                json.dumps(
                    dict(
                        id=row["id"],
                        text=row["text"],
                        audio=str(output / "media" / (row["id"] + ".flac")),
                        group_id="-".join(row["id"].split("-")[:2]),
                    )
                )
                + "\n"
                for row in records
            )
        )
        import_tts(
            raw, output / f"{split}.jsonl", name="librispeech-dev-clean-single-speaker", split=split
        )
    reference = output / "media" / (reference["id"] + ".flac")
    write_json(
        output / "dataset_info.json",
        dict(
            corpus="LibriSpeech dev-clean",
            repository="openslr/librispeech_asr",
            revision="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
            source_sha256=expected,
            license="CC BY 4.0",
            speaker=speaker,
            reference=str(reference),
            reference_sha256=sha256(reference),
            split_unit="chapter",
            speaker_disjoint=False,
            reference_disjoint_from_targets=True,
            train_records=len(training),
            validation_records=len(validation),
            quality_benchmark=False,
        ),
    )
    return reference


def release():
    gc.collect()
    torch.cuda.empty_cache()


def native_export_parity(checkpoint, export, output):
    """Compare merged native tensor values and an upstream loader independently of export."""
    from omnifactory._vendor.qwen3_tts.core.models import Qwen3TTSForConditionalGeneration

    model, adapter, cfg, _ = load_checkpoint(checkpoint, for_export=True)
    model = merge_lora(model)
    expected_speaker = adapter.speaker_embedding()[0]
    # The upstream loader also loads its bundled native codec; no adapter loader.
    native = Qwen3TTSForConditionalGeneration.from_pretrained(
        export, local_files_only=True, torch_dtype=torch.float32
    )
    expected = model.state_dict()
    actual = native.state_dict()
    assert set(actual) == {k for k in expected if not k.startswith("speaker_encoder.")}
    for key, tensor in actual.items():
        before = expected[key]
        if key == "talker.model.codec_embedding.weight":
            before = before.clone()
            before[3000] = expected_speaker
        torch.testing.assert_close(tensor, before, atol=0, rtol=0)
    for path in (adapter.root / "speech_tokenizer").rglob("*"):
        if path.is_file() and ".cache" not in path.parts:
            assert sha256(path) == sha256(
                export / "speech_tokenizer" / path.relative_to(adapter.root / "speech_tokenizer")
            )
    result = dict(
        status="passed",
        artifact_id=verify_export(export)["artifact_id"],
        tensor_count=len(actual),
        merged_tensor_max_abs=0,
        speaker_slot=3000,
        native_codec_loaded=native.speech_tokenizer is not None,
        scope="native tensor mapping, frozen codec bytes, upstream loader; not speech quality",
    )
    write_json(output, result)
    del model, adapter, native, expected, actual
    release()
    return result


def validate_waveforms(root):
    """Persist EOS/waveform failures as an inference failure, not a green command run."""
    path = root / "acceptance/acceptance.json"
    report = AcceptanceReport.model_validate_json(path.read_text())
    result = dict(
        status="passed",
        records=0,
        natural_eos=True,
        quality_benchmark=False,
        artifact_id=report.artifact_id,
    )
    try:
        generated = [
            json.loads(line)
            for line in (root / "acceptance/predictions.jsonl").read_text().splitlines()
        ]
        if len(generated) != 4:
            raise ValueError("expected four held-out predictions")
        for record in generated:
            if not record["speech_finished"]:
                raise ValueError(f"{record['sample_id']}: generation hit its bound before EOS")
            wave, rate = sf.read(record["audio"])
            if not (
                rate == 24000
                and wave.size
                and np.isfinite(wave).all()
                and np.max(np.abs(wave)) > 1e-6
            ):
                raise ValueError(f"{record['sample_id']}: invalid or silent waveform")
            result["records"] += 1
    except Exception as exc:
        result.update(status="failed", natural_eos=False, error=str(exc))
    evidence = root / "waveform-check.json"
    write_json(evidence, result)
    previous = report.gates["inference"]
    report.gates["inference"] = Gate(
        status=result["status"],
        details={**previous.details, "waveforms": result},
        evidence={**previous.evidence, str(evidence): sha256(evidence)},
    )
    report.save(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA BF16 required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = prepare_data(args.corpus.resolve(), root / "data")
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        adapter="qwen3_tts",
        task="speech_sft",
        dataset=root / "data/train.jsonl",
        validation_dataset=root / "data/validation.jsonl",
        output_dir=root / "run",
        speech_reference=reference,
        speech_reference_sha256=sha256(reference),
        speech_speaker="librispeaker",
        device="cuda",
        dtype="bfloat16",
        attention="sdpa",
        gradient_checkpointing=True,
        max_steps=args.steps,
        save_steps=max(1, args.steps // 2),
        eval_steps=max(1, args.steps // 2),
        gradient_accumulation_steps=2,
        max_length=1024,
        max_audio_seconds=8,
        max_codec_tokens=256,
        components={
            k: dict(mode="lora", rank=8, alpha=16, lr=1e-4)
            for k in ("speech_generator", "code_predictor")
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=1024,
            max_codec_frames=128,
            max_audio_frames=1024,
            max_vision_patches=1,
            max_decoded_media_bytes=16777216,
        ),
    )
    write_json(root / "recipe.json", cfg.model_dump(mode="json"))
    capture_environment(root / "environment")
    report = run_acceptance(cfg, root / "acceptance")
    waveform = validate_waveforms(root)
    native_export_parity(
        Path(report.gates["train"].details["checkpoint"]),
        root / "acceptance/export",
        root / "native-reference.json",
    )
    attach_gate(
        root / "acceptance/acceptance.json", root / "native-reference.json", name="native_reference"
    )
    print((root / "acceptance/acceptance.json").read_text(), flush=True)
    if waveform["status"] != "passed":
        raise RuntimeError(waveform["error"])


if __name__ == "__main__":
    main()
