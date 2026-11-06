"""Make a speaker-disjoint pilot from pinned LibriSpeech dev-clean parquet.

This re-partitions the published development subset for a small adaptation
experiment. Its WER must not be reported as official LibriSpeech test-clean WER.
Install pyarrow in the data-preparation environment. No network is used here.
"""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import soundfile as sf

from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory, write_json

REPOSITORY = "openslr/librispeech_asr"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
FILENAME = "clean/validation/0000.parquet"
SOURCE_SHA256 = "c816e936fed8b83d5e3de28795bff1bcd3b46b3f1a5815bad4395d20186a9770"


def prepare(parquet, output, *, train_count=128, validation_count=32, max_seconds=8, seed=42):
    import pyarrow.parquet as pq

    if min(train_count, validation_count, max_seconds) <= 0:
        raise ValueError("counts and duration must be positive")
    with parquet.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if source_sha != SOURCE_SHA256:
        raise ValueError("parquet bytes do not match the pinned upstream LFS object")
    rows = pq.read_table(parquet).to_pylist()
    speakers = sorted({str(r["speaker_id"]) for r in rows})
    if len(speakers) < 2:
        raise ValueError("speaker-disjoint evaluation requires at least two speakers")

    def order(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    speakers.sort(key=order)
    validation_speakers = set(speakers[: max(1, len(speakers) // 5)])
    selected = {"train": [], "validation": []}
    limits = {"train": train_count, "validation": validation_count}
    skipped_long = 0
    for row in sorted(rows, key=lambda row: order(row["id"])):
        split = "validation" if str(row["speaker_id"]) in validation_speakers else "train"
        if len(selected[split]) >= limits[split]:
            continue
        if not re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+", row["id"]):
            raise ValueError("unexpected LibriSpeech utterance identifier")
        blob = row["audio"]["bytes"]
        info = sf.info(io.BytesIO(blob))
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError("expected original mono 16 kHz recordings")
        if info.duration > max_seconds:
            skipped_long += 1
            continue
        if not row["text"].strip() or not info.frames:
            raise ValueError("empty reference/audio")
        selected[split].append((row, info))
    if any(len(selected[s]) != limits[s] for s in limits):
        raise ValueError("not enough eligible recordings; lower counts or increase duration")
    with new_directory(output) as root:
        (root / "media").mkdir()
        manifest = []
        for split, entries in selected.items():
            samples = []
            for row, info in entries:
                name = row["id"]
                blob = row["audio"]["bytes"]
                digest = hashlib.sha256(blob).hexdigest()
                (root / f"media/{name}.flac").write_bytes(blob)
                sample = CanonicalSample.model_validate(
                    dict(
                        schema_version="0.1",
                        id=name,
                        provenance=dict(
                            dataset="librispeech-dev-clean-pilot",
                            source_id=name,
                            split=split,
                            group_id=f"speaker-{row['speaker_id']}",
                        ),
                        assets={
                            "recording": dict(
                                uri=f"media/{name}.flac",
                                modality="audio",
                                sha256=digest,
                                sample_rate_hz=16000,
                            )
                        },
                        messages=[
                            dict(
                                id="user",
                                role="user",
                                content=[
                                    dict(
                                        id="recording",
                                        type="audio",
                                        media=dict(asset_id="recording"),
                                    ),
                                    dict(
                                        id="instruction",
                                        type="text",
                                        text="Transcribe the speech in English. Return only the transcript.",
                                    ),
                                ],
                            ),
                            dict(
                                id="answer",
                                role="assistant",
                                content=[dict(id="transcript", type="text", text=row["text"])],
                            ),
                        ],
                        predictions=[
                            dict(
                                id="p0", target_message_id="answer", target_block_ids=["transcript"]
                            )
                        ],
                    )
                )
                samples.append(sample)
                manifest.append(
                    dict(
                        id=name,
                        split=split,
                        speaker_id=str(row["speaker_id"]),
                        sha256=digest,
                        seconds=info.duration,
                    )
                )
            (root / f"{split}.jsonl").write_text(
                "".join(s.model_dump_json() + "\n" for s in samples)
            )
        report = dict(
            repository=REPOSITORY,
            revision=REVISION,
            filename=FILENAME,
            source_parquet_sha256=source_sha,
            license="CC BY 4.0",
            source_url="https://www.openslr.org/12",
            seed=seed,
            synthetic=False,
            official_test_benchmark=False,
            max_seconds=max_seconds,
            counts=limits,
            skipped_long=skipped_long,
            examples=manifest,
        )
        write_json(root / "dataset_info.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-count", type=int, default=128)
    p.add_argument("--validation-count", type=int, default=32)
    p.add_argument("--max-seconds", type=float, default=8)
    args = p.parse_args()
    report = prepare(**vars(args))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}))


if __name__ == "__main__":
    main()
