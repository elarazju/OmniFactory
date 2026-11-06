"""Build a frozen speaker-disjoint TTS confirmation subset from pinned test-clean.

Select speakers/utterances by hash, never by model outcomes. References are extra
recordings, disjoint from scored targets. This is not the official full ASR test.
"""

import argparse
import hashlib
import io
import re
from pathlib import Path

import soundfile as sf

from omnifactory.artifacts import sha256
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory, write_json

REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
SOURCE_SHA256 = "7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0"


def select_rows(rows, excluded, *, speakers=24, per_speaker=4, max_seconds=12, seed=20261027):
    if speakers < 2 or per_speaker < 2 or per_speaker % 2 or max_seconds <= 0:
        raise ValueError("requires >=2 speakers, a positive even target count and duration")
    used_ids, used_groups, used_media = set(), set(), set()
    for sample in excluded:
        if not sample.provenance.group_id:
            raise ValueError("exclusion manifests require speaker identities")
        used_ids.add(sample.id)
        used_groups.add(sample.provenance.group_id)
        used_media.update(a.sha256 for a in sample.assets.values())
    if None in used_media:
        raise ValueError("exclusion manifests require asset hashes")

    def order(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    groups = {}
    ids = set()
    for row in rows:
        name, speaker = row["id"], str(row["speaker_id"])
        if name in ids or not re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+", name):
            raise ValueError("duplicate/invalid test utterance")
        ids.add(name)
        if name in used_ids or f"speaker-{speaker}" in used_groups:
            raise ValueError("upstream test split overlaps training/development identities")
        blob = row["audio"]["bytes"]
        info = sf.info(io.BytesIO(blob))
        if info.samplerate != 16000 or info.channels != 1 or not info.frames:
            raise ValueError("expected nonempty mono 16 kHz audio")
        if not row["text"].strip() or row["text"] != row["text"].upper():
            raise ValueError("expected original uppercase transcripts")
        if info.duration <= max_seconds:
            groups.setdefault(speaker, []).append(row)
    eligible = sorted((k for k, v in groups.items() if len(v) >= per_speaker + 1), key=order)
    if len(eligible) < speakers:
        raise ValueError("insufficient speakers with distinct reference recordings")
    selected = []
    for speaker in eligible[:speakers]:
        ordered = sorted(groups[speaker], key=lambda r: order(r["id"]))
        targets, reference = ordered[:per_speaker], ordered[per_speaker]
        for row in [*targets, reference]:
            if hashlib.sha256(row["audio"]["bytes"]).hexdigest() in used_media:
                raise ValueError("test waveform overlaps training/development")
        selected.append((speaker, targets, reference))
    return selected


def prepare(parquet, output, exclude, *, speakers=24, per_speaker=4, max_seconds=12, seed=20261027):
    import pyarrow.parquet as pq

    if sha256(parquet) != SOURCE_SHA256:
        raise ValueError("test parquet does not match pinned upstream LFS hash")
    excluded = [s for path in exclude for s in iter_samples(path)]
    groups = select_rows(
        pq.read_table(parquet).to_pylist(),
        excluded,
        speakers=speakers,
        per_speaker=per_speaker,
        max_seconds=max_seconds,
        seed=seed,
    )
    with new_directory(output) as root:
        (root / "media").mkdir()
        samples, changes, reference_ids = [], [], []
        for speaker, targets, reference in groups:
            reference_ids.append(reference["id"])
            for row in [*targets, reference]:
                (root / "media" / (row["id"] + ".flac")).write_bytes(row["audio"]["bytes"])

            def asset(row):
                return dict(
                    uri=f"media/{row['id']}.flac",
                    modality="audio",
                    sample_rate_hz=16000,
                    sha256=hashlib.sha256(row["audio"]["bytes"]).hexdigest(),
                )

            for i, row in enumerate(targets):
                text = row["text"].capitalize()
                assets = {"recording": asset(row)}
                conditions = []
                if i % 2:
                    assets["reference"] = asset(reference)
                    conditions = [dict(kind="reference_audio", media=dict(asset_id="reference"))]
                samples.append(
                    CanonicalSample.model_validate(
                        dict(
                            schema_version="0.1",
                            id=row["id"],
                            provenance=dict(
                                dataset="librispeech-test-clean-tts-confirmation",
                                source_id=row["id"],
                                split="test",
                                group_id=f"speaker-{speaker}",
                            ),
                            assets=assets,
                            messages=[
                                dict(
                                    id="user",
                                    role="user",
                                    content=[dict(id="utterance", type="text", text=text)],
                                ),
                                dict(
                                    id="answer",
                                    role="assistant",
                                    content=[
                                        dict(id="transcript", type="text", text=text),
                                        dict(
                                            id="target",
                                            type="audio",
                                            media=dict(asset_id="recording"),
                                        ),
                                    ],
                                ),
                            ],
                            predictions=[
                                dict(
                                    id="p0",
                                    target_message_id="answer",
                                    target_block_ids=["transcript", "target"],
                                    conditions=conditions,
                                )
                            ],
                            speech_pairs=[
                                dict(text_block_id="transcript", audio_block_id="target")
                            ],
                        )
                    )
                )
                changes.append(dict(id=row["id"], original=row["text"], transformed=text))
        (root / "test.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in samples))
        write_json(root / "text-normalization.json", changes)
        write_json(
            root / "dataset_info.json",
            dict(
                repository="openslr/librispeech_asr",
                revision=REVISION,
                filename="clean/test/0000.parquet",
                source_parquet_sha256=SOURCE_SHA256,
                license="CC BY 4.0",
                source_url="https://www.openslr.org/12",
                selection_seed=seed,
                speakers=[s for s, _, _ in groups],
                per_speaker=per_speaker,
                count=len(samples),
                max_seconds=max_seconds,
                text_case="sentence",
                reference_ids=reference_ids,
                excluded_manifests={str(p.resolve()): sha256(p) for p in exclude},
                independent_of_training_and_development=True,
                official_full_test_benchmark=False,
                manifest_sha256=sha256(root / "test.jsonl"),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--speakers", type=int, default=24)
    parser.add_argument("--per-speaker", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=12)
    parser.add_argument("--seed", type=int, default=20261027)
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
