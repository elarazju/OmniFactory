"""Explicit source schemas converted to canonical records without guessing modalities."""

import json
from pathlib import Path

from omnifactory.data.io import atomic_writer
from omnifactory.data.schema import CanonicalSample
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import digest


def import_tts(source: Path, output: Path, *, name: str, split: str = "train"):
    """Utterance pairs with explicit speaker/group IDs; references are separate recordings."""
    if source.resolve() == output.resolve() or split not in ("train", "validation", "test"):
        raise OmniFactoryError("E_IMPORT", "use a distinct output and a valid split")
    import hashlib

    def file_sha256(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    count = 0
    seen = set()
    with source.open(encoding="utf-8") as stream, atomic_writer(output) as writer:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "text", "audio", "group_id"}
            if (
                not isinstance(row, dict)
                or not required <= row.keys()
                or row.keys() - required - {"reference_audio"}
            ):
                raise OmniFactoryError(
                    "E_IMPORT_FIELDS",
                    "expected id/text/audio/group_id and optional reference_audio",
                )
            if (
                any(not isinstance(v, str) or not v.strip() for v in row.values())
                or row["id"] in seen
            ):
                raise OmniFactoryError(
                    "E_IMPORT_FIELDS", "nonempty strings and unique IDs required"
                )
            seen.add(row["id"])
            assets = {}
            for key in ("audio", "reference_audio"):
                if key in row:
                    path = (source.parent / Path(row[key]).expanduser()).resolve()
                    if path == output.resolve():
                        raise OmniFactoryError(
                            "E_OUTPUT", "output must not overwrite an audio input"
                        )
                    assets[key] = {
                        "modality": "audio",
                        "uri": str(path),
                        "sha256": file_sha256(path),
                    }
            sample = CanonicalSample.model_validate(
                dict(
                    schema_version="0.1",
                    id=row["id"],
                    provenance=dict(
                        dataset=name, source_id=row["id"], group_id=row["group_id"], split=split
                    ),
                    assets=assets,
                    messages=[
                        dict(
                            id="user",
                            role="user",
                            content=[dict(id="utterance", type="text", text=row["text"])],
                        ),
                        dict(
                            id="assistant",
                            role="assistant",
                            content=[
                                dict(id="transcript", type="text", text=row["text"]),
                                dict(id="wave", type="audio", media=dict(asset_id="audio")),
                            ],
                        ),
                    ],
                    speech_pairs=[dict(text_block_id="transcript", audio_block_id="wave")],
                    predictions=[
                        dict(
                            id="p0",
                            target_message_id="assistant",
                            target_block_ids=["transcript", "wave"],
                            conditions=[
                                dict(kind="reference_audio", media=dict(asset_id="reference_audio"))
                            ]
                            if "reference_audio" in assets
                            else [],
                        )
                    ],
                )
            )
            writer.write(sample.model_dump_json() + "\n")
            count += 1
        if not count:
            raise OmniFactoryError("E_EMPTY_DATASET", "input contains no records")
    return {
        "records": count,
        "output": str(output.resolve()),
        "source_license": "retained from original dataset; not reassigned",
    }


def import_alpaca(source: Path, output: Path, *, name: str, split: str = "train"):
    if source.resolve() == output.resolve():
        raise OmniFactoryError("E_OUTPUT", "output must differ from source")
    if split not in ("train", "validation", "test"):
        raise OmniFactoryError("E_SPLIT", "expected train, validation, or test")
    count = 0
    with source.open(encoding="utf-8") as stream, atomic_writer(output) as writer:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) - {"instruction", "input", "output"}:
                raise OmniFactoryError(
                    "E_IMPORT_FIELDS",
                    "expected text-only instruction/input/output; use canonical JSONL for media",
                )
            if not all(
                isinstance(raw.get(k), str) and raw[k].strip() for k in ("instruction", "output")
            ):
                raise OmniFactoryError(
                    "E_IMPORT_FIELDS", "instruction and output must be nonempty strings"
                )
            extra = raw.get("input", "")
            if not isinstance(extra, str):
                raise OmniFactoryError("E_IMPORT_FIELDS", "input must be a string")
            prompt = raw["instruction"] + ("\n" + extra if extra else "")
            group = digest({"prompt": prompt})
            row = {
                "schema_version": "0.1",
                "id": f"{name}-{count:08d}",
                "provenance": {
                    "dataset": name,
                    "source_id": str(count),
                    "group_id": group,
                    "split": split,
                },
                "assets": {},
                "messages": [
                    {
                        "id": "user",
                        "role": "user",
                        "content": [{"id": "prompt", "type": "text", "text": prompt}],
                    },
                    {
                        "id": "assistant",
                        "role": "assistant",
                        "content": [{"id": "answer", "type": "text", "text": raw["output"]}],
                    },
                ],
                "predictions": [
                    {"id": "p0", "target_message_id": "assistant", "target_block_ids": ["answer"]}
                ],
            }
            writer.write(CanonicalSample.model_validate(row).model_dump_json() + "\n")
            count += 1
        if not count:
            raise OmniFactoryError("E_EMPTY_DATASET", "input contains no records")
    return {
        "records": count,
        "output": str(output.resolve()),
        "source_license": "retained from original dataset; not reassigned",
    }
