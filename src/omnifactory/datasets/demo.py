"""Procedural image/audio/text fixtures, never a quality benchmark."""

import hashlib
import math
import struct
import wave
from pathlib import Path

from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory, write_json

CATALOG = (
    {
        "name": "image-demo",
        "license": "Apache-2.0",
        "source": "procedurally generated locally",
        "modalities": ["text"],
        "output_modalities": ["image"],
        "purpose": "native image flow execution checks; synthetic shapes, not quality",
        "command": "omnifactory datasets image-demo --output outputs/image-data",
    },
    {
        "name": "librispeech-dev-clean-pilot",
        "license": "CC BY 4.0",
        "source": "https://www.openslr.org/12",
        "modalities": ["audio", "text"],
        "output_modalities": ["text"],
        "purpose": "128/32 speaker-disjoint real ASR adaptation pilot; not official test-clean WER",
        "preparation": "scripts/prepare_librispeech.py (repository checkout; explicit source download)",
        "documentation": "docs/models.md#qwen3-omni",
    },
    {
        "name": "flow-demo",
        "license": "Apache-2.0",
        "source": "procedurally generated locally",
        "modalities": ["audio"],
        "output_modalities": ["audio"],
        "purpose": "native conditional flow training with separate reference and target tones; not speech quality",
        "command": "omnifactory datasets flow-demo --output outputs/flow-data",
    },
    {
        "name": "mimo-audio-demo",
        "license": "Apache-2.0",
        "source": "procedurally generated locally",
        "modalities": ["text", "audio"],
        "output_modalities": ["text", "audio"],
        "purpose": "multi-codebook native codec execution; random codec and tones, not speech quality",
        "command": "omnifactory datasets mimo-demo --output outputs/mimo-data",
    },
    {
        "name": "omni-demo",
        "license": "Apache-2.0",
        "source": "procedurally generated locally",
        "modalities": ["text", "image", "audio"],
        "purpose": "pipeline smoke tests, not model quality",
        "command": "omnifactory datasets demo --output outputs/demo",
    },
    {
        "name": "omni-av-demo",
        "license": "Apache-2.0",
        "source": "procedurally generated locally",
        "modalities": ["text", "image", "audio", "video"],
        "output_modalities": ["text", "audio"],
        "purpose": "native speech pipeline smoke tests; synthetic tone targets, not speech quality",
        "command": "omnifactory datasets omni-demo --output outputs/omni-data",
    },
    {
        "name": "qwen-native-speech",
        "source": "explicit native Qwen teacher synthesis from supplied transcripts",
        "modalities": ["text", "image", "audio", "video"],
        "output_modalities": ["text", "audio"],
        "purpose": "native codec-supervised training; synthesis quality depends on teacher and transcripts",
        "command": "omnifactory datasets qwen-speech --config examples/prepare-qwen-speech.yaml --output outputs/qwen-speech-data",
    },
    {
        "name": "contract-examples",
        "license": "Apache-2.0",
        "source": "examples/samples.jsonl",
        "modalities": ["text", "image", "audio", "video"],
        "purpose": "schema examples only; media files are not included",
    },
)


def create_demo(output: Path):
    # Pillow is part of the optional train extra; WAV generation needs only stdlib.
    from PIL import Image

    with new_directory(output) as root:
        (root / "media").mkdir()
        records = []
        for split, count in (("train", 12), ("validation", 6)):
            rows = []
            for index in range(count):
                kind = ("text", "image", "audio")[index % 3]
                identifier = f"{split}-{index:03d}"
                assets, blocks = {}, []
                if kind == "text":
                    number = index + (100 if split == "validation" else 0)
                    prompt, answer = f"Repeat the number {number}.", str(number)
                elif kind == "image":
                    color = ("red", "blue")[index % 2]
                    path = root / "media" / f"{identifier}.png"
                    shade = 200 + index + (20 if split == "validation" else 0)
                    rgb = (shade, 0, 0) if color == "red" else (0, 0, shade)
                    Image.new("RGB", (56, 56), rgb).save(path)
                    assets["input"] = {
                        "uri": f"media/{path.name}",
                        "modality": kind,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    blocks.append({"id": "media", "type": "image", "asset_id": "input"})
                    prompt, answer = "What color is this image?", color
                else:
                    path = root / "media" / f"{identifier}.wav"
                    frequency = (220 if index % 2 else 880) + index
                    phase = 0.2 if split == "validation" else 0.0
                    with wave.open(str(path), "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(16000)
                        audio.writeframes(
                            b"".join(
                                struct.pack(
                                    "<h",
                                    int(
                                        8000 * math.sin(2 * math.pi * frequency * i / 16000 + phase)
                                    ),
                                )
                                for i in range(4000)
                            )
                        )
                    assets["input"] = {
                        "uri": f"media/{path.name}",
                        "modality": kind,
                        "sample_rate_hz": 16000,
                        "duration_us": 250000,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    blocks.append({"id": "media", "type": "audio", "media": {"asset_id": "input"}})
                    prompt, answer = (
                        "Is the tone low or high?",
                        "low" if frequency < 500 else "high",
                    )
                blocks.append({"id": "question", "type": "text", "text": prompt})
                raw = {
                    "schema_version": "0.1",
                    "id": identifier,
                    "provenance": {
                        "dataset": "omni-demo",
                        "source_id": identifier,
                        "group_id": identifier,
                        "split": split,
                    },
                    "assets": assets,
                    "messages": [
                        {"id": "user", "role": "user", "content": blocks},
                        {
                            "id": "answer",
                            "role": "assistant",
                            "content": [{"id": "answer-text", "type": "text", "text": answer}],
                        },
                    ],
                    "predictions": [
                        {
                            "id": "p0",
                            "target_message_id": "answer",
                            "target_block_ids": ["answer-text"],
                        }
                    ],
                }
                sample = CanonicalSample.model_validate(raw)
                rows.append(sample.model_dump_json())
            (root / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
            records.extend(rows)
        write_json(
            root / "dataset_info.json",
            {
                **CATALOG[0],
                "train": 12,
                "validation": 6,
                "fingerprint": hashlib.sha256("\n".join(records).encode()).hexdigest(),
            },
        )
    return {"dataset": str(output.resolve()), "train": 12, "validation": 6, "synthetic": True}
