"""Audio-native synthetic fixtures with independent prompt/target WAV assets."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory


def create_mimo_demo(output: Path):
    with new_directory(output) as root:
        (root / "media").mkdir()
        for split in ("train", "validation"):
            rows = []
            for i in range(3):
                assets = {}
                for name in ("input", "target"):
                    identifier = f"{split}-{i}-{name}"
                    rate = 24000
                    length = 4800 + 960 * i
                    t = np.arange(length) / rate
                    frequency = 200 + 110 * i + (23 if split == "validation" else 0)
                    wav = (
                        0.1 * np.sin(t * frequency * 2 * np.pi + (0.7 if name == "target" else 0))
                    ).astype(np.float32)
                    path = root / "media" / f"{identifier}.wav"
                    sf.write(path, wav, rate)
                    assets[name] = {
                        "uri": f"media/{path.name}",
                        "modality": "audio",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "sample_rate_hz": rate,
                    }
                blocks = [{"id": "prompt", "type": "text", "text": "Say hello."}]
                if i:
                    blocks.insert(
                        0, {"id": "input", "type": "audio", "media": {"asset_id": "input"}}
                    )
                sample = CanonicalSample.model_validate(
                    {
                        "schema_version": "0.1",
                        "id": f"{split}-{i}",
                        "provenance": {
                            "dataset": "mimo-demo",
                            "source_id": f"{split}-{i}",
                            "group_id": f"{split}-{i}",
                            "split": split,
                        },
                        "assets": assets,
                        "messages": [
                            {"id": "user", "role": "user", "content": blocks},
                            {
                                "id": "assistant",
                                "role": "assistant",
                                "content": [
                                    {"id": "answer", "type": "text", "text": "Hello."},
                                    {
                                        "id": "speech",
                                        "type": "audio",
                                        "media": {"asset_id": "target"},
                                    },
                                ],
                            },
                        ],
                        "predictions": [
                            {
                                "id": "p0",
                                "target_message_id": "assistant",
                                "target_block_ids": ["answer", "speech"],
                            }
                        ],
                        "speech_pairs": [{"text_block_id": "answer", "audio_block_id": "speech"}],
                    }
                )
                rows.append(sample.model_dump_json())
            (root / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
    return {
        "output": str(output.resolve()),
        "train": 3,
        "validation": 3,
        "synthetic_tones": True,
        "quality_benchmark": False,
    }
