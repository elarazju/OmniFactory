"""Local conditional-flow fixtures: separate reference and target audio, synthetic tones."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory


def create_flow_demo(output: Path):
    with new_directory(output) as root:
        for i, duration in enumerate((1.0, 0.8, 0.6, 0.7)):
            t = np.arange(round(24000 * duration)) / 24000
            signal = (0.1 * np.sin(2 * np.pi * (220 + i * 60) * t)).astype(np.float32)
            sf.write(root / f"audio-{i}.wav", signal, 24000)
        records = []
        for i in range(1, 4):
            split = "validation" if i == 3 else "train"
            record = CanonicalSample.model_validate(
                {
                    "schema_version": "0.1",
                    "id": f"flow-{i}",
                    "provenance": {
                        "dataset": "flow-demo",
                        "source_id": str(i),
                        "group_id": str(i),
                        "split": split,
                    },
                    "assets": {
                        "ref": {"uri": "audio-0.wav", "modality": "audio"},
                        "target": {"uri": f"audio-{i}.wav", "modality": "audio"},
                    },
                    "messages": [
                        {
                            "id": "user",
                            "role": "user",
                            "content": [
                                {"id": "request", "type": "text", "text": "Synthesize audio."}
                            ],
                        },
                        {
                            "id": "assistant",
                            "role": "assistant",
                            "content": [
                                {"id": "audio", "type": "audio", "media": {"asset_id": "target"}}
                            ],
                        },
                    ],
                    "predictions": [
                        {
                            "id": "p0",
                            "target_message_id": "assistant",
                            "target_block_ids": ["audio"],
                            "conditions": [
                                {"kind": "reference_audio", "media": {"asset_id": "ref"}}
                            ],
                        }
                    ],
                }
            )
            records.append(record)
        for split in ("train", "validation"):
            (root / f"{split}.jsonl").write_text(
                "".join(
                    json.dumps(r.model_dump(mode="json")) + "\n"
                    for r in records
                    if r.provenance.split == split
                )
            )
    return {"dataset": str(output.resolve()), "synthetic_audio": True, "quality_benchmark": False}
