"""Deterministic offline audio/video fixtures; not a natural-speech quality corpus."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.demo import create_demo
from omnifactory.training.artifacts import new_directory, write_json


def create_omni_demo(output: Path):
    with new_directory(output) as root:
        create_demo(root / "seed")
        video = root / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=56x56:r=8:d=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=1",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                "-shortest",
                str(video),
            ],
            check=True,
        )
        records = {}
        for split in ("train", "validation"):
            originals = [
                json.loads(line) for line in (root / f"seed/{split}.jsonl").read_text().splitlines()
            ]
            rows = []
            for i in range(4):
                raw = deepcopy(originals[i % 3])
                raw["id"] = f"{split}-omni-{i}"
                raw["provenance"]["group_id"] = raw["id"]
                raw["provenance"]["source_id"] = raw["id"]
                for asset in raw["assets"].values():
                    asset["uri"] = "seed/" + asset["uri"]
                if i == 3:
                    raw["assets"] = {
                        "video": {
                            "uri": "clip.mp4",
                            "modality": "video",
                            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                        }
                    }
                    raw["messages"][0]["content"] = [
                        {
                            "id": "video",
                            "type": "video",
                            "media": {"asset_id": "video", "end_us": 1000000},
                            "audio_mode": "include",
                        },
                        {"id": "question", "type": "text", "text": "Describe the color and sound."},
                    ]
                raw["messages"][-1]["content"][0]["text"] = "A tone."
                raw["assets"]["target-wave"] = {
                    "uri": f"seed/media/{split}-002.wav",
                    "modality": "audio",
                    "sample_rate_hz": 16000,
                }
                raw["messages"][-1]["content"].append(
                    {"id": "answer-audio", "type": "audio", "media": {"asset_id": "target-wave"}}
                )
                raw["predictions"][0]["target_block_ids"].append("answer-audio")
                raw["speech_pairs"] = [
                    {
                        "text_block_id": raw["messages"][-1]["content"][0]["id"],
                        "audio_block_id": "answer-audio",
                    }
                ]
                rows.append(CanonicalSample.model_validate(raw))
            (root / f"{split}.jsonl").write_text(
                "".join(row.model_dump_json() + "\n" for row in rows)
            )
            records[split] = len(rows)
        write_json(
            root / "dataset_info.json",
            {
                "name": "omni-av-demo",
                "records": records,
                "synthetic_tone_targets": True,
                "speech_quality_benchmark": False,
                "description": "Audio/video execution fixtures; replace tones with aligned speech for useful SFT.",
            },
        )
    return {"dataset": str(output.resolve()), "records": records}
