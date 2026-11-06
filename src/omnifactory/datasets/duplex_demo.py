"""Synthetic paired-tone sessions with explicit token timing; not speech data."""

import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece
import soundfile as sf

from omnifactory.artifacts import sha256
from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory


def create_duplex_demo(output: Path, tokenizer: Path):
    sp = sentencepiece.SentencePieceProcessor(str(tokenizer))
    transcript = "Hello there."
    tokens = sp.encode(transcript)
    frames = max(24, len(tokens) + 12)
    rate, length = 24000, frames * 1920
    with new_directory(output) as root:
        rows, alignment = [], []
        for i in range(6):
            assets = {}
            for name, frequency, start in (
                ("user", 170 + i * 30, 0),
                ("agent", 310 + i * 30, 5 * 1920),
            ):
                time = np.arange(length) / rate
                wave = (0.08 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
                wave[:start] = 0
                wave[-1920:] = 0
                path = root / f"{name}-{i}.wav"
                sf.write(path, wave, rate, subtype="PCM_16")
                assets[name] = dict(
                    uri=path.name, modality="audio", sha256=sha256(path), sample_rate_hz=rate
                )
            row = CanonicalSample.model_validate(
                dict(
                    schema_version="0.1",
                    id=f"duplex-{i}",
                    provenance=dict(
                        dataset="omnifactory-synthetic-duplex-fixture",
                        source_id=str(i),
                        group_id=str(i),
                        split="train" if i < 4 else "validation",
                    ),
                    assets=assets,
                    messages=[
                        dict(
                            id="system",
                            role="system",
                            content=[
                                dict(id="persona", type="text", text="You are a helpful assistant.")
                            ],
                        ),
                        dict(
                            id="user",
                            role="user",
                            content=[
                                dict(
                                    id="user_audio",
                                    type="audio",
                                    media=dict(asset_id="user"),
                                    placement=dict(timeline_id="session", offset_us=0),
                                )
                            ],
                        ),
                        dict(
                            id="assistant",
                            role="assistant",
                            content=[
                                dict(
                                    id="agent_audio",
                                    type="audio",
                                    media=dict(asset_id="agent"),
                                    placement=dict(timeline_id="session", offset_us=0),
                                ),
                                dict(id="transcript", type="text", text=transcript),
                            ],
                        ),
                    ],
                    predictions=[
                        dict(
                            id="p0",
                            target_message_id="assistant",
                            target_block_ids=["agent_audio", "transcript"],
                        )
                    ],
                    speech_pairs=[dict(text_block_id="transcript", audio_block_id="agent_audio")],
                )
            )
            rows.append(row)
            alignment.append(
                dict(
                    sample_id=row.id,
                    prediction_id="p0",
                    tokenizer_sha256=sha256(tokenizer),
                    audio_sha256=assets["agent"]["sha256"],
                    text_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
                    token_ids=tokens,
                    frames=list(range(6, 6 + len(tokens))),
                )
            )
        for split in ("train", "validation"):
            (root / f"{split}.jsonl").write_text(
                "".join(r.model_dump_json() + "\n" for r in rows if r.provenance.split == split)
            )
        (root / "text-alignment.json").write_text(json.dumps(alignment, indent=2))
    return dict(
        dataset=str(output.resolve()),
        synthetic_tones=True,
        transcript_is_synthetic=True,
        quality_benchmark=False,
        alignment=str((output / "text-alignment.json").resolve()),
    )
