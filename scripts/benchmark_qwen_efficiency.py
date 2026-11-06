"""Pretrained CUDA efficiency matrix over lengths and text/image/audio/video inputs."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from omnifactory.data.schema import CanonicalSample
from omnifactory.training.benchmark import benchmark
from omnifactory.training.calibration import calibrate
from omnifactory.training.config import TrainConfig


def create_fixture(root):
    root.mkdir(parents=True, exist_ok=False)
    media = root / "media"
    media.mkdir()
    rows = []
    # Interleave modalities and lengths for a source-order baseline. All benchmark
    # modes receive the same bytes and supervised predictions.
    for index in [0, 5, 1, 4, 2, 3]:
        repeats = [1, 8, 24, 48, 96, 160][index]
        duration = [0.25, 0.5, 1, 2, 4, 8][index]
        for kind in ("text", "image", "audio", "video"):
            identity = f"{kind}-{index}"
            assets, blocks = {}, []
            if kind != "text":
                suffix = {"image": "png", "audio": "wav", "video": "mp4"}[kind]
                path = media / f"{identity}.{suffix}"
                if kind == "image":
                    size = 56 * (index + 1)
                    Image.new("RGB", (size, size), (0, 0, 220)).save(path)
                elif kind == "audio":
                    t = np.arange(int(duration * 16000)) / 16000
                    sf.write(path, 0.1 * np.sin(2 * math.pi * 440 * t), 16000)
                else:
                    seconds = [0.5, 0.75, 1, 1.5, 2, 3][index]
                    size = 56 * (1 + index // 2)
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-nostdin",
                            "-f",
                            "lavfi",
                            "-i",
                            f"color=c=blue:s={size}x{size}:r=8:d={seconds}",
                            "-f",
                            "lavfi",
                            "-i",
                            f"sine=frequency=440:sample_rate=16000:duration={seconds}",
                            "-c:v",
                            "mpeg4",
                            "-c:a",
                            "aac",
                            "-shortest",
                            str(path),
                        ],
                        check=True,
                    )
                assets["input"] = {
                    "uri": f"media/{path.name}",
                    "modality": kind,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                if kind == "image":
                    blocks.append({"id": "media", "type": kind, "asset_id": "input"})
                else:
                    block = {"id": "media", "type": kind, "media": {"asset_id": "input"}}
                    if kind == "video":
                        block["audio_mode"] = "include"
                    blocks.append(block)
            blocks.append(
                {
                    "id": "question",
                    "type": "text",
                    "text": "Observe the supplied context. " * repeats + "Answer briefly.",
                }
            )
            rows.append(
                CanonicalSample.model_validate(
                    {
                        "schema_version": "0.1",
                        "id": identity,
                        "provenance": {
                            "dataset": "efficiency-fixture-v1",
                            "source_id": identity,
                            "group_id": identity,
                            "split": "train",
                        },
                        "assets": assets,
                        "messages": [
                            {"id": "user", "role": "user", "content": blocks},
                            {
                                "id": "answer",
                                "role": "assistant",
                                "content": [
                                    {"id": "text", "type": "text", "text": "A short description."}
                                ],
                            },
                        ],
                        "predictions": [
                            {
                                "id": "p0",
                                "target_message_id": "answer",
                                "target_block_ids": ["text"],
                            }
                        ],
                    }
                )
            )
    (root / "train.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batching", choices=("source", "modal"), required=True)
    parser.add_argument(
        "--dataset", type=Path, help="Reuse exactly the same fixture for comparisons"
    )
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--packing", choices=("none", "flash_attention_2"), default="none")
    parser.add_argument("--feature-cache", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This acceptance requires the GPU host")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset.resolve() if args.dataset else root / "data/train.jsonl"
    if args.dataset is None:
        create_fixture(dataset.parent)
    config = TrainConfig(
        model=str(args.model.resolve()),
        dataset=dataset,
        dataset_root=dataset.parent,
        output_dir=root / "unused",
        device="cuda",
        dtype="bfloat16",
        quantization=args.quantization,
        packing=args.packing,
        feature_cache=args.feature_cache.resolve() if args.feature_cache else None,
        gradient_checkpointing=True,
        max_length=4096,
        max_audio_seconds=10,
        max_image_pixels=1048576,
        video_fps=2,
        max_video_frames=10,
        max_video_seconds=4,
        gradient_accumulation_steps=1,
        components={"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.0001}},
        budget={
            "max_examples": 2,
            "max_llm_tokens": 8192,
            "max_vision_patches": 16384,
            "max_audio_frames": 5000,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 128 * 1024 * 1024,
        },
    )
    report = benchmark(
        config, root / "benchmark", batching=args.batching, warmup_epochs=1, repeats=3
    )
    print(json.dumps(report["summary"]), flush=True)
    if args.batching == "modal":
        result = calibrate(root / "benchmark", root / "memory-profile.json")
        print(
            json.dumps(
                {
                    "calibration_accepted": result["accepted"],
                    "heldout_cases": result["heldout_cases"],
                }
            ),
            flush=True,
        )
        # Verify the bounded artifact can drive the real preparation/planning path.
        from omnifactory.planning.batching import SampleCost
        from omnifactory.training.calibration import configured_planner
        from omnifactory.training.prepare import prepare

        bound = config.model_copy(update={"memory_profile": root / "memory-profile.json"})
        bound = bound.model_copy(
            update={
                "budget": bound.budget.model_copy(
                    update={"max_gpu_bytes": int(report["environment"]["gpu_total_bytes"] * 0.8)}
                )
            }
        )
        manifest = prepare(bound, root / "bounded-prepared")
        planner = configured_planner(bound, manifest["contract_id"], manifest["source_fingerprint"])
        plans = list(
            planner.plan(SampleCost.model_validate(e["cost"]) for e in manifest["entries"])
        )
        assert sum(len(p.example_ids) for p in plans) == len(manifest["entries"])
        (root / "bounded-plans.json").write_text(
            json.dumps([p.model_dump(mode="json") for p in plans], indent=2)
        )
        print("BOUNDED GPU PLANNING PASSED", flush=True)


if __name__ == "__main__":
    main()
