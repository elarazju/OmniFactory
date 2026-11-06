"""Pretrained PersonaPlex execution acceptance on tones or supplied duplex recordings.

Validates training/resume/native tensors/frozen assets/bounded streaming decode.
Conversation quality and real WebSocket deployment remain separate gates.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open

from omnifactory.acceptance import attach_gate, run_acceptance
from omnifactory.adapters.personaplex_model import inspect_native_weights
from omnifactory.artifacts import sha256, verify_export
from omnifactory.data.io import iter_samples
from omnifactory.datasets.duplex_demo import create_duplex_demo
from omnifactory.environment import capture_environment
from omnifactory.training.artifacts import write_json
from omnifactory.training.config import TrainConfig, read_config
from omnifactory.training.engine import load_checkpoint
from omnifactory.training.policy import merge_lora


def native_parity(checkpoint, export, output):
    expected, adapter, _, _ = load_checkpoint(checkpoint, for_export=True)
    expected = merge_lora(expected)
    _, count = inspect_native_weights(export)
    state = expected.state_dict()
    with safe_open(export / "model.safetensors", framework="pt", device="cpu") as native:
        assert set(native.keys()) == set(state)
        for name, value in state.items():
            actual = native.get_tensor(name)
            torch.testing.assert_close(
                value.detach().to(device="cpu", dtype=actual.dtype),
                actual,
                atol=0,
                rtol=0,
                msg=name,
            )
    frozen = {}
    for name in (
        "config.json",
        "tokenizer-e351c8d8-checkpoint125.safetensors",
        "tokenizer_spm_32k_3.model",
        "voices.tgz",
    ):
        digest = sha256(adapter.root / name)
        assert sha256(export / name) == digest, name
        frozen[name] = digest
    write_json(
        output,
        dict(
            status="passed",
            artifact_id=verify_export(export)["artifact_id"],
            tensor_count=count,
            merged_tensor_max_abs=0,
            frozen_files=frozen,
            scope="native strict tensor inventory and BF16 merge parity; frozen assets byte parity; not quality",
        ),
    )
    del expected, adapter, state
    gc.collect()
    torch.cuda.empty_cache()


def expected_session_frames(data):
    import soundfile as sf

    from omnifactory.adapters.personaplex import input_context
    from omnifactory.data.media import LocalMediaResolver
    from omnifactory.data.schema import inference_view

    resolver = LocalMediaResolver(data)
    expected = {}
    for sample in iter_samples(data / "validation.jsonl"):
        for prediction in sample.predictions:
            view = inference_view(sample, prediction.id, ("text", "audio"))
            block, _ = input_context(view)
            info = sf.info(resolver.resolve(view.assets[block.media.asset_id]).path)
            key = (sample.id, prediction.id)
            if key in expected or info.channels != 1 or not info.frames:
                raise ValueError("unique, nonempty mono validation sessions required")
            expected[key] = (info.frames * 25 + info.samplerate * 2 - 1) // (info.samplerate * 2)
    if not expected:
        raise ValueError("nonempty validation recordings required")
    return expected


def verify_session_coverage(predictions, expected):
    import soundfile as sf

    if len(predictions) != len(expected) or {
        (r["sample_id"], r["prediction_id"]) for r in predictions
    } != set(expected):
        raise ValueError("validation session identities/count differ")
    sessions = []
    for row in predictions:
        frames = expected[(row["sample_id"], row["prediction_id"])]
        if not (
            row["stop_reason"] == "input_session_end"
            and row["sample_rate"] == 24000
            and len(row["codec_tokens"]) == row["input_frames"] == frames
            and all(len(codes) == 8 for codes in row["codec_tokens"])
        ):
            raise ValueError("native session did not cover every original input frame")
        info = sf.info(row["audio"])
        if info.channels != 1 or info.samplerate != 24000 or info.frames != frames * 1920:
            raise ValueError("decoded waveform does not cover original input frames")
        sessions.append(
            dict(
                sample_id=row["sample_id"],
                prediction_id=row["prediction_id"],
                input_frames=frames,
                output_frames=len(row["codec_tokens"]),
                waveform_samples=info.frames,
                audio_sha256=sha256(Path(row["audio"])),
            )
        )
    return sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--data", type=Path, help="Imported duplex directory; omit for synthetic tones"
    )
    parser.add_argument(
        "--config", type=Path, help="Explicit recipe and budgets for supplied recordings"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA BF16 required")
    if args.steps < 2:
        parser.error("at least two steps required for resume parity")
    if args.data is not None and args.config is None:
        parser.error("supplied recordings require --config with explicit duration/frame budgets")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = args.data.resolve() if args.data else root / "data"
    if args.data is None:
        create_duplex_demo(data, args.model / "tokenizer_spm_32k_3.model")
    template = read_config(
        args.config or Path(__file__).resolve().parents[1] / "examples/train-personaplex.yaml"
    )
    expected_frames = expected_session_frames(data)
    config = TrainConfig.model_validate(
        {
            **template.model_dump(),
            "model": str(args.model.resolve()),
            "revision": None,
            "dataset": data / "train.jsonl",
            "dataset_root": data,
            "validation_dataset": data / "validation.jsonl",
            "validation_root": data,
            "duplex_text_alignment": data / "text-alignment.json",
            "output_dir": root / "run",
            "max_steps": args.steps,
            "save_steps": args.steps // 2,
            "eval_steps": args.steps // 2,
            "gradient_accumulation_steps": 1,
            "deterministic": True,
        }
    )
    write_json(root / "recipe.json", config.model_dump(mode="json"))
    write_json(
        root / "data-provenance.json",
        dict(
            origin="supplied_recordings" if args.data else "synthetic_tones",
            source_directory=str(data),
            files={p.name: sha256(p) for p in data.glob("*.json*")},
            expected_validation_frames=[
                dict(sample_id=s, prediction_id=p, frames=n)
                for (s, p), n in expected_frames.items()
            ],
            perceptual_quality_verified=False,
        ),
    )
    capture_environment(root / "environment")
    report = run_acceptance(config, root / "acceptance")
    predictions = [
        json.loads(row) for row in (root / "acceptance/predictions.jsonl").read_text().splitlines()
    ]
    coverage = verify_session_coverage(predictions, expected_frames)
    write_json(
        root / "session-coverage.json",
        dict(status="passed", artifact_id=report.artifact_id, sessions=coverage),
    )
    native_parity(
        Path(report.gates["train"].details["checkpoint"]),
        root / "acceptance/export",
        root / "native-reference.json",
    )
    attach_gate(
        root / "acceptance/acceptance.json", root / "native-reference.json", name="native_reference"
    )
    print(
        json.dumps(
            dict(
                status="passed",
                artifact_id=report.artifact_id,
                native_reference="passed",
                quality="not_run",
                realtime="not_run",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
