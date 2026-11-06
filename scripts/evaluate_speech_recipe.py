"""Evaluate frozen VoxCPM2 checkpoints with independent inference RNG and data.

Generate arms separately, then score all saved waveforms in one scorer process.
This never trains on the evaluation manifest or drops unfinished/long outputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from run_real_data_baseline import generate, read_rows, reference_text
from validate_infrastructure_gpu import transcribe

from omnifactory.artifacts import sha256, verify_export
from omnifactory.data.io import iter_samples
from omnifactory.training.artifacts import write_json
from omnifactory.training.asr_metrics import corpus_wer
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.control import deterministic_training
from omnifactory.training.prepare import source_fingerprint


def load_config(path):
    value = json.loads(path.read_text())
    return TrainConfig.model_validate(value.get("config", value))


def checked_samples(dataset):
    rows = list(iter_samples(dataset))
    if not rows or len({s.id for s in rows}) != len(rows):
        raise ValueError("evaluation requires nonempty unique sample IDs")
    for sample in rows:
        if len(sample.predictions) != 1:
            raise ValueError("one prediction per case is required")
        for asset in sample.assets.values():
            path = Path(asset.uri)
            path = path if path.is_absolute() else dataset.parent / path
            if not asset.sha256 or sha256(path) != asset.sha256:
                raise ValueError(f"evaluation media changed: {sample.id}")
    return rows


def audio_diagnostics(path):
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("expected finite nonempty mono waveform")
    width = round(rate * 0.02)
    n = len(wave)
    full = wave[: n // width * width].reshape(-1, width)
    rms = np.sqrt(np.mean(full * full, axis=1))
    ends = np.arange(1, len(rms) + 1) * width
    if n % width:
        rms = np.r_[rms, np.sqrt(np.mean(wave[n // width * width :] ** 2))]
        ends = np.r_[ends, n]
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    tails = {}
    for threshold in (-50, -60):
        active = np.flatnonzero(db > threshold)
        last = int(ends[active[-1]]) if len(active) else 0
        tails[str(threshold)] = (n - last) / rate
    return dict(seconds=n / rate, trailing_low_energy_seconds=tails)


def generate_arm(args):
    cfg = load_config(args.config)
    if cfg.adapter != "voxcpm2" or cfg.device != "cuda" or not cfg.deterministic:
        raise ValueError("requires deterministic CUDA VoxCPM2 recipe")
    if (args.mode == "adapter") != (args.checkpoint is not None):
        raise ValueError("adapter mode requires --checkpoint exclusively")
    if (args.mode == "export") != (args.export is not None):
        raise ValueError("export mode requires --export exclusively")
    training_contract = cfg.execution_contract()
    samples = checked_samples(args.dataset)
    cfg = cfg.model_copy(
        update=dict(validation_dataset=args.dataset, validation_root=args.dataset.parent)
    )
    evidence = {
        str(args.config.resolve()): sha256(args.config),
        str(args.dataset.resolve()): sha256(args.dataset),
    }
    if args.checkpoint:
        saved = load_config(args.checkpoint / "checkpoint.json")
        if saved.execution_contract() != training_contract:
            raise ValueError("checkpoint/base recipe mismatch")
        for name in ("checkpoint.json", "trainable.safetensors"):
            p = args.checkpoint / name
            evidence[str(p.resolve())] = sha256(p)
    else:
        saved = cfg
    artifact_id = verify_export(args.export)["artifact_id"] if args.export else None
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "identity.json",
        dict(
            mode=args.mode,
            config=cfg.model_dump(mode="json"),
            training_seed=saved.seed,
            generation_seed=args.generation_seed,
            max_new_tokens=192,
            max_audio_units=256,
            artifact_id=artifact_id,
            evidence=evidence,
            code=runtime_hashes(),
            script_sha256=sha256(Path(__file__)),
            generator_sha256=sha256(Path(__file__).with_name("run_real_data_baseline.py")),
            source_fingerprint=source_fingerprint(cfg.model, cfg.revision),
        ),
    )
    with deterministic_training(cfg):
        generate(
            cfg,
            samples,
            args.output,
            args.mode,
            max_new_tokens=192,
            max_audio_units=256,
            checkpoint=args.checkpoint,
            export_path=args.export,
            generation_seed=args.generation_seed,
        )
    path = args.output / f"{args.mode}-predictions.jsonl"
    records = read_rows(path)
    diagnostics = [
        dict(sample_id=r["sample_id"], **audio_diagnostics(Path(r["audio"]))) for r in records
    ]
    write_json(args.output / "audio-diagnostics.json", diagnostics)
    write_json(
        args.output / "report.json",
        dict(
            status="completed",
            cases=len(records),
            finished=sum(r["finished"] for r in records),
            evidence={
                p.name: sha256(p)
                for p in (
                    path,
                    args.output / "identity.json",
                    args.output / "audio-diagnostics.json",
                )
            },
        ),
    )


def score_arms(args):
    manifest = json.loads(args.manifest.read_text())
    dataset = Path(manifest["dataset"])
    samples = checked_samples(dataset)
    cfg = load_config(Path(manifest["config"]))
    expected = [(s.id, s.predictions[0].id, reference_text(s, s.predictions[0])) for s in samples]
    sets, evidence, energy = {}, {}, {}
    for label, name in manifest["arms"].items():
        if not label.replace("_", "").isalnum():
            raise ValueError("unsafe arm label")
        path = Path(name)
        records = read_rows(path)
        if [(r["sample_id"], r["prediction_id"], r["reference"]) for r in records] != expected:
            raise ValueError("incomplete or mismatched arm identities/references")
        for record in records:
            if sha256(Path(record["audio"])) != record["audio_sha256"]:
                raise ValueError("waveform changed")
        sets[label] = records
        evidence[str(path)] = sha256(path)
        energy[label] = [
            dict(sample_id=r["sample_id"], **audio_diagnostics(Path(r["audio"]))) for r in records
        ]
    if manifest.get("groundtruth"):
        records = []
        for s in samples:
            prediction = s.predictions[0]
            msg = next(m for m in s.messages if m.id == prediction.target_message_id)
            block = next(
                b for b in msg.content if b.id in prediction.target_block_ids and b.type == "audio"
            )
            asset = s.assets[block.media.asset_id]
            path = Path(asset.uri)
            path = path if path.is_absolute() else dataset.parent / path
            records.append(
                dict(
                    sample_id=s.id,
                    prediction_id=prediction.id,
                    reference=reference_text(s, prediction),
                    audio=str(path.resolve()),
                    audio_sha256=asset.sha256,
                    finished=True,
                )
            )
        if "groundtruth" in sets:
            raise ValueError("groundtruth label is reserved")
        sets["groundtruth"] = records
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(
        manifest=manifest,
        manifest_sha256=sha256(args.manifest),
        dataset_sha256=sha256(dataset),
        source_evidence=evidence,
        scorer_fingerprint=source_fingerprint(str(args.asr_model), None),
        code=runtime_hashes(),
        script_sha256=sha256(Path(__file__)),
        scorer_script_sha256=sha256(Path(__file__).with_name("validate_infrastructure_gpu.py")),
    )
    write_json(args.output / "identity.json", identity)
    write_json(args.output / "audio-diagnostics.json", energy)
    paths = transcribe(args.asr_model, cfg, samples, sets, args.output)
    write_json(
        args.output / "report.json",
        dict(
            status="completed",
            cases=len(samples),
            metrics={k: corpus_wer(read_rows(p)) for k, p in zip(sets, paths, strict=True)},
            termination={
                k: dict(finished=sum(r["finished"] for r in rows), total=len(rows))
                for k, rows in sets.items()
                if k != "groundtruth"
            },
            evidence={
                p.name: sha256(p)
                for p in [
                    *paths,
                    args.output / "identity.json",
                    args.output / "audio-diagnostics.json",
                ]
            },
            scope="complete raw waveforms; fixed ASR proxy, not human preference; no output filtering",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--config", type=Path, required=True)
    gen.add_argument("--dataset", type=Path, required=True)
    gen.add_argument("--mode", choices=("base", "adapter", "export"), required=True)
    gen.add_argument("--checkpoint", type=Path)
    gen.add_argument("--export", type=Path)
    gen.add_argument("--generation-seed", type=int, default=42)
    gen.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--asr-model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    (generate_arm if args.command == "generate" else score_arms)(args)


if __name__ == "__main__":
    main()
