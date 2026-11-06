"""Controlled prompt/casing screen before changing a real-data training recipe.

Vox uses the first available case in each speaker/reference-condition stratum;
Qwen uses every validation case. Selection happens before any new generation.
Outputs are development diagnostics, never an untouched test benchmark.
"""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
from prepare_asr_recipe import prepare as prepare_asr
from prepare_librispeech_tts import prepare as prepare_tts
from run_real_data_baseline import generate, read_rows
from validate_infrastructure_gpu import transcribe

from omnifactory.artifacts import sha256
from omnifactory.data.io import iter_samples
from omnifactory.training.artifacts import write_json
from omnifactory.training.asr_metrics import corpus_wer, words
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.control import deterministic_training
from omnifactory.training.prepare import source_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--asr-source", type=Path, required=True)
    p.add_argument("--asr-model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--stage", choices=("all", "prepare", "control", "base", "adapter", "score"), default="all"
    )
    args = p.parse_args()
    root = args.output.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = json.loads((args.checkpoint / "checkpoint.json").read_text())
    cfg = TrainConfig.model_validate(metadata["config"])
    vox = cfg.adapter == "voxcpm2"
    if cfg.adapter not in ("voxcpm2", "qwen2_5_omni") or (vox and args.asr_model is None):
        p.error("Qwen2.5 ASR or Vox TTS + scorer required")
    identity = dict(
        scorer=source_fingerprint(str(args.asr_model), None) if vox else None,
        checkpoint=sha256(args.checkpoint / "checkpoint.json"),
        trainable=sha256(args.checkpoint / "trainable.safetensors"),
        source={s: sha256(args.asr_source / f"{s}.jsonl") for s in ("train", "validation")},
        code=runtime_hashes(),
        scripts={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "run_recipe_screen.py",
                "run_real_data_baseline.py",
                "prepare_asr_recipe.py",
                "prepare_librispeech_tts.py",
                "validate_infrastructure_gpu.py",
            )
        },
    )
    if args.stage == "all":
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "identity.json", identity)
        for stage in ("prepare", "control", "base", "adapter", "score"):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--stage", stage], check=True)
        return
    if json.loads((root / "identity.json").read_text()) != identity:
        raise ValueError("experiment inputs/code changed")
    if args.stage == "prepare":
        counts = {
            s: sum(1 for _ in iter_samples(args.asr_source / f"{s}.jsonl"))
            for s in ("train", "validation")
        }
        if vox:
            prepare_tts(
                args.asr_source,
                root / "data",
                train_count=counts["train"],
                validation_count=counts["validation"],
                text_case="sentence",
            )
            prepare_tts(
                args.asr_source,
                root / "control-data",
                train_count=counts["train"],
                validation_count=counts["validation"],
            )
        else:
            prepare_asr(args.asr_source, root / "data")
        samples = list(iter_samples(root / "data/validation.jsonl"))
        selected = []
        seen = set()
        for sample in samples:
            key = (sample.provenance.group_id, bool(sample.predictions[0].conditions))
            if not vox or key not in seen:
                selected.append(sample.id)
                seen.add(key)
        write_json(
            root / "selection.json",
            dict(
                ids=selected,
                rule="first per speaker/reference stratum" if vox else "all validation",
                scope="development screen selected before generation",
            ),
        )
        (root / "train").mkdir()
        write_json(root / "train/result.json", dict(checkpoint=str(args.checkpoint.resolve())))
    else:
        ids = set(json.loads((root / "selection.json").read_text())["ids"])
        data = root / "data/validation.jsonl"
        if args.stage == "control":
            data = (
                (root / "control-data/validation.jsonl")
                if vox
                else args.asr_source / "validation.jsonl"
            )
        samples = [s for s in iter_samples(data) if s.id in ids]
        cfg = cfg.model_copy(
            update=dict(validation_dataset=data, validation_root=data.parent, deterministic=True)
        )
        if args.stage != "score":
            arm = "base" if args.stage == "control" else args.stage
            dest = root / "control" if args.stage == "control" else root
            dest.mkdir(exist_ok=True)
            with deterministic_training(cfg):
                generate(cfg, samples, dest, arm, max_new_tokens=192, max_audio_units=256)
        else:
            arms = {
                arm: read_rows(root / f"{arm}-predictions.jsonl") for arm in ("base", "adapter")
            }
            arms["control"] = read_rows(root / "control/base-predictions.jsonl")
            if vox:
                ground = []
                for sample in samples:
                    asset = sample.assets["recording"]
                    path = Path(asset.uri)
                    ground.append(
                        dict(sample_id=sample.id, audio=str(path), audio_sha256=asset.sha256)
                    )
                arms["groundtruth"] = ground
                paths = transcribe(args.asr_model, cfg, samples, arms, root)
                gc.collect()
                torch.cuda.empty_cache()
                scored = {arm: read_rows(path) for arm, path in zip(arms, paths, strict=True)}
            else:
                scored = arms
            expected = {s.id for s in samples}
            for arm, records in scored.items():
                if len(records) != len(expected) or {r["sample_id"] for r in records} != expected:
                    raise ValueError("incomplete scoring")
                for a, b in zip(scored["base"], records, strict=True):
                    if a["sample_id"] != b["sample_id"] or words(a["reference"]) != words(
                        b["reference"]
                    ):
                        raise ValueError("reference changed beyond casing")
            write_json(
                root / "report.json",
                dict(
                    status="completed",
                    cases=len(samples),
                    metrics={arm: corpus_wer(records) for arm, records in scored.items()},
                    termination={
                        arm: dict(finished=sum(r["finished"] for r in records), total=len(records))
                        for arm, records in arms.items()
                        if arm != "groundtruth"
                    },
                    evidence={
                        str(path.relative_to(root)): sha256(path)
                        for path in [
                            root / "identity.json",
                            root / "selection.json",
                            *root.glob("*-predictions.jsonl"),
                            *root.glob("*-transcripts.jsonl"),
                            root / "control/base-predictions.jsonl",
                        ]
                    },
                    scope="controlled development prompt/casing screen; groundtruth is scorer-floor diagnostic; not training benefit",
                ),
            )
    write_json(root / f"{args.stage}-stage.json", dict(status="completed"))


if __name__ == "__main__":
    main()
