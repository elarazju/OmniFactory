"""Score immutable baseline waveforms in a fresh directory without regenerating them."""

import argparse
import json
from pathlib import Path

import torch
from run_real_data_baseline import check_dataset, read_rows, reference_text
from validate_infrastructure_gpu import transcribe

from omnifactory.artifacts import sha256, verify_export
from omnifactory.training.artifacts import write_json
from omnifactory.training.asr_metrics import corpus_wer
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.prepare import source_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--asr-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    original = json.loads((args.run / "identity.json").read_text())
    cfg = TrainConfig.model_validate(original["config"])
    if cfg.adapter != "voxcpm2":
        p.error("requires saved VoxCPM2 baseline outputs")
    for path, expected in original["datasets"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("source dataset changed")
    samples = check_dataset(cfg)["validation"]
    expected = [(s.id, p.id, reference_text(s, p)) for s in samples for p in s.predictions]
    arms = {}
    evidence = {str((args.run / "identity.json").resolve()): sha256(args.run / "identity.json")}
    for label in ("base", "adapter", "export"):
        path = args.run / f"{label}-predictions.jsonl"
        records = read_rows(path)
        if [(r["sample_id"], r["prediction_id"], r["reference"]) for r in records] != expected:
            raise ValueError("incomplete or mismatched generation records")
        for row in records:
            if sha256(Path(row["audio"])) != row["audio_sha256"]:
                raise ValueError("generated waveform changed")
        arms[label] = records
        evidence[str(path.resolve())] = sha256(path)
    artifact = verify_export(args.run / "export")["artifact_id"]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "identity.json",
        dict(
            source_run=str(args.run.resolve()),
            generation_identity=original,
            artifact_id=artifact,
            source_evidence=evidence,
            code=runtime_hashes(),
            script_sha256=sha256(Path(__file__)),
            scorer_script_sha256=sha256(Path(__file__).with_name("validate_infrastructure_gpu.py")),
            scorer_fingerprint=source_fingerprint(str(args.asr_model), None),
        ),
    )
    for label in arms:
        name = f"{label}-predictions.jsonl"
        (args.output / name).write_bytes((args.run / name).read_bytes())
    paths = transcribe(args.asr_model, cfg, samples, arms, args.output)
    report = dict(
        status="completed",
        source_run=str(args.run.resolve()),
        artifact_id=artifact,
        cases=len(expected),
        metrics={k: corpus_wer(read_rows(v)) for k, v in zip(arms, paths, strict=True)},
        termination={
            k: dict(finished=sum(r["finished"] for r in rows), total=len(rows))
            for k, rows in arms.items()
        },
        evidence={
            p.name: sha256(p) for p in [args.output / "identity.json", *args.output.glob("*.jsonl")]
        },
        scope="complete saved waveforms, including unfinished synthesis; development ASR proxy",
    )
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
