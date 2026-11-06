"""One auditable report for shared gates; missing gates are never implied passed."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from omnifactory.artifacts import sha256, verify_export
from omnifactory.data.io import atomic_writer
from omnifactory.data.schema import Record
from omnifactory.errors import OmniFactoryError


class Gate(Record):
    status: Literal["not_run", "passed", "failed"] = "not_run"
    details: dict = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict)


class AcceptanceReport(Record):
    model_config = ConfigDict(frozen=False, extra="forbid", validate_assignment=True)
    format: str = "omnifactory-acceptance-v1"
    adapter: str
    task: str
    model: str
    revision: str | None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_fingerprint: str | None = None
    artifact_id: str | None = None
    gates: dict[str, Gate] = Field(
        default_factory=lambda: {
            k: Gate()
            for k in (
                "prepare",
                "train",
                "resume",
                "export",
                "inference",
                "native_reference",
                "http",
                "quality",
            )
        }
    )

    def save(self, path):
        with atomic_writer(path) as stream:
            stream.write(self.model_dump_json(indent=2))


def run_acceptance(config, output, *, generate=True):
    """Shared GPU/CPU protocol. Native-reference/quality/HTTP are separate explicit gates."""
    import gc

    import torch
    from safetensors.torch import load_file

    from omnifactory.adapters.runtime import adapter_spec
    from omnifactory.training.artifacts import write_json
    from omnifactory.training.benchmark import runtime_hashes
    from omnifactory.training.engine import train
    from omnifactory.training.evaluation import infer_export
    from omnifactory.training.export import export_checkpoint
    from omnifactory.training.prepare import prepare

    if config.max_steps < 2:
        raise OmniFactoryError("E_ACCEPTANCE", "resume acceptance needs >=2 optimizer steps")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = AcceptanceReport(
        adapter=config.adapter, task=config.task, model=config.model, revision=config.revision
    )
    path = output / "acceptance.json"
    cfg = config.model_copy(
        update={
            "output_dir": output / "run",
            "save_steps": max(1, config.max_steps // 2),
            "save_total_limit": None,
            "prepared_dataset": output / "prepared",
        }
    )
    write_json(output / "config.json", cfg.model_dump(mode="json"))
    write_json(output / "tested-code.json", runtime_hashes())
    active = "prepare"
    try:
        manifest = prepare(cfg, output / "prepared")
        report.source_fingerprint = manifest["source_fingerprint"]
        report.gates[active] = Gate(
            status="passed",
            details={
                "records": len(manifest["entries"]),
                "format": manifest["format"],
                "storage": manifest["storage"],
                "preparation": manifest["preparation"],
                "data_fingerprint": manifest["fingerprint"],
            },
        )
        report.save(path)
        active = "train"
        run = train(cfg)
        report.gates[active] = Gate(status="passed", details=run)
        report.save(path)
        active = "resume"
        checkpoint = next(
            p for p in sorted((output / "run").glob("checkpoint-*")) if p != Path(run["checkpoint"])
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        resumed = train(
            cfg.model_copy(update={"output_dir": output / "resumed", "resume_from": checkpoint})
        )
        a, b = (
            load_file(Path(p) / "trainable.safetensors")
            for p in (run["checkpoint"], resumed["checkpoint"])
        )
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise OmniFactoryError("E_RESUME_PARITY", "resumed trainable weights differ")
        del a, b

        def identical(left, right):
            if isinstance(left, torch.Tensor):
                return isinstance(right, torch.Tensor) and torch.equal(left, right)
            if isinstance(left, dict):
                return (
                    isinstance(right, dict)
                    and left.keys() == right.keys()
                    and all(identical(left[k], right[k]) for k in left)
                )
            if isinstance(left, (list, tuple)):
                return (
                    type(left) is type(right)
                    and len(left) == len(right)
                    and all(identical(x, y) for x, y in zip(left, right))
                )
            return left == right

        states = [
            torch.load(Path(p) / "optimizer.pt", map_location="cpu", weights_only=True)
            for p in (run["checkpoint"], resumed["checkpoint"])
        ]
        cursors = [
            json.loads((Path(p) / "checkpoint.json").read_text())["state"]
            for p in (run["checkpoint"], resumed["checkpoint"])
        ]
        # best_checkpoint paths legitimately differ between the two run directories.
        for cursor in cursors:
            cursor.pop("best_checkpoint", None)
        if not identical(*states) or cursors[0] != cursors[1]:
            raise OmniFactoryError("E_RESUME_PARITY", "resumed optimizer/RNG/cursor differ")
        del states, cursors
        report.gates[active] = Gate(
            status="passed",
            details={
                "max_abs": 0,
                "optimizer_rng_cursor_exact": True,
                "checkpoint": str(checkpoint),
            },
        )
        report.save(path)
        active = "export"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        exported = export_checkpoint(Path(run["checkpoint"]), output / "export", format="merged")
        artifact = verify_export(output / "export")
        report.artifact_id = artifact["artifact_id"]
        report.gates[active] = Gate(
            status="passed",
            details=exported,
            evidence={
                "export/artifact-manifest.json": sha256(output / "export/artifact-manifest.json")
            },
        )
        report.save(path)
        spec = adapter_spec(cfg.adapter)
        if generate and cfg.validation_dataset and not spec.stage_only:
            active = "inference"
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            generated = infer_export(
                output / "export",
                cfg.validation_dataset,
                output=output / "predictions.jsonl",
                root=cfg.validation_root,
                generate_image="image" in spec.outputs,
                generate_audio=cfg.task in ("speech_sft", "flow_sft") and "audio" in spec.outputs,
                max_codec_tokens=cfg.max_codec_tokens,
                device=cfg.device,
                dtype=cfg.dtype,
            )
            report.gates[active] = Gate(
                status="passed",
                details=generated,
                evidence={"predictions.jsonl": sha256(output / "predictions.jsonl")},
            )
        report.save(path)
    except Exception as exc:
        report.gates[active] = Gate(status="failed", details={"error": str(exc)})
        report.save(path)
        raise
    return report


def attach_gate(report_path, gate_path, *, name):
    if name not in ("http", "quality", "native_reference"):
        raise OmniFactoryError("E_ACCEPTANCE", "unsupported external gate")
    report = AcceptanceReport.model_validate_json(Path(report_path).read_text())
    gate_path = Path(gate_path)
    data = json.loads(gate_path.read_text())
    if not report.artifact_id or data.get("artifact_id") != report.artifact_id:
        raise OmniFactoryError(
            "E_ACCEPTANCE_ARTIFACT", "gate belongs to a different or unsealed export"
        )
    if data.get("status") != "passed":
        raise OmniFactoryError("E_ACCEPTANCE_GATE", "gate did not pass")
    report.gates[name] = Gate(
        status="passed", details=data, evidence={str(gate_path.resolve()): sha256(gate_path)}
    )
    report.save(Path(report_path))
    return report
