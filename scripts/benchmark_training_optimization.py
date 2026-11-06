"""Paired production-training comparison, including weights and exact resume.

Each arm runs in a fresh process from the same prepared records, base weights,
seed and optimizer schedule. Validation/generation are excluded in both arms.
This is one paired measurement, not a statistical or cross-hardware speed claim.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from omnifactory.artifacts import sha256
from omnifactory.training.artifacts import write_json
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig, read_config
from omnifactory.training.efficiency import summarize_steps
from omnifactory.training.engine import train
from omnifactory.training.prepare import prepare, source_fingerprint


def make_config(config, root, optimization, arm, cache_bytes):
    updates = dict(
        output_dir=root / arm,
        prepared_dataset=root / "prepared",
        prepared_cache_bytes=cache_bytes if optimization == "cache" and arm != "baseline" else 0,
        personaplex_agent_only_depth=optimization == "agent_depth" and arm != "baseline",
        validation_dataset=None,
        validation_root=None,
        eval_steps=0,
        deterministic=True,
        save_steps=config.max_steps // 2,
        save_total_limit=None,
        resume_from=None,
    )
    if arm == "resumed":
        origin = "baseline" if optimization == "cache" else "optimized"
        updates["resume_from"] = root / origin / f"checkpoint-{config.max_steps // 2:08d}"
    return TrainConfig.model_validate({**config.model_dump(), **updates})


def compare_weights(reference, actual, *, exact):
    left, right = (load_file(p / "trainable.safetensors") for p in (reference, actual))
    if set(left) != set(right):
        raise ValueError("different trainable parameter sets")
    stats = {}
    for name, value in left.items():
        candidate = right[name]
        if candidate.shape != value.shape or candidate.dtype != value.dtype:
            raise ValueError(f"different parameter shape/dtype: {name}")
        a, b = value.float(), candidate.float()
        delta = (a - b).abs()
        stats[name] = dict(
            max_abs=float(delta.max()),
            reference_max_abs=float(a.abs().max()),
            unequal=int((value != candidate).sum()),
            elements=value.numel(),
            close=bool(torch.allclose(a, b, rtol=0 if exact else 1e-4, atol=0 if exact else 1e-6)),
        )
    return dict(
        status="passed" if all(s["close"] for s in stats.values()) else "failed",
        exact=exact,
        rtol=0 if exact else 1e-4,
        atol=0 if exact else 1e-6,
        max_abs=max(s["max_abs"] for s in stats.values()),
        tensors=stats,
    )


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(root, optimization):
    results = {
        arm: json.loads((root / arm / "result.json").read_text())
        for arm in ("baseline", "optimized", "resumed")
    }
    paired = compare_weights(
        Path(results["baseline"]["checkpoint"]),
        Path(results["optimized"]["checkpoint"]),
        exact=optimization == "cache",
    )
    resumed = compare_weights(
        Path(results["optimized"]["checkpoint"]), Path(results["resumed"]["checkpoint"]), exact=True
    )
    write_json(root / "weight-parity.json", paired)
    write_json(root / "resume-parity.json", resumed)
    steps = {arm: read_rows(root / arm / "metrics.jsonl") for arm in ("baseline", "optimized")}
    if len(steps["baseline"]) != len(steps["optimized"]):
        raise ValueError("different optimizer step counts")
    for a, b in zip(steps["baseline"], steps["optimized"], strict=True):
        for key in (
            "step",
            "epoch",
            "examples",
            "microbatches",
            "normalization_counts",
            "normalization_units",
            "useful_llm_slots",
            "learning_rates",
        ):
            if a[key] != b[key]:
                raise ValueError(f"different step workload: {key}")
    resources = {}
    for arm, rows in steps.items():
        warm = [row for row in rows if row["epoch"] > 0]
        resources[arm] = dict(
            after_two_warmup_steps=summarize_steps(rows[2:]),
            after_first_epoch=summarize_steps(warm) if warm else None,
            host_loading_seconds=sum(row["data_load_host_seconds"] for row in rows),
            all_step_peak_allocated_bytes=max(
                (
                    row["cuda_peak_allocated_bytes"]
                    for row in rows
                    if row["cuda_peak_allocated_bytes"] is not None
                ),
                default=None,
            ),
            cache=rows[-1].get("prepared_cache"),
        )
    a, b = (resources[arm]["after_two_warmup_steps"] for arm in ("baseline", "optimized"))
    report = dict(
        status="passed" if paired["status"] == resumed["status"] == "passed" else "failed",
        optimization=optimization,
        source_fingerprint=json.loads((root / "prepared/manifest.json").read_text())[
            "source_fingerprint"
        ],
        resources=resources,
        step_time_ratio=a["seconds"] / b["seconds"],
        weights=paired["status"],
        max_weight_delta=paired["max_abs"],
        exact_resume=resumed["status"],
        evidence={
            str(p.relative_to(root)): sha256(p)
            for p in [
                root / "identity.json",
                root / "weight-parity.json",
                root / "resume-parity.json",
                root / "baseline/metrics.jsonl",
                root / "optimized/metrics.jsonl",
                root / "resumed/metrics.jsonl",
            ]
        },
        scope="single paired production run with matched steps; validation/checkpoints excluded from step timing; not quality or minimum VRAM",
    )
    write_json(root / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--optimization", choices=("cache", "agent_depth"), required=True)
    p.add_argument("--cache-bytes", type=int, default=128 * 1024**2)
    p.add_argument(
        "--stage",
        choices=("all", "prepare", "baseline", "optimized", "resumed", "report"),
        default="all",
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, root = read_config(args.config), args.output.resolve()
    if cfg.max_steps < 4 or args.cache_bytes <= 0 or cfg.resume_from is not None:
        p.error("requires >=4 steps, positive cache bytes and a fresh base configuration")
    if args.optimization == "agent_depth" and cfg.adapter != "personaplex":
        p.error("agent_depth requires PersonaPlex")
    identity = dict(
        config=cfg.model_dump(mode="json"),
        optimization=args.optimization,
        cache_bytes=args.cache_bytes,
        source=source_fingerprint(cfg.model, cfg.revision),
        code=runtime_hashes(),
        script_sha256=sha256(Path(__file__)),
        dataset_sha256=sha256(cfg.dataset),
    )
    if args.stage == "all":
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "identity.json", identity)
        for stage in ("prepare", "baseline", "optimized", "resumed", "report"):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--stage", stage], check=True)
        return
    if json.loads((root / "identity.json").read_text()) != identity:
        raise ValueError("source, data, config or code changed during comparison")
    if args.stage == "report":
        report = summarize(root, args.optimization)
        print(json.dumps(report, indent=2), flush=True)
        if report["status"] != "passed":
            raise ValueError("parameter comparison failed; see retained parity artifacts")
    else:
        config = make_config(
            cfg,
            root,
            args.optimization,
            "baseline" if args.stage == "prepare" else args.stage,
            args.cache_bytes,
        )
        if args.stage == "prepare":
            prepare(config, root / "prepared")
        else:
            result = train(config, progress=lambda row: print(json.dumps(row), flush=True))
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
