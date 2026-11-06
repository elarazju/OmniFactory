"""Reproducible optimizer-step benchmarks using the production loss/backward path."""

import hashlib
import json
import math
from pathlib import Path

import torch

from omnifactory.adapters.runtime import adapter_class, load_model, task_batch_capability
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import ModalBatchPlanner, SampleCost
from omnifactory.training.artifacts import write_json
from omnifactory.training.data_loading import PreparedPayloadLoader
from omnifactory.training.efficiency import StepMeasurement, environment, summarize_steps
from omnifactory.training.engine import backward_window
from omnifactory.training.optimizer import create_optimizer
from omnifactory.training.policy import apply_policy
from omnifactory.training.prepare import prepare


def source_batches(costs, planner):
    """Contiguous source-order baseline with the same resource limits; never drop rows."""
    batch = []
    for cost in costs:
        failures = planner.exceeded(planner.cost([cost]))
        if failures:
            raise OmniFactoryError("E_SAMPLE_OVER_BUDGET", f"{cost.sample_id}: {failures}")
        compatible = not batch or (
            batch[0].contract_id == cost.contract_id
            and (
                planner.capability.grouping_key(batch[0]) == planner.capability.grouping_key(cost)
                if planner.capability
                else set(batch[0].modalities) == set(cost.modalities)
            )
        )
        if batch and (not compatible or planner.exceeded(planner.cost([*batch, cost]))):
            yield tuple((r.sample_id, r.prediction_id) for r in batch)
            batch = []
        batch.append(cost)
    if batch:
        yield tuple((r.sample_id, r.prediction_id) for r in batch)


def runtime_hashes():
    root = Path(__file__).resolve().parents[1]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def benchmark(config, output, *, batching="modal", warmup_epochs=1, repeats=3):
    """Full dataset passes avoid comparing throughput on different subsets.

    Each pass groups consecutive microbatches into accumulation windows. Its last
    window may be shorter. Warmup performs real updates and initializes Adam state;
    measured repeats retain the optimizer and allocator, as a normal run does.
    """
    if batching not in ("modal", "source") or warmup_epochs < 1 or repeats < 1:
        raise OmniFactoryError("E_BENCHMARK", "use modal/source, warmup_epochs>=1, repeats>=1")
    if config.memory_profile:
        raise OmniFactoryError(
            "E_BENCHMARK", "calibrate from count budgets without an existing profile"
        )
    if config.resume_from:
        raise OmniFactoryError("E_BENCHMARK", "benchmark starts from the pinned base, not resume")
    output = Path(output).resolve()
    if output.exists():
        raise OmniFactoryError("E_OUTPUT_EXISTS", "choose a new benchmark directory")
    if config.device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(config.seed)
    output.mkdir(parents=True)
    write_json(output / "config.json", config.model_dump(mode="json"))
    manifest = prepare(config, output / "prepared")
    adapter = adapter_class(config.adapter).from_config(
        config.model_copy(update={"model": str(output / "prepared/processor"), "revision": None})
    )
    model, groups, policy = apply_policy(load_model(config), config)
    model.train()
    optimizer = create_optimizer(groups, config, weight_decay=0.0)
    trainables = [p for p in model.parameters() if p.requires_grad]
    entries = {(e["sample_id"], e["prediction_id"]): e for e in manifest["entries"]}
    costs = [SampleCost.model_validate(e["cost"]) for e in manifest["entries"]]
    cost_by_id = {(c.sample_id, c.prediction_id): c for c in costs}
    planner = ModalBatchPlanner(
        config.budget,
        capability=task_batch_capability(config),
        timed=config.duplex_schedule is not None,
        seed=config.seed,
        window_size=config.window_size,
        packing=config.packing != "none",
    )
    batches = (
        [p.example_ids for p in planner.plan(costs)]
        if batching == "modal"
        else list(source_batches(costs, planner))
    )
    flat = [key for b in batches for key in b]
    if len(flat) != len(entries) or set(flat) != set(entries):
        raise OmniFactoryError("E_BENCHMARK", "batch schedule must cover every prediction once")
    windows = [
        batches[i : i + config.gradient_accumulation_steps]
        for i in range(0, len(batches), config.gradient_accumulation_steps)
    ]
    schedule = [
        {
            "case": i,
            "microbatches": [
                {
                    "example_ids": ids,
                    "cost": planner.cost([cost_by_id[k] for k in ids]).model_dump(),
                    "modalities": sorted({m for k in ids for m in cost_by_id[k].modalities}),
                }
                for ids in window
            ],
        }
        for i, window in enumerate(windows)
    ]
    env = environment(config.device)
    write_json(output / "environment.json", env)
    write_json(output / "schedule.json", schedule)
    write_json(output / "tested-code.json", runtime_hashes())
    measured = []
    payload_loader = PreparedPayloadLoader(
        output / "prepared", max_bytes=config.prepared_cache_bytes
    )
    with (output / "measurements.jsonl").open("w") as stream:
        for epoch in range(warmup_epochs + repeats):
            for case, window in enumerate(windows):
                measurement = StepMeasurement(config.device).start()
                optimizer.zero_grad(set_to_none=True)
                bundle = backward_window(
                    model,
                    adapter,
                    config,
                    [[entries[k] for k in ids] for ids in window],
                    payload_loader,
                    measurement=measurement,
                    planner=planner,
                )
                norm = torch.nn.utils.clip_grad_norm_(
                    trainables, config.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                record = {
                    **measurement.finish(),
                    "epoch": epoch,
                    "case": case,
                    "warmup": epoch < warmup_epochs,
                    "loss": float(bundle.loss),
                    "grad_norm": float(norm),
                    **bundle.report(),
                }
                if config.prepared_cache_bytes:
                    record["prepared_cache"] = payload_loader.report()
                if not math.isfinite(record["loss"]):
                    raise OmniFactoryError("E_TRAINING_LOSS", "benchmark loss is not finite")
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if not record["warmup"]:
                    measured.append(record)
    result = {
        "format": "omnifactory-benchmark-v1",
        "environment": env,
        "contract_id": manifest["contract_id"],
        "source_fingerprint": manifest["source_fingerprint"],
        "data_fingerprint": manifest["fingerprint"],
        "batching": batching,
        "warmup_epochs": warmup_epochs,
        "repeats": repeats,
        "predictions_per_epoch": len(entries),
        "trainable_parameters": policy.trainable_numel,
        "summary": summarize_steps(measured),
        "cases": {
            str(i): summarize_steps([r for r in measured if r["case"] == i])
            for i in range(len(windows))
        },
        "timing_scope": "prepared tensor IO + collate + transfer + forward/backward + optimizer; excludes prepare/load/checkpoint",
        "llm_slot_unit": "native attention-mask positions; MiMo patches, not codec symbols",
        "quality_verified": False,
        "minimum_vram_verified": False,
        "prepared_cache": payload_loader.report(),
    }
    from omnifactory.training.feature_cache import cache_stats

    result["feature_cache"] = cache_stats(model)
    write_json(output / "benchmark.json", result)
    return result
