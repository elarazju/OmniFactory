"""Single-device SFT with disk-backed batches and explicit per-objective normalization."""

import json
import math
from contextlib import closing, nullcontext
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from omnifactory.adapters.base import EncodedExample
from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.adapters.supervision import expected_losses
from omnifactory.data.supervision import SupervisionPlan
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import SampleCost
from omnifactory.training.artifacts import new_directory, write_json
from omnifactory.training.config import TrainConfig
from omnifactory.training.control import (
    deterministic_training,
    evaluate_prepared,
    learning_rate_factor,
    retain_checkpoints,
)
from omnifactory.training.data_loading import PreparedPayloadLoader
from omnifactory.training.efficiency import StepMeasurement, environment
from omnifactory.training.losses import LossBundle
from omnifactory.training.policy import apply_policy
from omnifactory.training.prepare import load_prepared, prepare, source_fingerprint


def to_device(batch, model, config):
    return adapter_class(config.adapter).to_device(batch, model, config)


def load_encoded(root, entry):
    if "supervision" not in entry:
        raise OmniFactoryError(
            "E_PREPARED_VERSION", "missing supervision contract; regenerate data with prepare"
        )
    from omnifactory.training.storage import read_record

    encoded = EncodedExample(
        tensors=read_record(root, entry),
        cost=SampleCost.model_validate(entry["cost"]),
        prompt_length=entry.get("prompt_length"),
        answer=entry.get("answer"),
        layout=entry.get("layout"),
        metadata=entry.get("metadata"),
        supervision=SupervisionPlan.model_validate(entry["supervision"]),
    ).validate()
    if encoded.normalization_counts != entry["normalization_counts"]:
        raise OmniFactoryError("E_PREPARED_MEASURE", "stored measures differ from tensors")
    return encoded


def load_tensors(root, entry):
    return load_encoded(root, entry).payload


def entry_expectations(entries):
    return expected_losses(
        (SupervisionPlan.model_validate(e["supervision"]), e["normalization_counts"])
        for e in entries
    )


def restore_weights(model, checkpoint):
    weights = load_file(checkpoint / "trainable.safetensors")
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(weights) != expected:
        raise OmniFactoryError(
            "E_CHECKPOINT_PARAMETERS", "saved trainable parameters do not match policy"
        )
    model.load_state_dict(weights, strict=False)


def save_checkpoint(path, model, optimizer, config, manifest, state, plan):
    with new_directory(path) as root:
        save_file(
            {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            root / "trainable.safetensors",
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
            },
            root / "optimizer.pt",
        )
        write_json(
            root / "checkpoint.json",
            {
                "format": "omnifactory-checkpoint-v1",
                "config": config.model_dump(mode="json"),
                "state": state,
                "data_fingerprint": manifest["fingerprint"],
                "source_fingerprint": manifest["source_fingerprint"],
                "contract_id": manifest["contract_id"],
                "groups_sha256": manifest["groups"].copy_to(root / "groups.sqlite"),
                "adaptation": plan.model_dump(mode="json"),
            },
        )


def load_checkpoint(checkpoint: Path, *, device=None, for_export=False):
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    config = TrainConfig.model_validate(metadata["config"])
    if source_fingerprint(config.model, config.revision) != metadata["source_fingerprint"]:
        raise OmniFactoryError("E_SOURCE_CHANGED", "base model differs from checkpoint source")
    # Checkpoint loading here is for evaluation/export; resume uses restore_weights.
    config = config.model_copy(
        update={
            "packing": "none",
            "feature_cache": None,
            "memory_profile": None,
            "budget": config.budget.model_copy(update={"max_gpu_bytes": None}),
        }
    )
    if for_export:
        config = config.model_copy(
            update={
                "device": device or "cpu",
                "quantization": "none",
                "dtype": "float32",
                "gradient_checkpointing": False,
                "packing": "none",
                "memory_profile": None,
                "budget": config.budget.model_copy(update={"max_gpu_bytes": None}),
            }
        )
    elif device is not None:
        config = TrainConfig.model_validate({**config.model_dump(), "device": device})
    model, _, _ = apply_policy(load_model(config), config)
    restore_weights(model, checkpoint)
    model.eval()
    adapter = adapter_class(config.adapter).from_config(config)
    return model, adapter, config, metadata


def backward_window(
    model, adapter, config, chunks, load_payload, *, measurement=None, planner=None
):
    """Shared execution path: objective sums scale by the whole window's measures."""
    window = entry_expectations(e for chunk in chunks for e in chunk)
    if not any(window[1].values()):
        raise OmniFactoryError("E_EMPTY_SUPERVISION", "accumulation window has no active measure")
    measured = []
    executions = []
    entries = (e for chunk in chunks for e in chunk)
    payload_iterator = (
        load_payload.iterate(entries)
        if hasattr(load_payload, "iterate")
        else (load_payload(e) for e in entries)
    )
    with closing(payload_iterator):
        for chunk in chunks:
            with measurement.host_stage("data_load") if measurement is not None else nullcontext():
                payloads = [next(payload_iterator) for _ in chunk]
            with measurement.host_stage("collate") if measurement is not None else nullcontext():
                if config is None:  # Low-level supervision fixtures have no installed adapter.
                    batches, execution = [adapter.collate(payloads)], None
                else:
                    from omnifactory.adapters.contracts import collate_task

                    batches, execution = collate_task(
                        adapter,
                        config,
                        payloads,
                        [SampleCost.model_validate(e["cost"]) for e in chunk],
                    )
            del payloads
            outputs = []
            for batch in batches:
                if measurement is not None:
                    measurement.observe(batch, len(chunk) if len(batches) == 1 else 1)
                outputs.append(
                    adapter.forward(model, adapter.to_device(batch, model, config)).losses
                )
            bundle = LossBundle.combine(outputs)
            if execution is not None:
                costs = [SampleCost.model_validate(e["cost"]) for e in chunk]
                if planner is not None:
                    cost = planner.cost(costs)
                else:
                    from omnifactory.adapters.runtime import task_batch_capability
                    from omnifactory.planning.batching import estimate_batch_cost

                    cost = estimate_batch_cost(
                        costs,
                        packing=config.packing != "none",
                        capability=task_batch_capability(config),
                        timed=config.duplex_schedule is not None,
                    )
                execution["budget_cost"] = cost.model_dump()
                executions.append(execution)
            bundle.validate_expected(entry_expectations(chunk))
            scaled = bundle.scaled_loss(window[1])
            if not torch.isfinite(scaled) or not scaled.requires_grad:
                raise OmniFactoryError(
                    "E_TRAINING_LOSS", "nonfinite loss or no gradient path to selected components"
                )
            scaled.backward()
            measured.append(bundle.detached())
    result = LossBundle.combine(measured)
    result.execution = executions
    return result


def train(config: TrainConfig, *, progress=None):
    with deterministic_training(config):
        return _train(config, progress=progress)


def _train(config: TrainConfig, *, progress=None):
    if config.output_dir.exists():
        raise OmniFactoryError(
            "E_OUTPUT_EXISTS", "choose a new run directory; resume_from points at an old checkpoint"
        )
    if config.device == "mps" and config.resume_from:
        raise OmniFactoryError(
            "E_RESUME_DEVICE", "exact MPS RNG restoration is not implemented; use CPU/CUDA"
        )
    torch.manual_seed(config.seed)
    if config.device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    # Keep completed checkpoints if a later training step fails.
    config.output_dir.mkdir(parents=True)
    write_json(config.output_dir / "config.json", config.model_dump(mode="json"))
    prepared = config.prepared_dataset or config.output_dir / "prepared"
    manifest = (
        load_prepared(config, prepared) if config.prepared_dataset else prepare(config, prepared)
    )
    validation = None
    validation_path = config.output_dir / "validation-prepared"
    if config.eval_steps:
        val_cfg = config.model_copy(
            update={
                "dataset": config.validation_dataset,
                "dataset_root": config.validation_root or config.validation_dataset.parent,
            }
        )
        validation = prepare(val_cfg, validation_path, split="validation")
        if manifest["groups"].overlaps(validation["groups"]):
            raise OmniFactoryError("E_SPLIT_LEAK", "validation groups occurred in training")
    adapter = adapter_class(config.adapter).from_config(
        config.model_copy(update={"model": str(prepared / "processor"), "revision": None})
    )
    model, groups, plan = apply_policy(load_model(config), config)
    from omnifactory.training.optimizer import create_optimizer

    optimizer = create_optimizer(groups, config)
    base_lrs = [g["lr"] for g in groups]
    write_json(config.output_dir / "adaptation.json", plan.model_dump(mode="json"))
    write_json(config.output_dir / "environment.json", environment(config.device))
    state = {"step": 0, "epoch": 0, "batch": 0}
    if config.resume_from:
        prior = json.loads((config.resume_from / "checkpoint.json").read_text())
        for key in ("contract_id", "data_fingerprint", "source_fingerprint"):
            actual = manifest["fingerprint"] if key == "data_fingerprint" else manifest[key]
            if prior[key] != actual:
                raise OmniFactoryError(
                    "E_RESUME_CONTRACT",
                    f"{key} differs; resume requires identical model, data, and training policy",
                )
        restore_weights(model, config.resume_from)
        saved = torch.load(
            config.resume_from / "optimizer.pt", map_location=config.device, weights_only=True
        )
        optimizer.load_state_dict(saved["optimizer"])
        state = prior["state"]
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if config.device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        if state["step"] >= config.max_steps:
            raise OmniFactoryError(
                "E_RESUME_STEPS", "max_steps must exceed the saved optimizer step"
            )
    entries = manifest["entries"]
    from omnifactory.training.calibration import configured_planner

    planner = configured_planner(config, manifest["contract_id"], manifest["source_fingerprint"])
    cached_epoch, plans = -1, iter(())

    def next_batch():
        from itertools import islice

        nonlocal cached_epoch, plans
        while True:
            if cached_epoch != state["epoch"]:
                planner.seed = config.seed + state["epoch"]
                plans = iter(planner.plan(SampleCost.model_validate(e["cost"]) for e in entries))
                plans = islice(plans, state["batch"], None)
                cached_epoch = state["epoch"]
            plan = next(plans, None)
            if plan is not None:
                state["batch"] += 1
                return [entries[key] for key in plan.example_ids]
            state["epoch"] += 1
            state["batch"] = 0

    trainables = [p for p in model.parameters() if p.requires_grad]
    payload_loader = PreparedPayloadLoader(
        prepared,
        max_bytes=config.prepared_cache_bytes,
        prefetch_records=config.prepared_prefetch_records,
        prefetch_bytes=config.prepared_prefetch_bytes,
    )
    model.train()
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    last_checkpoint = None
    with (config.output_dir / "metrics.jsonl").open("w") as log:
        while state["step"] < config.max_steps:
            chunks = [next_batch() for _ in range(config.gradient_accumulation_steps)]
            measurement = StepMeasurement(config.device).start()
            optimizer.zero_grad(set_to_none=True)
            aggregate = backward_window(
                model,
                adapter,
                config,
                chunks,
                payload_loader,
                measurement=measurement,
                planner=planner,
            )
            weighted_loss = float(aggregate.loss)
            norm = torch.nn.utils.clip_grad_norm_(
                trainables, config.max_grad_norm, error_if_nonfinite=True
            )
            factor = learning_rate_factor(config, state["step"] + 1)
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * factor
            optimizer.step()
            efficiency = measurement.finish()
            state["step"] += 1
            metrics = {
                "step": state["step"],
                "loss": weighted_loss,
                "learning_rates": [g["lr"] for g in optimizer.param_groups],
                **aggregate.report(),
                "batch_execution": aggregate.execution,
                "target_tokens": sum(aggregate.counts.values())
                if all(u == "symbols" for u in aggregate.units.values())
                else None,
                "grad_norm": float(norm),
                "epoch": state["epoch"],
                **efficiency,
            }
            if config.prepared_cache_bytes or config.prepared_prefetch_records:
                metrics["prepared_cache"] = payload_loader.report()
            if not math.isfinite(weighted_loss):
                raise OmniFactoryError("E_TRAINING_LOSS", "nonfinite accumulated loss")
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if progress:
                progress(metrics)
            improved = False
            if validation is not None and (
                state["step"] % config.eval_steps == 0 or state["step"] == config.max_steps
            ):
                evaluation = evaluate_prepared(model, adapter, config, validation_path, validation)
                with (config.output_dir / "evaluation.jsonl").open("a") as evaluation_log:
                    evaluation_log.write(json.dumps({"step": state["step"], **evaluation}) + "\n")
                if evaluation["loss"] < state.get("best_metric", float("inf")):
                    improved = True
                    state["best_metric"] = evaluation["loss"]
                    state["best_checkpoint"] = str(
                        (config.output_dir / f"checkpoint-{state['step']:08d}").resolve()
                    )
            if (
                improved
                or state["step"] % config.save_steps == 0
                or state["step"] == config.max_steps
            ):
                last_checkpoint = config.output_dir / f"checkpoint-{state['step']:08d}"
                save_checkpoint(last_checkpoint, model, optimizer, config, manifest, state, plan)
                retain_checkpoints(
                    config.output_dir,
                    config.save_total_limit,
                    protected=(last_checkpoint, state.get("best_checkpoint")),
                )
    result = {
        "checkpoint": str(last_checkpoint),
        "steps": state["step"],
        "last_loss": weighted_loss,
        "best_checkpoint": state.get("best_checkpoint"),
        "best_metric": state.get("best_metric"),
        "prepared_dataset": str(prepared.resolve()),
        "trainable_parameters": plan.trainable_numel,
        "device": config.device,
        "base_model": config.model,
        "vllm_deployment_verified": False,
    }
    from omnifactory.training.feature_cache import cache_stats

    result["feature_cache"] = cache_stats(model)
    write_json(config.output_dir / "result.json", result)
    return result
