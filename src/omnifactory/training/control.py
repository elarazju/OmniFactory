"""Deterministic learning-rate schedules, held-out evaluation and checkpoint retention."""

import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch

from omnifactory.errors import OmniFactoryError
from omnifactory.training.losses import LossBundle


@contextmanager
def deterministic_training(config):
    """Scope reproducible kernels to this run, including restoration on failure."""
    if not config.deterministic:
        yield
        return
    if config.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
        ":4096:8",
        ":16:8",
    ):
        raise OmniFactoryError(
            "E_DETERMINISTIC_CUDA",
            "launch deterministic CUDA training with CUBLAS_WORKSPACE_CONFIG=:4096:8",
        )
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cudnn.deterministic = previous[3]


def learning_rate_factor(config, step):
    """One-based optimizer step; horizon is explicit and stable across resumed runs."""
    if config.warmup_steps and step <= config.warmup_steps:
        return step / config.warmup_steps
    if config.lr_scheduler == "constant":
        return 1.0
    progress = min(
        1.0,
        max(
            0.0,
            (step - config.warmup_steps - 1)
            / max(1, config.scheduler_steps - config.warmup_steps - 1),
        ),
    )
    return (
        1 - progress if config.lr_scheduler == "linear" else (1 + math.cos(math.pi * progress)) / 2
    )


def evaluate_prepared(model, adapter, config, root, manifest):
    from omnifactory.training.engine import load_tensors

    devices = [torch.cuda.current_device()] if config.device == "cuda" else []
    modes = [(m, m.training) for m in model.modules()]
    total = None
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(config.seed)
            for entry in manifest["entries"]:
                payload = load_tensors(root, entry)
                result = adapter.forward(model, adapter.to_device(payload, model, config)).losses
                from omnifactory.training.engine import entry_expectations

                result.validate_expected(entry_expectations([entry]))
                total = (
                    result.detached()
                    if total is None
                    else LossBundle.combine([total, result.detached()])
                )
    finally:
        for module, training in modes:
            module.training = training
    if total is None or not math.isfinite(float(total.loss)):
        raise OmniFactoryError("E_EVALUATION", "empty/nonfinite validation objective")
    return {
        "loss": float(total.loss),
        **total.report(),
        "scope": "held-out native objectives; not perceptual quality",
    }


def retain_checkpoints(root, limit, *, protected=()):
    if limit is None:
        return
    protected = {Path(p).resolve() for p in protected if p}
    paths = sorted(p for p in root.glob("checkpoint-????????") if (p / "checkpoint.json").is_file())
    for path in paths:
        if len(paths) <= limit:
            break
        if path.resolve() not in protected:
            shutil.rmtree(path)
            paths = [p for p in paths if p != path]
