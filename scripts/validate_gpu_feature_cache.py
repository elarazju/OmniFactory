"""Full-weight CUDA frozen-cache acceptance with a trainable vision projector."""

import argparse
import json
import os
from pathlib import Path

import torch

from omnifactory.adapters.qwen2_5 import Qwen25Adapter, load_thinker
from omnifactory.planning.adaptation import ComponentPolicy
from omnifactory.training.artifacts import write_json
from omnifactory.training.config import TrainConfig
from omnifactory.training.efficiency import StepMeasurement
from omnifactory.training.engine import load_tensors
from omnifactory.training.feature_cache import cache_stats, install_feature_cache
from omnifactory.training.policy import apply_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.benchmark_dir.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = TrainConfig.model_validate(json.loads((root / "config.json").read_text()))
    cfg = cfg.model_copy(
        update={
            "components": {"vision_projector": ComponentPolicy(mode="full", lr=0.001)},
            "feature_cache": None,
            "packing": "none",
        }
    )
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    adapter = Qwen25Adapter.from_config(cfg)
    model, _, _ = apply_policy(load_thinker(cfg), cfg)
    model.train()
    rows = [
        load_tensors(root / "prepared", e)
        for e in manifest["entries"]
        if e["sample_id"] in ("image-5", "audio-4")
    ]
    assert len(rows) == 2
    batch = adapter.to_device(adapter.collate_padded(rows), model, cfg)

    def step():
        model.zero_grad(set_to_none=True)
        meter = StepMeasurement("cuda").start()
        output = adapter.forward(model, batch)
        output.loss.backward()
        metric = meter.finish()
        metric["loss"] = float(output.loss.detach())
        grads = {
            n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        assert all(g.isfinite().all() for g in grads.values())
        return metric, grads

    step()  # warmup
    before = [step() for _ in range(3)]
    install_feature_cache(model, cfg.model_copy(update={"feature_cache": out / "features"}))
    cold, cold_gradients = step()
    after = [step() for _ in range(3)]
    reference = before[-1][1]
    baseline_jitter = max(
        float((g.float() - reference[n].float()).abs().max())
        for _, grads in before
        for n, g in grads.items()
    )
    print(json.dumps({"baseline_gradient_jitter": baseline_jitter}), flush=True)
    assert baseline_jitter == 0, "deterministic reference is required"
    maximum = 0.0
    for metric, grads in [(cold, cold_gradients), *after]:
        assert metric["loss"] == before[-1][0]["loss"]
        for n, g in grads.items():
            maximum = max(maximum, float((g.float() - reference[n].float()).abs().max()))
            torch.testing.assert_close(g, reference[n], rtol=0, atol=0)
    from omnifactory.training.benchmark import runtime_hashes
    from omnifactory.training.efficiency import environment

    result = dict(
        runtime_hashes=runtime_hashes(),
        environment=environment("cuda"),
        passed=True,
        max_gradient_delta=maximum,
        cold_gradient_verified=True,
        cache=cache_stats(model),
        uncached=[r[0] for r in before],
        cold=cold,
        cached=[r[0] for r in after],
        model=cfg.model,
        projector_trainable=True,
    )
    # These timings are forward/backward only; the benchmark CLI supplies optimizer throughput.
    result["median_uncached_seconds"] = sorted(r[0]["step_seconds"] for r in before)[1]
    result["median_cached_seconds"] = sorted(r[0]["step_seconds"] for r in after)[1]
    write_json(out / "acceptance.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("uncached", "cached")}))


if __name__ == "__main__":
    main()
