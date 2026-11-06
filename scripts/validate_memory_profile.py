"""Use an accepted profile in real training and check fail-closed recipe matching."""

import argparse
import json
from pathlib import Path

from omnifactory.errors import OmniFactoryError
from omnifactory.training.calibration import configured_planner
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    artifact = json.loads((root / "memory-profile.json").read_text())
    cfg = TrainConfig.model_validate(json.loads((root / "benchmark/config.json").read_text()))
    cfg = cfg.model_copy(
        update={
            "memory_profile": root / "memory-profile.json",
            "output_dir": root / "profiled-training",
            "max_steps": 2,
            "save_steps": 1,
        }
    )
    cap = max(r["estimated_bytes"] for r in artifact["checks"])
    cfg = cfg.model_copy(update={"budget": cfg.budget.model_copy(update={"max_gpu_bytes": cap})})
    result = train(cfg)
    metrics = [
        json.loads(line) for line in (cfg.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(r["cuda_peak_allocated_bytes"] <= cap for r in metrics)
    manifest = json.loads((cfg.output_dir / "prepared/manifest.json").read_text())
    bad = cfg.model_copy(update={"gradient_checkpointing": not cfg.gradient_checkpointing})
    try:
        configured_planner(bad, manifest["contract_id"], manifest["source_fingerprint"])
    except OmniFactoryError as error:
        assert "E_MEMORY_PROFILE" in str(error)
    else:
        raise AssertionError("Changed training recipe reused an incompatible profile")
    report = {
        "training": result,
        "gpu_budget_bytes": cap,
        "measured_peaks": [r["cuda_peak_allocated_bytes"] for r in metrics],
        "changed_recipe_rejected": True,
    }
    (root / "profile-training-acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
