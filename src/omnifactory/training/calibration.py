"""Bounded, environment-specific nonnegative memory estimates with held-out checks."""

import json
from math import ceil
from pathlib import Path

import numpy as np

from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import MemoryProfile, ModalBatchPlanner
from omnifactory.training.artifacts import digest, write_json
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.efficiency import environment

FEATURES = {
    "allocated_llm_tokens": "bytes_per_llm_token",
    "allocated_vision_patches": "bytes_per_vision_patch",
    "allocated_audio_frames": "bytes_per_audio_frame",
    "allocated_codec_frames": "bytes_per_codec_frame",
    "allocated_attention_pairs": "bytes_per_attention_pair",
    "allocated_latent_elements": "bytes_per_latent_element",
    "allocated_stream_events": "bytes_per_stream_event",
}


def recipe_signature(config, source_hash):
    settings = config.execution_contract()
    for key in ("dataset", "dataset_root", "budget", "window_size", "memory_profile"):
        settings.pop(key, None)
    return digest({"config": settings, "source": source_hash})


def case_features(case):
    rows = []
    for batch in case["microbatches"]:
        cost = dict(batch["cost"])
        rows.append(cost)
    return {key: max(r[key] for r in rows) for key in ["examples", *FEATURES]}


def fit_nonnegative(x, y):
    """Cyclic coordinate descent on scaled nonnegative least squares.

    No scipy dependency or unconstrained negative memory slopes. Collinear axes
    are allowed; coefficients are predictive, not causal per-component costs.
    """
    scale = np.maximum(x.max(axis=0), 1)
    a = x / scale
    beta = np.zeros(a.shape[1], dtype=np.float64)
    residual = y.copy()
    for _ in range(10000):
        before = beta.copy()
        for j in range(len(beta)):
            norm = a[:, j] @ a[:, j]
            if norm == 0:
                continue
            new = max(0.0, beta[j] + (a[:, j] @ residual) / norm)
            residual -= a[:, j] * (new - beta[j])
            beta[j] = new
        if np.max(np.abs(beta - before)) <= max(1, y.max()) * 1e-9:
            break
    return beta / scale


def calibrate(benchmark_dir, output, *, safety_fraction=0.15):
    """Fit on 2/3 of shape cases; hold out complete cases, never repeat measurements."""
    from omnifactory.training.config import TrainConfig

    root, output = Path(benchmark_dir), Path(output)
    if output.exists():
        raise OmniFactoryError("E_OUTPUT_EXISTS", str(output))
    report = json.loads((root / "benchmark.json").read_text())
    if report["environment"]["device"] != "cuda":
        raise OmniFactoryError(
            "E_CALIBRATION", "memory calibration requires real CUDA observations"
        )
    if report["repeats"] < 2 or not 0 <= safety_fraction <= 1:
        raise OmniFactoryError("E_CALIBRATION", "need >=2 measured repeats and a valid margin")
    schedule = json.loads((root / "schedule.json").read_text())
    if len(schedule) < 9:
        raise OmniFactoryError(
            "E_CALIBRATION", "need at least nine distinct optimizer-window cases"
        )
    observations = [
        json.loads(line) for line in (root / "measurements.jsonl").read_text().splitlines()
    ]
    cold_and_warm_peaks = []
    for case in schedule:
        all_rows = [r for r in observations if r["case"] == case["case"]]
        cold_and_warm_peaks.append(max(r["cuda_peak_allocated_bytes"] for r in all_rows))
        rows = [r for r in observations if not r["warmup"] and r["case"] == case["case"]]
        if len(rows) != report["repeats"] or len({r["epoch"] for r in rows}) != len(rows):
            raise OmniFactoryError("E_CALIBRATION", "missing or duplicate measured repeats")
        if (
            max(r["cuda_peak_allocated_bytes"] for r in rows)
            != report["cases"][str(case["case"])]["cuda_peak_allocated_bytes"]
        ):
            raise OmniFactoryError("E_CALIBRATION", "summary differs from recorded peaks")
    features = [case_features(c) for c in schedule]
    x = np.array([[1, *(row[k] for k in FEATURES)] for row in features], dtype=np.float64)
    y = np.array(
        cold_and_warm_peaks,
        dtype=np.float64,
    )
    if not np.isfinite(y).all() or (y <= 0).any():
        raise OmniFactoryError("E_CALIBRATION", "invalid measured CUDA peaks")
    # Deterministic split after sorting cases by input shape; repeated measurements
    # of one shape cannot enter both partitions. Duplicate shapes are kept together.
    groups = {}
    for i, f in enumerate(features):
        groups.setdefault(tuple(f.values()), []).append(i)
    if len(groups) < 9:
        raise OmniFactoryError(
            "E_CALIBRATION", "need >=9 distinct cost vectors, not repeated shapes"
        )
    train, heldout = [], []
    for i, (_, indices) in enumerate(sorted(groups.items())):
        (heldout if i % 3 == 1 else train).extend(indices)
    beta = np.ceil(fit_nonnegative(x[train], y[train])).astype(np.int64)
    workspace = ceil(max(0.0, float((y[train] - x[train] @ beta).max())))
    prediction = np.ceil((x @ beta + workspace) * (1 + safety_fraction))
    checks = [
        {
            "case": int(i),
            "partition": "heldout" if i in heldout else "fit",
            "measured_bytes": int(y[i]),
            "estimated_bytes": int(prediction[i]),
            "relative_error": float((prediction[i] - y[i]) / y[i]),
        }
        for i in range(len(schedule))
    ]
    passed = bool((prediction[heldout] >= y[heldout]).all())
    config = TrainConfig.model_validate(json.loads((root / "config.json").read_text()))
    profile = MemoryProfile(
        profile_id=digest({"benchmark": report, "coefficients": beta.tolist()}),
        contract_id=report["contract_id"],
        hardware=report["environment"]["gpu"],
        provenance="calibrated",
        fixed_bytes=int(beta[0]),
        workspace_bytes=workspace,
        **{field: int(beta[i + 1]) for i, field in enumerate(FEATURES.values())},
        safety_fraction=safety_fraction,
        max_calibrated_costs={k: max(f[k] for f in features) for k in features[0]},
        calibrated_modalities=tuple(
            sorted({tuple(b["modalities"]) for c in schedule for b in c["microbatches"]})
        ),
    )
    result = {
        "format": "omnifactory-memory-calibration-v1",
        "accepted": passed,
        "profile": profile.model_dump(mode="json"),
        "environment": report["environment"],
        "recipe_signature": recipe_signature(config, report["source_fingerprint"]),
        "runtime_hashes": json.loads((root / "tested-code.json").read_text()),
        "checks": checks,
        "evidence_hashes": {
            name: digest(json.loads((root / name).read_text()))
            for name in ("benchmark.json", "schedule.json", "config.json", "tested-code.json")
        },
        "peak_scope": "maximum over cold warmup and measured repeats; throughput excludes warmup",
        "heldout_cases": len(heldout),
        "fit_cases": len(train),
        "coefficient_matrix_rank": int(np.linalg.matrix_rank(x[train])),
        "oom_guarantee": False,
        "scope": "allocated-memory estimate within measured cost caps and modality combinations; reserved memory is reported separately",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    if not passed:
        raise OmniFactoryError(
            "E_CALIBRATION_VALIDATION", "held-out peak underestimated; rejected report saved"
        )
    return result


def configured_planner(config, contract_id, source_hash, *, seed=None):
    profile = None
    if config.memory_profile:
        artifact = json.loads(config.memory_profile.read_text())
        if artifact.get("format") != "omnifactory-memory-calibration-v1" or not artifact.get(
            "accepted"
        ):
            raise OmniFactoryError("E_MEMORY_PROFILE", "requires an accepted calibration artifact")
        if artifact["recipe_signature"] != recipe_signature(config, source_hash):
            raise OmniFactoryError(
                "E_MEMORY_PROFILE", "model/processor/training recipe differs from calibration"
            )
        if artifact["environment"] != environment(config.device):
            raise OmniFactoryError(
                "E_MEMORY_PROFILE", "hardware, precision or package environment changed"
            )
        if artifact["runtime_hashes"] != runtime_hashes():
            raise OmniFactoryError("E_MEMORY_PROFILE", "execution source changed; recalibrate")
        profile = MemoryProfile.model_validate({**artifact["profile"], "contract_id": contract_id})
    from omnifactory.adapters.runtime import task_batch_capability

    return ModalBatchPlanner(
        config.budget,
        capability=task_batch_capability(config),
        timed=config.duplex_schedule is not None,
        memory_profile=profile,
        seed=config.seed if seed is None else seed,
        window_size=config.window_size,
        packing=config.packing != "none",
    )
