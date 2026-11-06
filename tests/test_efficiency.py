import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from omnifactory.datasets.demo import create_demo
from omnifactory.training.benchmark import benchmark
from omnifactory.training.config import TrainConfig
from omnifactory.training.efficiency import StepMeasurement, batch_slots, summarize_steps
from omnifactory.training.tiny import create_tiny


def test_counts_real_padding_and_native_patch_slots():
    batch = {"attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])}
    assert batch_slots(batch) == (6, 8)
    # Raw codec channels must not inflate MiMo's LLM-patch throughput.
    batch["input_ids"] = torch.ones(2, 9, 16)
    assert batch_slots(batch) == (6, 8)
    assert batch_slots({"examples": [batch, {"attention_mask": torch.ones(1, 3)}]}) == (9, 11)
    assert batch_slots({"target_mel": torch.ones(2, 80, 20)}) == (0, 0)
    assert batch_slots(
        {"codes": torch.zeros(1, 17, 2270), "voice_state": {"embeddings": torch.zeros(51, 1, 1, 8)}}
    ) == (2321, 2321)
    assert batch_slots(
        {
            "text_mask": torch.tensor([[1, 1, 0, 0, 0]]),
            "audio_mask": torch.tensor([[0, 0, 1, 1, 0]]),
            "audio_feats": torch.ones(1, 5, 4, 64),
        }
    ) == (4, 5)
    with pytest.raises(ValueError, match="same 2D"):
        batch_slots({"text_mask": torch.ones(1, 5), "audio_mask": torch.ones(1, 4)})


def test_summary_uses_total_time_not_mean_of_rates():
    rows = []
    for seconds, count in [(1.0, 2), (3.0, 6)]:
        meter = StepMeasurement("cpu").start()
        meter.observe({"attention_mask": torch.ones(count, 4)}, count)
        row = meter.finish()
        row["step_seconds"] = seconds
        rows.append(row)
    report = summarize_steps(rows)
    assert report["examples_per_second"] == 2
    assert report["llm_slots_per_second"] == 8
    assert report["llm_padding_fraction"] == 0
    assert report["cuda_peak_allocated_bytes"] is None
    assert report["cuda_peak_reserved_bytes"] is None


def test_real_benchmark_covers_identical_examples_and_excludes_warmup(tmp_path):
    torch.set_num_threads(1)
    create_tiny(tmp_path / "model")
    create_demo(tmp_path / "data")
    cfg = TrainConfig(
        model=str(tmp_path / "model"),
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "unused",
        max_length=512,
        gradient_accumulation_steps=2,
        components={"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.001}},
        budget={
            "max_examples": 2,
            "max_llm_tokens": 1024,
            "max_vision_patches": 256,
            "max_audio_frames": 400,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 1000000,
        },
    )
    reports = {}
    for mode in ("source", "modal"):
        root = tmp_path / mode
        result = benchmark(cfg, root, batching=mode, warmup_epochs=1, repeats=2)
        rows = [json.loads(s) for s in (root / "measurements.jsonl").read_text().splitlines()]
        measured = [r for r in rows if not r["warmup"]]
        assert result["summary"]["steps"] == len(measured)
        assert result["summary"]["examples"] == 24
        assert len(rows) == len(measured) * 3 / 2
        assert all(r["loss"] > 0 and r["grad_norm"] > 0 for r in rows)
        assert all(0 <= r["llm_padding_fraction"] < 1 for r in rows)
        schedule = json.loads((root / "schedule.json").read_text())
        ids = [tuple(k) for w in schedule for b in w["microbatches"] for k in b["example_ids"]]
        assert len(set(ids)) == len(ids) == 12
        reports[mode] = result
    assert reports["modal"]["data_fingerprint"] == reports["source"]["data_fingerprint"]
    assert (
        reports["modal"]["summary"]["useful_llm_slots"]
        == reports["source"]["summary"]["useful_llm_slots"]
    )
    assert not cfg.output_dir.exists()


def test_nonnegative_memory_fit_and_out_of_scope_limits():
    import numpy as np

    from omnifactory.errors import OmniFactoryError
    from omnifactory.planning.batching import (
        MemoryProfile,
        ModalBatchPlanner,
        ResourceBudget,
        SampleCost,
    )
    from omnifactory.training.calibration import fit_nonnegative

    x = np.array([[1, 1, 0], [1, 3, 1], [1, 2, 4], [1, 5, 8]], dtype=float)
    y = x @ np.array([100, 20, 5])
    fitted = fit_nonnegative(x, y)
    assert (fitted >= 0).all()
    np.testing.assert_allclose(x @ fitted, y, atol=0.001)
    profile = MemoryProfile(
        profile_id="synthetic-unit-test",
        contract_id="c",
        hardware="test",
        provenance="synthetic",
        fixed_bytes=100,
        bytes_per_llm_token=20,
        bytes_per_vision_patch=5,
        bytes_per_audio_frame=0,
        bytes_per_codec_frame=0,
        max_calibrated_costs={"allocated_llm_tokens": 10},
        calibrated_modalities=(("text",),),
    )
    budget = ResourceBudget(
        max_examples=4,
        max_llm_tokens=100,
        max_vision_patches=100,
        max_audio_frames=100,
        max_codec_frames=100,
        max_decoded_media_bytes=10000,
    )
    planner = ModalBatchPlanner(budget, memory_profile=profile)
    row = SampleCost(
        sample_id="a", prediction_id="p", contract_id="c", modalities=("text",), llm_tokens=8
    )
    assert len(list(planner.plan([row, row.model_copy(update={"sample_id": "b"})]))) == 2
    with pytest.raises(OmniFactoryError, match="uncalibrated"):
        list(planner.plan([row.model_copy(update={"llm_tokens": 11})]))
    with pytest.raises(OmniFactoryError, match="uncalibrated modalities"):
        planner.cost([row.model_copy(update={"modalities": ("audio",)})])


def test_calibration_holds_out_shapes_and_rejects_underestimates(tmp_path):
    """Synthetic measurements exercise the estimator, not hardware acceptance."""
    from omnifactory.errors import OmniFactoryError
    from omnifactory.training.calibration import calibrate

    root = tmp_path / "synthetic-measurements"
    root.mkdir()
    cfg = TrainConfig(
        model="synthetic",
        revision="a" * 40,
        dataset=root / "data.jsonl",
        output_dir=root / "run",
        device="cuda",
        dtype="bfloat16",
        components={"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.001}},
        budget={
            "max_examples": 2,
            "max_llm_tokens": 1000,
            "max_vision_patches": 100,
            "max_audio_frames": 100,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 1000000,
        },
    )
    report = {
        "environment": {"device": "cuda", "gpu": "synthetic-unit-test"},
        "repeats": 2,
        "contract_id": "test",
        "source_fingerprint": "test",
        "cases": {},
    }
    schedule, observations = [], []
    for i in range(12):
        tokens = (i + 1) * 10
        cost = {
            "examples": 1,
            "allocated_llm_tokens": tokens,
            "allocated_attention_pairs": tokens**2,
            "allocated_vision_patches": 0,
            "allocated_audio_frames": 0,
            "allocated_codec_frames": 0,
            "allocated_latent_elements": 0,
            "allocated_stream_events": 0,
        }
        schedule.append({"case": i, "microbatches": [{"cost": cost, "modalities": ["text"]}]})
        peak = 1_000_000 + tokens * 1000
        report["cases"][str(i)] = {"cuda_peak_allocated_bytes": peak}
        observations.extend(
            {"case": i, "epoch": e, "warmup": False, "cuda_peak_allocated_bytes": peak}
            for e in [1, 2]
        )

    def save():
        for name, value in [
            ("benchmark.json", report),
            ("schedule.json", schedule),
            ("config.json", cfg.model_dump(mode="json")),
            ("tested-code.json", {}),
        ]:
            (root / name).write_text(json.dumps(value))
        (root / "measurements.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in observations)
        )

    save()
    accepted = calibrate(root, tmp_path / "accepted.json")
    assert accepted["accepted"] and accepted["heldout_cases"] == 4
    assert all(c["estimated_bytes"] >= c["measured_bytes"] for c in accepted["checks"])
    # Cold cache fills can peak above steady-state repeats. They must affect
    # memory bounds while leaving measured-throughput summaries unchanged.
    observations.append(
        {"case": 0, "epoch": 0, "warmup": True, "cuda_peak_allocated_bytes": 3_000_000}
    )
    save()
    cold = calibrate(root, tmp_path / "cold.json")
    assert cold["checks"][0]["measured_bytes"] == 3_000_000
    observations.pop()
    # Shape case 1 is held out; it cannot influence fit coefficients or workspace.
    report["cases"]["1"]["cuda_peak_allocated_bytes"] *= 20
    for row in observations:
        if row["case"] == 1:
            row["cuda_peak_allocated_bytes"] *= 20
    save()
    with pytest.raises(OmniFactoryError, match="E_CALIBRATION_VALIDATION"):
        calibrate(root, tmp_path / "rejected.json")
    rejected = json.loads((tmp_path / "rejected.json").read_text())
    assert not rejected["accepted"]
    assert rejected["profile"]["fixed_bytes"] == accepted["profile"]["fixed_bytes"]
    assert rejected["profile"]["workspace_bytes"] == accepted["profile"]["workspace_bytes"]
