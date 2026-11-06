import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch
import yaml
from safetensors.torch import save_file

from omnifactory.datasets.demo import create_demo
from omnifactory.training.config import TrainConfig
from omnifactory.training.tiny import create_tiny

SCRIPT = Path(__file__).parents[1] / "scripts/benchmark_training_optimization.py"


def module():
    spec = importlib.util.spec_from_file_location("training_optimization_benchmark", SCRIPT)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_weight_comparison_rejects_wrong_values_and_parameter_set(tmp_path):
    compare = module().compare_weights
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    save_file({"x": torch.ones(4)}, a / "trainable.safetensors")
    save_file({"x": torch.ones(4) + 1e-5}, b / "trainable.safetensors")
    assert compare(a, b, exact=True)["status"] == "failed"
    assert compare(a, b, exact=False)["status"] == "passed"
    save_file({"y": torch.ones(4)}, b / "trainable.safetensors")
    with pytest.raises(ValueError, match="parameter sets"):
        compare(a, b, exact=False)


def test_cache_benchmark_uses_fresh_arms_and_cross_cache_resume(tmp_path):
    torch.set_num_threads(2)
    create_tiny(tmp_path / "model")
    create_demo(tmp_path / "data")
    cfg = TrainConfig(
        model=str(tmp_path / "model"),
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "unused",
        max_steps=8,
        save_steps=4,
        max_length=512,
        components={"language_model": dict(mode="lora", rank=2, alpha=4, lr=0.001)},
        budget=dict(
            max_examples=2,
            max_llm_tokens=1024,
            max_vision_patches=128,
            max_audio_frames=400,
            max_codec_frames=1,
            max_decoded_media_bytes=1000000,
        ),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    root = tmp_path / "comparison"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(path),
            "--output",
            str(root),
            "--optimization",
            "cache",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "passed" and report["max_weight_delta"] == 0
    assert report["exact_resume"] == "passed"
    assert report["resources"]["optimized"]["cache"]["hits"] > 0
    assert report["resources"]["baseline"]["after_first_epoch"]["steps"] > 0
    resumed = json.loads((root / "resumed/config.json").read_text())
    assert resumed["resume_from"].endswith("baseline/checkpoint-00000004")
