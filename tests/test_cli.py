import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from omnifactory.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_doctor_reports_actual_implementation_scope():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["training_backend_implemented"] is True
    assert report["gpu_tested"] is True
    assert {"qwen3-speech-v1", "voxcpm2-speech-v2", "bagel-dit-v3"} <= set(
        report["gpu_verification_scope"]
    )
    assert "packages" in report and report["cuda"] is None


def test_models_are_not_marked_as_training_verified():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert {r["family"] for r in rows} == {
        "qwen_omni",
        "minicpm_o",
        "mimo_audio",
        "voxcpm",
        "bagel",
        "qwen3_tts",
        "cosyvoice3",
        "qwen_image",
        "personaplex",
    }
    assert all("training_verified" not in r for r in rows)
    assert all(r["verification_records"] for r in rows)
    duplex = next(r for r in rows if r["key"] == "personaplex")
    assert {e["id"] for e in duplex["verification_records"]} == {
        "personaplex-v1",
        "personaplex-realtime-v2",
    }
    editing = next(r for r in rows if r["key"] == "qwen_image_edit")
    assert {e["kind"] for e in editing["verification_records"]} == {"native", "http"}
    assert editing["runtime_verification"] == "tiny_random_cpu"
    cosy = next(r for r in rows if r["key"] == "cosyvoice3")
    assert len(cosy["verification_records"]) == 4
    pending = next(r for r in rows if r["key"] == "qwen3_tts")
    assert {e["kind"] for e in pending["verification_records"]} == {"native", "http"}
    assert pending["status"] == "experimental_adapter"
    assert all(e["model_key"] == r["key"] for r in rows for e in r["verification_records"])
    assert (
        next(r for r in rows if r["key"] == "qwen2_5_omni_3b")["runtime_verification"]
        == "tiny_random_cpu"
    )


def test_unknown_model_is_actionable():
    result = runner.invoke(app, ["models", "show", "missing"])
    assert result.exit_code == 2
    assert "E_MODEL" in result.output
    assert "Traceback" not in result.output


def test_validate_examples(examples_path):
    result = runner.invoke(app, ["data", "validate", str(examples_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["valid"] == 4


def test_plan_sample(examples_path):
    result = runner.invoke(
        app,
        [
            "data",
            "plan",
            str(examples_path),
            "--sample-id",
            "speech-qa-001",
            "--model",
            "minicpm_o_4_5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["stages"] == ["llm", "tts", "code2wav"]


def test_write_inference_views(examples_path, tmp_path):
    output = tmp_path / "views.jsonl"
    result = runner.invoke(app, ["data", "inference", str(examples_path), "--output", str(output)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 5
    assert list(rows[0]["assets"]) == ["question"]


def test_refuse_overwriting_input(examples_path):
    result = runner.invoke(
        app, ["data", "inference", str(examples_path), "--output", str(examples_path)]
    )
    assert result.exit_code == 2
    assert "E_OUTPUT" in result.output


def test_manifest_late_error_preserves_previous_output(raw_sample, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(raw_sample) + "\n")
    output = tmp_path / "result.jsonl"
    output.write_text("previous result")
    result = runner.invoke(app, ["data", "manifest", str(source), "--output", str(output)])
    assert result.exit_code == 2
    assert "E_MEDIA_MISSING" in result.output
    assert output.read_text() == "previous result"


def test_batch_command_no_sample_loss(tmp_path):
    output = tmp_path / "batches.jsonl"
    result = runner.invoke(
        app,
        [
            "batch",
            "plan",
            "--costs",
            str(ROOT / "examples/costs.jsonl"),
            "--config",
            str(ROOT / "examples/batch-budget.yaml"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["examples"] == 6
    plans = [json.loads(line) for line in output.read_text().splitlines()]
    assert sum(len(p["example_ids"]) for p in plans) == 6


def test_batch_late_error_does_not_publish_partial_plan(tmp_path):
    source = tmp_path / "costs.jsonl"
    first = {
        "sample_id": "a",
        "prediction_id": "p0",
        "contract_id": "demo",
        "modalities": ["text"],
        "llm_tokens": 10,
    }
    source.write_text(
        json.dumps(first) + "\n" + json.dumps(dict(first, sample_id="b", llm_tokens=999999)) + "\n"
    )
    config = yaml.safe_load((ROOT / "examples/batch-budget.yaml").read_text())
    config["window_size"] = 1
    settings = tmp_path / "config.yaml"
    settings.write_text(yaml.safe_dump(config))
    output = tmp_path / "plan.jsonl"
    output.write_text("original")
    result = runner.invoke(
        app,
        [
            "batch",
            "plan",
            "--costs",
            str(source),
            "--config",
            str(settings),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2
    assert "E_SAMPLE_OVER_BUDGET" in result.output
    assert output.read_text() == "original"


def test_projector_plan_reports_frozen_gradient_path():
    result = runner.invoke(
        app,
        [
            "adaptation",
            "plan",
            "--config",
            str(ROOT / "examples/minicpm-projector.yaml"),
            "--inventory",
            str(ROOT / "examples/inventory.synthetic.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert "language_model" in plan["gradient_components"]
    assert len(plan["groups"]) == 1


def test_malformed_config_exits_cleanly(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("model: [")
    result = runner.invoke(
        app,
        [
            "adaptation",
            "plan",
            "--config",
            str(config),
            "--inventory",
            str(ROOT / "examples/inventory.synthetic.json"),
        ],
    )
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_environment_capture_resolves_active_distribution(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from omnifactory import environment

    monkeypatch.setattr(
        environment,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "transformers"}, version="4.57.6"),
            SimpleNamespace(metadata={"Name": "transformers"}, version="4.57.1.post1"),
        ],
    )
    monkeypatch.setattr(environment, "version", lambda name: "4.57.6")
    report = environment.capture_environment(tmp_path / "environment")
    assert report["packages"]["transformers"] == "4.57.6"
    assert (tmp_path / "environment/requirements.lock.txt").read_text() == "transformers==4.57.6\n"


def test_deploy_validation_accepts_audio_bound(monkeypatch):
    def validate(model, dataset, output, executable, deploy_config, *, max_audio_units):
        return {"max_audio_units": max_audio_units}

    monkeypatch.setattr("omnifactory.serving.validate_deployment", validate)
    result = runner.invoke(
        app,
        [
            "deploy",
            "validate",
            "--model",
            "export",
            "--dataset",
            "data.jsonl",
            "--output",
            "probe",
            "--executable",
            "vllm",
            "--deploy-config",
            "deploy.yaml",
            "--max-audio-units",
            "256",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"max_audio_units": 256}
