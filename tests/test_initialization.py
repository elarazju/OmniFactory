import json

import pytest

pytest.importorskip("torch")
from typer.testing import CliRunner

from omnifactory.cli import app
from omnifactory.environment import recipe_checks
from omnifactory.initialization import initialize
from omnifactory.training.config import read_config


@pytest.mark.parametrize(
    "recipe,adapter", [("qwen2.5-omni-3b", "qwen2_5_omni"), ("voxcpm2", "voxcpm2")]
)
def test_packaged_init_resolves_paths_and_preflight(tmp_path, monkeypatch, recipe, adapter):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    result = CliRunner().invoke(
        app, ["init", "--recipe", recipe, "--output", str(tmp_path / "first")]
    )
    assert result.exit_code == 0, result.output
    project = json.loads(result.output)
    cfg = read_config(tmp_path / "first/train.yaml")
    assert cfg.adapter == adapter and cfg.dataset.is_file()
    assert len(cfg.revision) == 40 and project["synthetic_data"]
    assert recipe_checks(cfg)["errors"] == []
    cfg.output_dir.mkdir()
    cfg.dataset.unlink()
    errors = recipe_checks(cfg)["errors"]
    assert any("output_dir" in e for e in errors) and any("train:" in e for e in errors)


def test_init_custom_dataset_and_missing_pin(tmp_path):
    from omnifactory.errors import OmniFactoryError

    with pytest.raises(OmniFactoryError, match="E_REVISION"):
        initialize(tmp_path / "bad", recipe="voxcpm2", model="other/remote")
    with pytest.raises(OmniFactoryError, match="E_RECIPE_DATA"):
        initialize(tmp_path / "bad", recipe="voxcpm2", dataset=tmp_path / "train.jsonl")
    initialize(
        tmp_path / "custom",
        recipe="voxcpm2",
        model="./missing-model",
        dataset=tmp_path / "train.jsonl",
        validation_dataset=tmp_path / "val.jsonl",
    )
    cfg = read_config(tmp_path / "custom/train.yaml")
    assert cfg.dataset == tmp_path / "train.jsonl" and cfg.revision is None
    assert any("local model directory" in e for e in recipe_checks(cfg)["errors"])


def test_preflight_rejects_split_leak_and_missing_media(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    initialize(tmp_path / "first", recipe="qwen2.5-omni-3b")
    cfg = read_config(tmp_path / "first/train.yaml")
    rows = [json.loads(s) for s in cfg.validation_dataset.read_text().splitlines()]
    rows[0]["provenance"]["group_id"] = "train-000"
    cfg.validation_dataset.write_text("\n".join(map(json.dumps, rows)))
    assert any("overlaps" in e for e in recipe_checks(cfg)["errors"])
    rows[0]["provenance"]["group_id"] = "heldout"
    cfg.validation_dataset.write_text("\n".join(map(json.dumps, rows)))
    next((cfg.dataset_root / "media").glob("*.wav")).unlink()
    assert any("E_MEDIA_MISSING" in e for e in recipe_checks(cfg)["errors"])


@pytest.mark.parametrize("recipe", ["qwen2.5-omni-3b", "voxcpm2"])
def test_initialized_recipe_prepares_with_native_encoder(tmp_path, recipe):
    from omnifactory.training.prepare import prepare

    if recipe == "voxcpm2":
        from omnifactory.training.tiny_voxcpm2 import create_tiny_voxcpm2 as create
    else:
        from omnifactory.training.tiny import create_tiny as create
    create(tmp_path / "model")
    initialize(tmp_path / "project", recipe=recipe, model=str(tmp_path / "model"))
    cfg = read_config(tmp_path / "project/train.yaml").model_copy(
        update={"device": "cpu", "dtype": "float32", "max_length": 512}
    )
    manifest = prepare(cfg, tmp_path / "prepared")
    assert len(manifest["entries"]) == (2 if recipe == "voxcpm2" else 12)
    if recipe == "voxcpm2":
        entry = manifest["entries"][0]
        assert set(entry["normalization_counts"]) == {"latent_frames", "stop_symbols"}
