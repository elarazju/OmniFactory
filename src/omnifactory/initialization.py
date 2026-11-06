"""Packaged pretrained first-run recipes: usable from a wheel, without a checkout."""

from importlib.resources import files
from pathlib import Path

import yaml

from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import new_directory, write_json

RECIPES = ("qwen2.5-omni-3b", "voxcpm2")


def initialize(
    output,
    *,
    recipe,
    model=None,
    revision=None,
    dataset=None,
    dataset_root=None,
    validation_dataset=None,
    validation_root=None,
):
    if recipe not in RECIPES:
        raise OmniFactoryError("E_RECIPE", f"choose one of {', '.join(RECIPES)}")
    if (dataset is None) != (validation_dataset is None):
        raise OmniFactoryError("E_RECIPE_DATA", "supply both --dataset and --validation-dataset")
    if dataset is None and (dataset_root is not None or validation_root is not None):
        raise OmniFactoryError("E_RECIPE_DATA", "media roots require your dataset paths")
    cfg = yaml.safe_load(files("omnifactory").joinpath(f"recipes/{recipe}.yaml").read_text())
    if model:
        candidate = Path(model).expanduser()
        local = candidate.is_dir() or str(model).startswith((".", "/", "~"))
        cfg.update(
            model=str(candidate.resolve()) if local else model, revision=None if local else revision
        )
        if not local and not revision:
            raise OmniFactoryError(
                "E_REVISION", "a custom remote model requires --revision (full commit SHA)"
            )
    elif revision:
        cfg["revision"] = revision
    from omnifactory.training.config import TrainConfig

    TrainConfig.model_validate(cfg)
    output = Path(output).resolve()
    with new_directory(output) as root:
        if dataset is None:
            if recipe == "voxcpm2":
                create_vox_fixture(root / "data")
            else:
                from omnifactory.datasets.demo import create_demo

                create_demo(root / "data")
        else:
            cfg.update(
                dataset=str(Path(dataset).resolve()),
                dataset_root=str(Path(dataset_root or Path(dataset).parent).resolve()),
                validation_dataset=str(Path(validation_dataset).resolve()),
                validation_root=str(
                    Path(validation_root or Path(validation_dataset).parent).resolve()
                ),
            )
        (root / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        result = dict(
            recipe=recipe,
            config=str(output / "train.yaml"),
            model=cfg["model"],
            revision=cfg["revision"],
            synthetic_data=dataset is None,
            purpose="pretrained framework workflow; no quality improvement claim",
            stages=["doctor", "prepare", "train", "resume", "export", "inference"],
            required_environment={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        )
        write_json(root / "project.json", result)
    return result


def create_vox_fixture(output):
    """Native TTS-shaped synthetic fixtures; tone waveforms are not spoken transcripts."""
    from omnifactory.datasets.flow_demo import create_flow_demo

    create_flow_demo(output)
    for split in ("train", "validation"):
        rows = []
        for sample in iter_samples(output / f"{split}.jsonl"):
            row = sample.model_dump(mode="json")
            text = "Hello world."
            row["messages"][0]["content"][0]["text"] = text
            row["messages"][1]["content"].append({"id": "transcript", "type": "text", "text": text})
            row["predictions"][0]["target_block_ids"].append("transcript")
            if sample.id == "flow-1":
                row["predictions"][0]["conditions"] = []
            row["speech_pairs"] = [{"text_block_id": "transcript", "audio_block_id": "audio"}]
            rows.append(CanonicalSample.model_validate(row).model_dump_json())
        (output / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
    write_json(
        output / "dataset_info.json",
        dict(
            name="voxcpm2-first-run",
            synthetic=True,
            train=2,
            validation=1,
            purpose="framework execution only; tones do not contain the synthetic transcript",
        ),
    )
