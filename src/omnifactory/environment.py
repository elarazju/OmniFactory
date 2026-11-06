"""Inspectable runtime diagnostics and portable version inventories; no hidden installs."""

import importlib.util
import json
import platform
import shutil
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path

from omnifactory import __version__
from omnifactory.adapters.runtime import adapter_specs
from omnifactory.capabilities import catalog
from omnifactory.data.io import atomic_writer


def diagnose(config=None):
    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "flash-attn",
        "soundfile",
        "diffusers",
        "sphn",
        "sentencepiece",
        "vllm",
        "vllm-omni",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    errors = []
    cuda = None
    if config:
        for name in ("torch", "transformers", "peft", "accelerate", "soundfile"):
            if not packages[name]:
                errors.append(f"missing {name}: install the train extra")
        if packages["transformers"] and packages["transformers"] != "4.57.6":
            errors.append(
                "training requires transformers==4.57.6; keep serving in a separate environment"
            )
        if config.quantization != "none" and not packages["bitsandbytes"]:
            errors.append("NF4 requires bitsandbytes")
        if config.adapter == "qwen_image_edit" and packages["diffusers"] != "0.39.0":
            errors.append("Qwen-Image-Edit requires diffusers==0.39.0: install the image extra")
        if config.adapter == "personaplex" and (
            packages["sphn"] != "0.1.12" or packages["sentencepiece"] != "0.2.1"
        ):
            errors.append(
                "PersonaPlex requires the personaplex extra (sphn 0.1.12, sentencepiece 0.2.1)"
            )
        if config.packing != "none" and not packages["flash-attn"]:
            errors.append("packing requires a matching FlashAttention 2 build")
        if config.device == "cuda" and packages["torch"]:
            import torch

            cuda = dict(available=torch.cuda.is_available(), runtime=torch.version.cuda)
            if not cuda["available"]:
                errors.append("CUDA requested but unavailable")
            elif config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                errors.append("CUDA device lacks BF16 support")
    preflight = recipe_checks(config) if config else None
    if preflight:
        errors.extend(preflight["errors"])
    records = catalog()["records"]
    return dict(
        version=__version__,
        python=platform.python_version(),
        platform=platform.system(),
        torch_installed=importlib.util.find_spec("torch") is not None,
        packages=packages,
        training_backend_implemented=True,
        training_scope=[
            dict(adapter=s.name, tasks=s.tasks, outputs=s.outputs) for s in adapter_specs()
        ],
        gpu_tested=bool(records),
        gpu_verification_scope=[r["id"] for r in records],
        verification="Historical per-checkpoint evidence; this does not validate the current machine",
        ffmpeg=shutil.which("ffmpeg"),
        cuda=cuda,
        preflight=preflight,
        errors=errors,
        ready=not errors,
    )


def capture_environment(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # system-site-packages can expose two distributions with the same name.
    # Resolve the first active distribution just as importlib.metadata.version
    # does, rather than letting a later system package overwrite the venv pin.
    names = {d.metadata["Name"] for d in distributions() if d.metadata.get("Name")}
    packages = {name: version(name) for name in names}
    # Versions only: never serialize credential-bearing index URLs or environment variables.
    lock = (
        "\n".join(f"{k}=={v}" for k, v in sorted(packages.items(), key=lambda kv: kv[0].lower()))
        + "\n"
    )
    (output / "requirements.lock.txt").write_text(lock)
    report = dict(
        omnifactory_version=__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        packages=packages,
        scope="Observed environment; CUDA wheels/index and native build provenance require the accompanying recipe",
    )
    with atomic_writer(output / "environment.json") as stream:
        stream.write(json.dumps(report, indent=2))
    return report


def recipe_checks(config):
    """Offline preflight: paths, pinned identity, splits and media bytes; no model loads."""
    import os
    import re
    import sqlite3
    import tempfile

    from omnifactory.data.io import iter_samples, validate_dataset
    from omnifactory.data.media import LocalMediaResolver

    errors, datasets = [], {}
    model = Path(config.model).expanduser()
    if model.is_dir():
        if not (model / "config.json").is_file():
            errors.append(f"local model lacks config.json: {model}; use a complete snapshot")
        if (
            not any(model.glob("*.safetensors"))
            and not any(model.glob("*.pt"))
            and not any(model.glob("*.bin"))
        ):
            errors.append(f"local model contains no recognized weights: {model}")
    elif config.model.startswith((".", "/", "~")):
        errors.append(f"local model directory does not exist: {model}")
    elif not re.fullmatch(r"[0-9a-f]{40}", config.revision or ""):
        errors.append("remote model requires revision: a full 40-character commit SHA")
    if (
        config.deterministic
        and config.device == "cuda"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8")
    ):
        errors.append("export CUBLAS_WORKSPACE_CONFIG=:4096:8 before deterministic CUDA training")
    if config.output_dir.exists():
        errors.append("output_dir already exists; choose a new directory, including for resume")
    for name in (
        "prepared_dataset",
        "resume_from",
        "speech_tokenizer",
        "speech_assets",
        "speech_codes",
        "speech_reference",
        "duplex_schedule",
        "duplex_text_alignment",
        "memory_profile",
    ):
        path = getattr(config, name)
        if path is not None and not path.exists():
            errors.append(f"{name} does not exist: {path}")
    if config.prepared_dataset and config.prepared_dataset.is_dir():
        try:
            metadata = json.loads((config.prepared_dataset / "manifest.json").read_text())
            if metadata.get("format") != "omnifactory-prepared-v4":
                errors.append(
                    "prepared data format is outdated; run prepare again into a new directory"
                )
        except (OSError, ValueError) as exc:
            errors.append(f"prepared manifest: {exc}")
    with (
        tempfile.TemporaryDirectory(prefix="omni-preflight-") as tmp,
        sqlite3.connect(Path(tmp) / "groups.sqlite") as db,
    ):
        db.execute(
            "CREATE TABLE groups (dataset TEXT, name TEXT, split TEXT, PRIMARY KEY(dataset,name)) WITHOUT ROWID"
        )
        for name, path, media_root in (
            ("train", config.dataset, config.dataset_root),
            ("validation", config.validation_dataset, config.validation_root),
        ):
            if path is None:
                continue
            try:
                report = validate_dataset(path)
                if report.invalid or not report.records:
                    errors.extend(f"{name}: {error}" for error in report.errors)
                    continue
                resolver = LocalMediaResolver(media_root or path.parent)
                modalities = set()
                for sample in iter_samples(path):
                    if sample.provenance.split != name:
                        raise ValueError(
                            f"{sample.id} has split {sample.provenance.split}; expected {name}"
                        )
                    key = (sample.provenance.dataset, sample.provenance.group_id)
                    previous = db.execute(
                        "SELECT split FROM groups WHERE dataset=? AND name=?", key
                    ).fetchone()
                    if previous and previous[0] != name:
                        raise ValueError(f"source group {key} overlaps training and validation")
                    db.execute("INSERT OR IGNORE INTO groups VALUES (?,?,?)", (*key, name))
                    resolver.manifest(sample)
                    modalities.update(a.modality for a in sample.assets.values())
                if "video" in modalities and (
                    not shutil.which("ffmpeg") or not shutil.which("ffprobe")
                ):
                    errors.append("video data requires ffmpeg and ffprobe on PATH")
                datasets[name] = dict(records=report.records, modalities=sorted(modalities))
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: {exc}")
    return dict(
        errors=errors,
        datasets=datasets,
        scope="offline identity, paths, data structure, group separation and media hashes; model compatibility/download and training are checked by prepare/acceptance",
    )
