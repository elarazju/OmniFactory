"""Local CPU commands. JSON output is suitable for scripts and a future Studio."""

import importlib.util
import json
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import Field, ValidationError

from omnifactory.data.io import atomic_writer, iter_samples, load_json, validate_dataset
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.processor import OmniDataProcessor
from omnifactory.data.schema import CanonicalSample, Record
from omnifactory.errors import OmniFactoryError
from omnifactory.models.registry import builtin_registry
from omnifactory.planning.adaptation import AdaptationRequest, ParameterSpec, compile_adaptation
from omnifactory.planning.batching import (
    MemoryProfile,
    ModalBatchPlanner,
    ResourceBudget,
    SampleCost,
)

app = typer.Typer(
    no_args_is_help=True, help="OmniFactory — Unified Efficient Fine-Tuning of Omni Models."
)
models = typer.Typer(no_args_is_help=True)
data = typer.Typer(no_args_is_help=True)
batch = typer.Typer(no_args_is_help=True)
adaptation = typer.Typer(no_args_is_help=True)
app.add_typer(models, name="models")
app.add_typer(data, name="data")
app.add_typer(batch, name="batch")
app.add_typer(adaptation, name="adaptation")


def handled(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (
            OmniFactoryError,
            ValidationError,
            OSError,
            yaml.YAMLError,
            json.JSONDecodeError,
        ) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc

    return wrapper


def emit(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


def find_sample(path: Path, sample_id: str) -> CanonicalSample:
    for sample in iter_samples(path):
        if sample.id == sample_id:
            return sample
    raise OmniFactoryError("E_SAMPLE", f"no sample {sample_id!r} in {path}")


def separate_output(output: Path, *inputs: Path):
    if any(output.resolve() == source.resolve() for source in inputs):
        raise OmniFactoryError("E_OUTPUT", "output must not overwrite an input file")


@app.command()
@handled
def doctor(config: Path | None = None, strict: bool = False):
    """Inspect the current environment against an optional training recipe."""
    from omnifactory.environment import diagnose
    from omnifactory.training.config import read_config

    report = diagnose(read_config(config) if config else None)
    emit(report)
    if strict and report["errors"]:
        raise typer.Exit(2)


@app.command("init")
@handled
def initialize_project(
    recipe: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
    model: str | None = None,
    revision: str | None = None,
    dataset: Path | None = None,
    dataset_root: Path | None = None,
    validation_dataset: Path | None = None,
    validation_root: Path | None = None,
):
    """Create a pretrained first-run recipe and explicit workflow fixtures."""
    from omnifactory.initialization import initialize

    if dataset is None:
        require_training()
    emit(
        initialize(
            output,
            recipe=recipe,
            model=model,
            revision=revision,
            dataset=dataset,
            dataset_root=dataset_root,
            validation_dataset=validation_dataset,
            validation_root=validation_root,
        )
    )


@models.command("list")
def list_models():
    """List architecture contracts; contract_only does not mean trainable."""
    emit([m.model_dump(mode="json") for m in builtin_registry().list()])


@models.command("show")
@handled
def show_model(model: str):
    emit(builtin_registry().get(model))


@data.command("schema")
def show_schema():
    emit(CanonicalSample.model_json_schema())


@data.command("validate")
@handled
def validate(path: Path, model: str | None = None, task: str = "understanding_sft"):
    """Check structure, IDs, split groups and optional model data requirements."""
    report = validate_dataset(path)
    result = report.model_dump(mode="json")
    diagnostics = []
    model_errors = 0
    if model is not None:
        processor = OmniDataProcessor()
        processor.registry.get(model)
        if report.invalid == 0:
            for sample in iter_samples(path):
                for issue in processor.validate(sample, model=model, task=task):
                    model_errors += issue.severity == "error"
                    if len(diagnostics) < 20:
                        diagnostics.append(
                            {"sample_id": sample.id, **issue.model_dump(mode="json")}
                        )
    result.update(
        model_diagnostics=diagnostics,
        model_errors=model_errors,
        scope="semantic_contracts_only; media bytes and model training are not checked",
    )
    emit(result)
    if report.invalid or report.records == 0 or model_errors:
        raise typer.Exit(2)


@data.command("inspect")
@handled
def inspect_sample(path: Path, sample_id: Annotated[str, typer.Option()]):
    emit(find_sample(path, sample_id))


@data.command("plan")
@handled
def plan_data(
    path: Path,
    sample_id: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    task: str = "understanding_sft",
):
    emit(OmniDataProcessor().plan(find_sample(path, sample_id), model=model, task=task))


@data.command("inference")
@handled
def inference(path: Path, output: Annotated[Path, typer.Option()], outputs: str = "text"):
    """Write target-free semantic views, not HTTP requests or generated results."""
    separate_output(output, path)
    requested = tuple(outputs.split(","))
    if (
        not requested
        or len(set(requested)) != len(requested)
        or not set(requested) <= {"text", "audio"}
    ):
        raise OmniFactoryError("E_OUTPUT_MODALITY", "use text, audio, or text,audio")
    count = 0
    with atomic_writer(output) as stream:
        for sample in iter_samples(path):
            for prediction in sample.predictions:
                view = OmniDataProcessor().to_inference(sample, prediction.id, requested)
                stream.write(view.model_dump_json() + "\n")
                count += 1
        if not count:
            raise OmniFactoryError("E_EMPTY_DATASET", "no predictions")
    emit({"output": str(output), "predictions": count})


@data.command("manifest")
@handled
def manifest(path: Path, output: Annotated[Path, typer.Option()], root: Path | None = None):
    """Hash local assets and validate aliases; does not decode/encode media."""
    separate_output(output, path)
    resolver = LocalMediaResolver(root or path.parent)
    count = 0
    with atomic_writer(output) as stream:
        for sample in iter_samples(path):
            row = resolver.manifest(sample)
            if any(Path(asset.path).resolve() == output.resolve() for asset in row.assets.values()):
                raise OmniFactoryError("E_OUTPUT", "output must not overwrite a media asset")
            stream.write(row.model_dump_json() + "\n")
            count += 1
        if not count:
            raise OmniFactoryError("E_EMPTY_DATASET", "no records")
    emit({"output": str(output), "samples": count, "verification": "file_bytes_only"})


class BatchConfig(Record):
    budget: ResourceBudget
    memory_profile: MemoryProfile | None = None
    window_size: Annotated[int, Field(gt=0)] = 128
    seed: int = 0


@batch.command("plan")
@handled
def plan_batches(
    costs: Annotated[Path, typer.Option()],
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
):
    """Group explicit model costs under resource budgets; no packing kernels are run."""
    separate_output(output, costs, config)
    settings = BatchConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    planner = ModalBatchPlanner(
        settings.budget,
        memory_profile=settings.memory_profile,
        window_size=settings.window_size,
        seed=settings.seed,
    )

    def rows():
        seen = set()
        with costs.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = SampleCost.model_validate_json(line)
                identity = (row.sample_id, row.prediction_id)
                if identity in seen:
                    raise OmniFactoryError("E_DUPLICATE_PREDICTION", str(identity))
                seen.add(identity)
                yield row

    count = examples = 0
    with atomic_writer(output) as stream:
        for planned in planner.plan(rows()):
            stream.write(planned.model_dump_json() + "\n")
            count += 1
            examples += len(planned.example_ids)
        if not count:
            raise OmniFactoryError("E_EMPTY_DATASET", "no costs")
    emit({"output": str(output), "batches": count, "examples": examples, "gpu_fit_verified": False})


@adaptation.command("plan")
@handled
def plan_adaptation(
    config: Annotated[Path, typer.Option()], inventory: Annotated[Path, typer.Option()]
):
    """Resolve full/LoRA/frozen policies against an explicit parameter inventory."""
    request = AdaptationRequest.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    raw = load_json(inventory)
    if not isinstance(raw, list):
        raise OmniFactoryError("E_INVENTORY", "inventory must be a JSON array")
    parameters = [ParameterSpec.model_validate(item) for item in raw]
    emit(compile_adaptation(builtin_registry().get(request.model), request, parameters))


# Optional execution commands import heavy dependencies only when invoked.
def require_training():
    if any(
        importlib.util.find_spec(name) is None
        for name in ("torch", "transformers", "peft", "PIL", "soundfile")
    ):
        raise OmniFactoryError(
            "E_TRAINING_INSTALL",
            'install the wheel with the train extra (e.g. uv pip install "./package.whl[train]") '
            "or run uv sync --extra train in a source checkout",
        )


datasets = typer.Typer(
    no_args_is_help=True, help="Provided datasets and reproducible local fixtures."
)
app.add_typer(datasets, name="datasets")


@datasets.command("list")
def list_datasets():
    from omnifactory.datasets.demo import CATALOG

    emit(CATALOG)


@datasets.command("demo")
@handled
def demo_dataset(output: Annotated[Path, typer.Option()]):
    require_training()
    from omnifactory.datasets.demo import create_demo

    emit(create_demo(output))


@models.command("init-tiny")
@handled
def init_tiny(
    output: Annotated[Path, typer.Option()],
    seed: int = 42,
    adapter: str = "qwen2_5_omni",
    speech: bool = False,
):
    """Create random tiny native weights for offline execution tests, not useful inference."""
    require_training()
    from omnifactory.adapters.runtime import adapter_class
    from omnifactory.training.tiny import create_tiny

    adapter_class(adapter)
    if adapter == "qwen3_omni":
        from omnifactory.training.tiny_qwen3 import create_tiny_qwen3

        path = create_tiny_qwen3(output, seed, speech=speech)
        emit(
            {
                "output": str(path),
                "random_weights": True,
                "architecture": "qwen3_omni_moe" if speech else "qwen3_omni_moe_thinker",
            }
        )
    elif adapter == "voxcpm2":
        from omnifactory.training.tiny_voxcpm2 import create_tiny_voxcpm2

        emit(create_tiny_voxcpm2(output, seed))
    elif adapter == "bagel_dit":
        from omnifactory.training.tiny_bagel import create_tiny_bagel

        emit(create_tiny_bagel(output, seed))
    elif adapter == "mimo_audio":
        from omnifactory.training.tiny_mimo import create_tiny_mimo

        emit(create_tiny_mimo(output, seed))
    elif adapter == "minicpm_o_4_5":
        from omnifactory.training.tiny_minicpm import create_tiny_minicpm

        emit(create_tiny_minicpm(output, seed, speech=speech))
    else:
        emit(create_tiny(output, seed, speech=speech))


@app.command("prepare")
@handled
def prepare_training(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    resume: bool = False,
    reuse_from: Path | None = None,
    split: str = "train",
):
    """Encode real text/image/audio inputs to disk-backed training tensors."""
    require_training()
    from omnifactory.training.config import read_config
    from omnifactory.training.prepare import prepare

    manifest = prepare(
        read_config(config), output, split=split, resume=resume, reuse_from=reuse_from
    )
    emit(
        {
            "output": str(output.resolve()),
            "predictions": len(manifest["entries"]),
            "fingerprint": manifest["fingerprint"],
        }
    )


def training_progress(metrics):
    typer.echo(
        f"step={metrics['step']} loss={metrics['loss']:.6f} measures={metrics['normalization_counts']}",
        err=True,
    )


@app.command("train")
@handled
def train_model(config: Annotated[Path, typer.Option()]):
    """Run single-device understanding or speech SFT with a registered family adapter."""
    require_training()
    from omnifactory.training.config import read_config
    from omnifactory.training.engine import train

    emit(train(read_config(config), progress=training_progress))


@app.command("evaluate")
@handled
def evaluate_model(
    checkpoint: Annotated[Path, typer.Option()],
    dataset: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    root: Path | None = None,
    generate: bool = False,
    max_new_tokens: int = 32,
):
    require_training()
    from omnifactory.training.evaluation import evaluate

    emit(
        evaluate(
            checkpoint,
            dataset,
            root=root,
            output=output,
            generate=generate,
            max_new_tokens=max_new_tokens,
        )
    )


@app.command("export")
@handled
def export_model(
    checkpoint: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    format: str = "merged",
):
    """Merge adapters into native weights for inference or a supported serving backend."""
    require_training()
    from omnifactory.training.export import export_checkpoint

    emit(export_checkpoint(checkpoint, output, format=format))


@app.command("infer")
@handled
def infer_model(
    model: Annotated[Path, typer.Option()],
    dataset: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    root: Path | None = None,
    max_new_tokens: int = 32,
    generate_audio: bool = False,
    generate_image: bool = False,
    image_steps: Annotated[int, typer.Option(min=2)] = 24,
    max_codec_tokens: Annotated[int, typer.Option(min=1)] = 128,
    max_audio_units: Annotated[int | None, typer.Option(min=1)] = None,
    speech_assets: Path | None = None,
    reference_audio: Path | None = None,
    speaker: str | None = None,
    device: str = "cpu",
    dtype: str = "float32",
):
    """Reload an exported model and generate text or native speech on CPU/CUDA."""
    require_training()
    from omnifactory.training.evaluation import infer_export

    emit(
        infer_export(
            model,
            dataset,
            output=output,
            root=root,
            max_new_tokens=max_new_tokens,
            generate_audio=generate_audio,
            generate_image=generate_image,
            image_steps=image_steps,
            max_codec_tokens=max_codec_tokens,
            max_audio_units=max_audio_units,
            speech_assets=speech_assets,
            reference_audio=reference_audio,
            speaker=speaker,
            device=device,
            dtype=dtype,
        )
    )


@app.command("smoke")
@handled
def smoke_workflow(
    output: Annotated[Path, typer.Option()],
    steps: Annotated[int, typer.Option(min=1)] = 12,
    adapter: str = "qwen2_5_omni",
    speech_assets: Path | None = None,
    speech: bool = False,
):
    """Offline data → real tiny native training → evaluation → export → inference."""
    require_training()
    from omnifactory.training.smoke import smoke

    emit(
        smoke(
            output,
            steps,
            progress=training_progress,
            adapter=adapter,
            speech_assets=speech_assets,
            speech=speech,
        )
    )


@datasets.command("import-alpaca")
@handled
def import_alpaca_dataset(
    source: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    name: Annotated[str, typer.Option()],
    split: str = "train",
):
    """Convert local text-only Alpaca JSONL to the canonical conversation schema."""
    from omnifactory.datasets.importers import import_alpaca

    emit(import_alpaca(source, output, name=name, split=split))


@datasets.command("import-tts")
@handled
def import_tts_dataset(
    source: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    name: Annotated[str, typer.Option()],
    split: str = "train",
):
    """Import text/waveform utterances with explicit speaker/group IDs and optional references."""
    from omnifactory.datasets.importers import import_tts

    emit(import_tts(source, output, name=name, split=split))


deploy = typer.Typer(no_args_is_help=True, help="Check an already running vLLM-Omni API.")
app.add_typer(deploy, name="deploy")


@deploy.command("check")
@handled
def check_deployment(
    model: Annotated[str, typer.Option()],
    base_url: str = "http://127.0.0.1:8091/v1",
    dataset: Path | None = None,
    root: Path | None = None,
    output: Path | None = None,
):
    """Send target-free prompts and validate text responses; does not start a server."""
    from omnifactory.deployment import check_server

    emit(check_server(base_url, model, dataset=dataset, root=root, output=output))


@models.command("fetch-speech-assets")
@handled
def fetch_speech_assets(output: Annotated[Path, typer.Option()]):
    """Download pinned native MiniCPM S3/Token2Wav assets (about 1.23 GB)."""
    require_training()
    from omnifactory.adapters.token2wav import fetch_assets

    emit(fetch_assets(output))


@datasets.command("omni-demo")
@handled
def omni_demo(output: Annotated[Path, typer.Option()]):
    """Create real MP4/WAV full-modality execution fixtures; requires FFmpeg."""
    require_training()
    from omnifactory.datasets.omni_demo import create_omni_demo

    emit(create_omni_demo(output))


@datasets.command("qwen-speech")
@handled
def qwen_speech_dataset(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    max_codec_tokens: Annotated[int, typer.Option(min=1)] = 750,
    allow_truncated: bool = False,
    validation_dataset: Path | None = None,
):
    """Synthesize native Qwen waveform/code pairs from supplied transcripts (not encode recordings)."""
    require_training()
    from omnifactory.datasets.qwen_speech import synthesize_dataset
    from omnifactory.training.config import read_config

    resolved = read_config(config)
    emit(
        synthesize_dataset(
            resolved,
            output,
            datasets=[resolved.dataset, validation_dataset] if validation_dataset else None,
            max_codec_tokens=max_codec_tokens,
            allow_truncated=allow_truncated,
        )
    )


@datasets.command("mimo-demo")
@handled
def mimo_demo(output: Annotated[Path, typer.Option()]):
    """Create audio-native input/output fixtures; synthetic tones, not speech quality."""
    require_training()
    from omnifactory.datasets.mimo_demo import create_mimo_demo

    emit(create_mimo_demo(output))


@models.command("fingerprint")
@handled
def fingerprint_model(
    path: Annotated[Path, typer.Option(exists=True, file_okay=False)], refresh: bool = False
):
    """Hash a local model/codec directory for reproducible training configuration."""
    require_training()
    from omnifactory.training.prepare import source_fingerprint

    emit(
        {
            "path": str(path.resolve()),
            "sha256": source_fingerprint(str(path), None, refresh=refresh),
        }
    )


@app.command("flow-decode")
@handled
def decode_flow(
    assets: Annotated[Path, typer.Option()],
    codes: Annotated[Path, typer.Option()],
    reference_audio: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    device: str = "cpu",
):
    """Decode explicit native S3 codes with original or fine-tuned Token2Wav assets."""
    require_training()
    import soundfile as sf

    from omnifactory.adapters.token2wav import NativeToken2Wav

    if output.exists():
        raise OmniFactoryError("E_OUTPUT_EXISTS", str(output))
    if device not in ("cpu", "cuda"):
        raise OmniFactoryError("E_DEVICE", "flow-decode supports cpu or cuda")
    waveform, rate = NativeToken2Wav(assets, device=device).decode(
        json.loads(codes.read_text()), reference_audio
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, waveform, rate)
    emit({"audio": str(output.resolve()), "sample_rate": rate, "frames": len(waveform)})


@datasets.command("image-demo")
@handled
def image_demo(output: Annotated[Path, typer.Option()], size: int = 256):
    """Generate local synthetic text/image pairs for execution checks."""
    from omnifactory.datasets.image_demo import create_image_demo

    emit(create_image_demo(output, size=size))


@datasets.command("image-edit-demo")
@handled
def image_edit_demo(output: Annotated[Path, typer.Option()], size: int = 256):
    """Generate synthetic source/instruction/target triples for editing execution checks."""
    from omnifactory.datasets.image_demo import create_image_demo

    emit(create_image_demo(output, size=size, editing=True))


@datasets.command("duplex-demo")
@handled
def duplex_demo(
    output: Annotated[Path, typer.Option()], tokenizer: Annotated[Path, typer.Option()]
):
    """Create synthetic two-track sessions and explicit token/frame annotations."""
    from omnifactory.datasets.duplex_demo import create_duplex_demo

    emit(create_duplex_demo(output, tokenizer))


@datasets.command("import-duplex")
@handled
def import_duplex_dataset(
    source: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    tokenizer: Annotated[Path, typer.Option()],
    word_slack_us: int = 0,
    max_seconds: int = 300,
):
    """Import synchronized mono/stereo recordings and explicitly timed transcripts."""
    from omnifactory.datasets.duplex_import import import_duplex

    emit(
        import_duplex(
            source, output, tokenizer, word_slack_us=word_slack_us, max_seconds=max_seconds
        )
    )


@datasets.command("flow-demo")
@handled
def flow_demo(output: Annotated[Path, typer.Option()]):
    """Create separate reference/target tone fixtures for native flow training."""
    require_training()
    from omnifactory.datasets.flow_demo import create_flow_demo

    emit(create_flow_demo(output))


@app.command("benchmark")
@handled
def benchmark_training(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    batching: str = "modal",
    warmup_epochs: Annotated[int, typer.Option(min=1)] = 1,
    repeats: Annotated[int, typer.Option(min=1)] = 3,
):
    """Measure full-dataset training passes: throughput, padding and CUDA memory."""
    require_training()
    from omnifactory.training.benchmark import benchmark
    from omnifactory.training.config import read_config

    emit(
        benchmark(
            read_config(config),
            output,
            batching=batching,
            warmup_epochs=warmup_epochs,
            repeats=repeats,
        )
    )


@app.command("calibrate-memory")
@handled
def calibrate_memory(
    benchmark_dir: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    safety_fraction: Annotated[float, typer.Option(min=0, max=1)] = 0.15,
):
    """Fit bounded GPU memory costs and validate on held-out shape cases."""
    require_training()
    from omnifactory.training.calibration import calibrate

    emit(calibrate(benchmark_dir, output, safety_fraction=safety_fraction))


@deploy.command("verify-export")
@handled
def verify_export_command(model: Annotated[Path, typer.Option()]):
    """Verify artifact hashes, native assets and complete shard indexes before serving."""
    from omnifactory.artifacts import verify_export

    emit(verify_export(model))


acceptance = typer.Typer(no_args_is_help=True)
app.add_typer(acceptance, name="acceptance")


@acceptance.command("run")
@handled
def acceptance_run(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    generate: bool = True,
):
    from omnifactory.acceptance import run_acceptance
    from omnifactory.training.config import read_config

    emit(run_acceptance(read_config(config), output, generate=generate))


@acceptance.command("attach")
@handled
def acceptance_attach(
    report: Annotated[Path, typer.Option()],
    gate: Annotated[Path, typer.Option()],
    name: Annotated[str, typer.Option()],
):
    from omnifactory.acceptance import attach_gate

    emit(attach_gate(report, gate, name=name))


@deploy.command("validate")
@handled
def deployment_validate(
    model: Annotated[Path, typer.Option()],
    dataset: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    executable: Annotated[Path, typer.Option()],
    deploy_config: Annotated[Path, typer.Option()],
    backend: str = "vllm-omni",
    startup_timeout: Annotated[int, typer.Option(min=1)] = 300,
    max_audio_units: Annotated[int, typer.Option(min=2)] = 128,
):
    from omnifactory.serving import validate_deployment

    emit(
        validate_deployment(
            model,
            dataset,
            output,
            executable,
            deploy_config,
            backend=backend,
            startup_timeout=startup_timeout,
            max_audio_units=max_audio_units,
        )
    )


@app.command("capture-environment")
@handled
def capture_environment_command(output: Annotated[Path, typer.Option()]):
    from omnifactory.environment import capture_environment

    emit(capture_environment(output))


@models.command("render-docs")
@handled
def render_model_docs(root: Path = Path("."), check: bool = False):
    from omnifactory.capabilities import update_readme

    update_readme(root / "README.md", "en", check=check)
    update_readme(root / "README_zh.md", "zh", check=check)
    emit({"checked": check, "source": "omnifactory/evidence/catalog.json"})


@app.command("compare-speech")
@handled
def compare_speech(
    baseline: Annotated[Path, typer.Option()],
    tuned: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    model: Annotated[Path, typer.Option()],
    scorer: Annotated[str, typer.Option()],
):
    from omnifactory.artifacts import verify_export
    from omnifactory.quality import compare_transcripts

    artifact = verify_export(model)
    emit(
        compare_transcripts(
            baseline, tuned, output, artifact_id=artifact["artifact_id"], scorer=scorer
        )
    )
