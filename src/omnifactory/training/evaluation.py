"""Held-out supervision objectives and target-free generation for family adapters."""

import json
import math
from pathlib import Path

import torch

from omnifactory.adapters.runtime import adapter_class, adapter_spec, load_model
from omnifactory.adapters.supervision import expected_losses, target_measure
from omnifactory.data.io import atomic_writer, iter_samples, validate_dataset
from omnifactory.data.processor import OmniDataProcessor
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, to_device
from omnifactory.training.losses import LossBundle


def evaluate(
    checkpoint: Path,
    dataset: Path,
    *,
    root: Path | None = None,
    output: Path | None = None,
    device: str | None = None,
    generate: bool = False,
    max_new_tokens: int = 32,
):
    if output and output.resolve() == dataset.resolve():
        raise OmniFactoryError("E_OUTPUT", "evaluation output must not overwrite the dataset")
    if max_new_tokens < 1:
        raise OmniFactoryError("E_GENERATION", "max_new_tokens must be positive")
    report = validate_dataset(dataset)
    if report.invalid or not report.records:
        raise OmniFactoryError("E_DATASET", "; ".join(report.errors))
    model, adapter, config, metadata = load_checkpoint(checkpoint, device=device)
    adapter.resolver.root = (root or dataset.parent).resolve()
    from omnifactory.training.groups import checkpoint_groups

    training_groups = checkpoint_groups(checkpoint, metadata)
    predictions = matches = 0
    aggregate = None
    objective_measures = {}
    rows = []
    with torch.no_grad():
        for sample in iter_samples(dataset):
            if sample.provenance.split not in ("validation", "test"):
                raise OmniFactoryError(
                    "E_EVAL_SPLIT", "evaluation requires validation or test rows"
                )
            if (sample.provenance.dataset, sample.provenance.group_id) in training_groups:
                raise OmniFactoryError("E_SPLIT_LEAK", "evaluation group occurred in training")
            for prediction in sample.predictions:
                encoded = OmniDataProcessor().encode(
                    sample, prediction.id, adapter=adapter, contract_id="evaluation"
                )
                forward = adapter.forward(model, to_device(encoded.payload, model, config))
                bundle = forward.losses
                bundle.validate_expected(
                    expected_losses([(encoded.supervision, encoded.normalization_counts)])
                )
                detached = bundle.detached()
                aggregate = (
                    detached if aggregate is None else LossBundle.combine([aggregate, detached])
                )
                targets = {t.id: t for t in encoded.supervision.targets}
                for objective in encoded.supervision.objectives:
                    objective_measures[objective.id] = objective_measures.get(
                        objective.id, 0
                    ) + target_measure(targets[objective.target_id], encoded.tensors)
                predictions += 1
                if generate:
                    prompt = OmniDataProcessor().encode(
                        sample,
                        prediction.id,
                        adapter=adapter,
                        contract_id="inference",
                        inference=True,
                    )
                    text = adapter.generate(
                        model,
                        to_device(prompt.payload, model, config),
                        max_new_tokens=max_new_tokens,
                    )[0]
                    matched = encoded.answer is not None and text.strip() == encoded.answer.strip()
                    matches += matched
                    rows.append(
                        {
                            "sample_id": sample.id,
                            "prediction_id": prediction.id,
                            "prediction": text,
                            "target": encoded.answer,
                            "exact_match": matched,
                        }
                    )
    if aggregate is None:
        raise OmniFactoryError("E_EMPTY_SUPERVISION", "no evaluation predictions")
    average = float(aggregate.loss)
    terms = {t.id: t for t in aggregate.terms}
    text_only_ce = (
        len(terms) == 1 and "text_ce" in terms and terms["text_ce"].kind == "cross_entropy"
    )
    text_mean = (
        float(terms["text_ce"].numerator) / objective_measures["text_ce"] if text_only_ce else None
    )
    codec_tokens = objective_measures.get("codec_ce", 0)
    result = {
        "predictions": predictions,
        **aggregate.report(),
        "objective_measures": objective_measures,
        "target_tokens": sum(aggregate.counts.values())
        if all(u == "symbols" for u in aggregate.units.values())
        else None,
        "loss": average,
        "perplexity": math.exp(text_mean) if text_mean is not None and text_mean < 700 else None,
        "codec_loss": float(terms["codec_ce"].numerator) / codec_tokens if codec_tokens else None,
        "codec_tokens": codec_tokens,
        "exact_match": matches / predictions if generate else None,
        "generation": rows if generate else None,
        "scope": "held-out declared supervision objectives; not perceptual quality",
    }
    if output:
        with atomic_writer(output) as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def infer_export(
    model_path: Path,
    dataset: Path,
    *,
    output: Path,
    root: Path | None = None,
    max_new_tokens: int = 32,
    generate_audio: bool = False,
    generate_image: bool = False,
    image_steps: int = 24,
    max_codec_tokens: int = 128,
    max_audio_units: int | None = None,
    speech_assets: Path | None = None,
    reference_audio: Path | None = None,
    speaker: str | None = None,
    device: str = "cpu",
    dtype: str = "float32",
):
    """Run a standalone exported understanding model in a fresh load, without its training adapter."""

    if output.resolve() == dataset.resolve():
        raise OmniFactoryError("E_OUTPUT", "inference output must not overwrite the dataset")
    if max_audio_units is not None:
        max_codec_tokens = max_audio_units
    if max_new_tokens < 1 or max_codec_tokens < 1:
        raise OmniFactoryError("E_GENERATION", "generation limits must be positive")
    if generate_image and (generate_audio or image_steps < 2):
        raise OmniFactoryError(
            "E_GENERATION", "image generation requires >=2 steps and no audio output"
        )
    metadata = json.loads((model_path / "omnifactory_export.json").read_text())
    if "adapter" not in metadata:
        raise OmniFactoryError("E_EXPORT_FORMAT", "export lacks adapter identity")
    spec = adapter_spec(metadata["adapter"])
    if spec.stage_only:
        raise OmniFactoryError(
            "E_FLOW_STAGE", "use flow-decode with codec tokens and reference audio for stage assets"
        )
    saved_config = metadata.get("inference_config")
    if saved_config is None:
        raise OmniFactoryError(
            "E_EXPORT_FORMAT",
            "export requires inference_config; export again with the current version",
        )
    config = TrainConfig.model_validate(
        {
            **saved_config,
            "model": str(model_path),
            "revision": None,
            "dataset": str(dataset),
            "dataset_root": str(root or dataset.parent),
            "device": device,
            "dtype": dtype,
            "quantization": "none",
            "gradient_checkpointing": False,
            "packing": "none",
            "feature_cache": None,
            "memory_profile": None,
            "budget": {**saved_config["budget"], "max_gpu_bytes": None},
        }
    )
    if speaker is not None and not spec.named_speakers:
        raise OmniFactoryError(
            "E_SPEAKER", "named speakers apply to Qwen; MiniCPM uses reference_audio"
        )
    if speaker is not None:
        config = config.model_copy(update={"speech_speaker": speaker})
    requested = "image" if generate_image else "audio" if generate_audio else "text"
    if requested not in spec.outputs:
        raise OmniFactoryError("E_OUTPUT_MODALITY", f"{config.adapter} cannot generate {requested}")
    model = load_model(config)
    model.eval()
    adapter = adapter_class(config.adapter).from_config(config)
    decoder = None
    if generate_audio:
        if config.task != "speech_sft":
            raise OmniFactoryError("E_SPEECH_INFERENCE", "audio output requires a speech export")
        decoder = adapter.speech_decoder(
            model, speech_assets=speech_assets, reference_audio=reference_audio
        )
    count = audio_count = empty_audio_count = image_count = 0
    torch.manual_seed(config.seed)
    with atomic_writer(output) as stream, torch.no_grad():
        for sample in iter_samples(dataset):
            for prediction in sample.predictions:
                encoded = OmniDataProcessor().encode(
                    sample,
                    prediction.id,
                    adapter=adapter,
                    contract_id="export-inference",
                    inference=True,
                )
                batch = adapter.to_device(encoded.payload, model, config)
                if generate_image:
                    image = adapter.generate_image(model, batch, num_steps=image_steps)
                    image_path = output.parent / (output.stem + f"-{count:06d}.png")
                    if image_path.exists():
                        raise OmniFactoryError("E_OUTPUT_EXISTS", str(image_path))
                    image.save(image_path)
                    image_count += 1
                    row = {
                        "sample_id": sample.id,
                        "prediction_id": prediction.id,
                        "image": str(image_path.resolve()),
                        "width": image.width,
                        "height": image.height,
                        "image_steps": image_steps,
                    }
                elif generate_audio:
                    import soundfile as sf

                    generated = adapter.generate_speech(
                        model,
                        batch,
                        max_new_tokens=max_new_tokens,
                        max_codec_tokens=max_codec_tokens,
                    )
                    representations = [
                        key for key in ("codec_tokens", "audio_latents") if key in generated
                    ]
                    if len(representations) != 1:
                        raise OmniFactoryError(
                            "E_AUDIO_REPRESENTATION", "exactly one audio representation required"
                        )
                    audio_values = generated[representations[0]]
                    if not audio_values:
                        empty_audio_count += 1
                        row = {
                            "sample_id": sample.id,
                            "prediction_id": prediction.id,
                            **generated,
                            "audio": None,
                            "sample_rate": None,
                            "audio_status": "no_audio_generated",
                        }
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
                        continue
                    waveform, rate = decoder(audio_values)
                    wav_path = output.parent / (output.stem + f"-{count:06d}.wav")
                    if wav_path.exists():
                        raise OmniFactoryError("E_OUTPUT_EXISTS", str(wav_path))
                    sf.write(wav_path, waveform, rate)
                    audio_count += 1
                    row = {
                        "sample_id": sample.id,
                        "prediction_id": prediction.id,
                        **generated,
                        "audio": str(wav_path.resolve()),
                        "sample_rate": rate,
                    }
                else:
                    text = adapter.generate(model, batch, max_new_tokens=max_new_tokens)[0]
                    row = {"sample_id": sample.id, "prediction_id": prediction.id, "text": text}
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
    return {
        "predictions": count,
        "audio_predictions": audio_count,
        "image_predictions": image_count,
        "empty_audio_predictions": empty_audio_count,
        "output": str(output.resolve()),
        "backend": f"native-{config.device}",
        "audio_limit_unit": spec.audio_unit if generate_audio else None,
    }
