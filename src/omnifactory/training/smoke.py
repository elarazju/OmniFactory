"""Offline reproducible smoke workflow through actual native model and PEFT implementations."""

from pathlib import Path

from omnifactory.datasets.demo import create_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import write_json
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train
from omnifactory.training.evaluation import evaluate, infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.tiny import create_tiny


def smoke(
    output: Path,
    steps: int = 12,
    progress=None,
    adapter="qwen2_5_omni",
    speech_assets: Path | None = None,
    speech: bool = False,
):
    from omnifactory.adapters.runtime import adapter_class

    adapter_class(adapter)  # Validate before creating outputs.
    if adapter == "mimo_audio" and speech_assets is not None:
        raise OmniFactoryError("E_SPEECH_ASSETS", "MiMo smoke uses its own native random codec")
    speech = speech or speech_assets is not None
    if adapter == "qwen3_omni" and speech:
        raise OmniFactoryError("E_SPEECH_MODEL", "Qwen3 smoke currently covers understanding only")
    extra = {}
    if speech and adapter == "minicpm_o_4_5":
        from omnifactory.adapters.token2wav import verify_assets

        if adapter != "minicpm_o_4_5":
            raise OmniFactoryError("E_SPEECH_MODEL", "speech smoke requires MiniCPM-o 4.5")
        if speech_assets is None:
            raise OmniFactoryError("E_SPEECH_ASSETS", "MiniCPM speech smoke requires speech_assets")
        speech_assets = speech_assets.resolve()
        metadata = verify_assets(speech_assets)
        extra = {
            "task": "speech_sft",
            "speech_assets": speech_assets,
            "speech_tokenizer": speech_assets / "speech_tokenizer_v2_25hz.onnx",
            "speech_tokenizer_sha256": metadata["sha256"]["speech_tokenizer_v2_25hz.onnx"],
        }
    if speech and adapter == "qwen2_5_omni":
        if speech_assets is not None:
            raise OmniFactoryError(
                "E_SPEECH_ASSETS", "Qwen uses checkpoint Token2Wav, not MiniCPM assets"
            )
        extra = {
            "task": "speech_sft",
            "speech_codes": output.resolve() / "data/speech_codes.jsonl",
            "video_fps": 2,
        }
    output = output.resolve()
    if output.exists():
        raise OmniFactoryError("E_OUTPUT_EXISTS", "smoke requires a new directory")
    output.mkdir(parents=True)
    if adapter == "mimo_audio":
        from omnifactory.training.tiny_mimo import create_tiny_mimo

        fixture = create_tiny_mimo(output / "base-model")
        extra = {
            "task": "speech_sft" if speech else "understanding_sft",
            "speech_tokenizer": output / "base-model/assets/mimo_codec",
            "speech_tokenizer_sha256": fixture["codec_fingerprint"],
        }
    elif adapter == "qwen3_omni":
        from omnifactory.training.tiny_qwen3 import create_tiny_qwen3

        create_tiny_qwen3(output / "base-model")
        fixture = {"random_weights": True, "architecture": "qwen3_omni_moe_thinker"}
    elif adapter == "minicpm_o_4_5":
        from omnifactory.training.tiny_minicpm import create_tiny_minicpm

        fixture = create_tiny_minicpm(output / "base-model", speech=speech)
    else:
        fixture = create_tiny(output / "base-model", speech=speech)
    if adapter == "mimo_audio":
        from omnifactory.datasets.mimo_demo import create_mimo_demo

        create_mimo_demo(output / "data")
    elif speech:
        from omnifactory.datasets.omni_demo import create_omni_demo

        create_omni_demo(output / ("seed-data" if adapter == "qwen2_5_omni" else "data"))
    else:
        create_demo(output / "data")
    components = {"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.01}}
    if speech:
        components["speech_generator"] = {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.01}
    if adapter == "mimo_audio":
        # A 40k-parameter random architecture needs its embeddings/heads learned too.
        # Component LoRA is verified separately in test_mimo_training.py.
        components = {
            name: {"mode": "full", "lr": 0.003}
            for name in (
                "language_model",
                "audio_encoder",
                "audio_projector",
                *(("speech_generator",) if speech else ()),
            )
        }
    config = TrainConfig(
        adapter=adapter,
        model=str(output / "base-model"),
        dataset=output / "data/train.jsonl",
        output_dir=output / "run",
        max_steps=steps,
        save_steps=steps,
        gradient_accumulation_steps=2,
        max_length=512,
        components=components,
        **extra,
        budget={
            "max_examples": 2,
            "max_llm_tokens": 1024,
            "max_vision_patches": 128,
            "max_audio_frames": 400,
            "max_codec_frames": 128 if speech else 1,
            "max_decoded_media_bytes": 1000000,
        },
    )
    if speech and adapter == "qwen2_5_omni":
        from omnifactory.datasets.qwen_speech import synthesize_dataset

        synthesize_dataset(
            config,
            output / "data",
            datasets=[output / "seed-data/train.jsonl", output / "seed-data/validation.jsonl"],
            max_codec_tokens=8,
            allow_truncated=True,
        )
    training = train(config, progress=progress)
    checkpoint = Path(training["checkpoint"])
    metrics = evaluate(
        checkpoint,
        output / "data/validation.jsonl",
        output=output / "evaluation.json",
        generate=True,
        max_new_tokens=8,
    )
    exported = export_checkpoint(checkpoint, output / "export")
    inference = infer_export(
        output / "export",
        output / "data/validation.jsonl",
        output=output / "predictions.jsonl",
        max_new_tokens=64 if adapter == "mimo_audio" else 8,
        generate_audio=speech,
        max_codec_tokens=16,
        reference_audio=output / "data/seed/media/train-002.wav"
        if speech and adapter == "minicpm_o_4_5"
        else None,
    )
    result = {
        "adapter": adapter,
        "task": config.task,
        "native_speech_assets": speech,
        "pretrained_speech_assets": speech and adapter == "minicpm_o_4_5",
        "speech_label_source": (
            "native_teacher"
            if adapter == "qwen2_5_omni"
            else "mimo_rvq_waveform_encoder"
            if adapter == "mimo_audio"
            else "s3_waveform_encoder"
        )
        if speech
        else None,
        "fixture": fixture,
        "training": training,
        "evaluation": metrics,
        "export": exported,
        "inference": inference,
        "random_weights": True,
        "production_model_verified": False,
        "vllm_deployment_verified": False,
    }
    write_json(output / "smoke-report.json", result)
    return result
