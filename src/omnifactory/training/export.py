"""Merge training weights and optionally assemble a native multi-stage checkpoint."""

import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import new_directory, write_json
from omnifactory.training.engine import load_checkpoint
from omnifactory.training.policy import merge_lora


def assemble_native(
    base: Path, model, output: Path, *, shard_bytes=1024**3, adapter="qwen2_5_omni"
):
    """Copy native weights, replacing the selected family's understanding components.

    Shards are streamed; extra peak host memory is one shard plus its largest tensor.
    This proves structural preservation, not successful vLLM-Omni execution.
    """
    config = json.loads((base / "config.json").read_text())
    mini = adapter == "minicpm_o_4_5"
    qwen3 = adapter == "qwen3_omni"
    expected = "minicpmo" if mini else "qwen3_omni_moe" if qwen3 else "qwen2_5_omni"
    if config.get("model_type") != expected:
        raise OmniFactoryError(
            "E_NATIVE_BASE", f"native assembly requires original full {expected} checkpoint"
        )
    if not mini and not qwen3 and not (base / "spk_dict.pt").is_file():
        raise OmniFactoryError("E_NATIVE_ASSETS", "native Qwen export requires spk_dict.pt")
    if mini:
        from omnifactory.adapters.minicpm import NATIVE_PREFIXES

        prefixes, namespace = NATIVE_PREFIXES + (("tts.",) if hasattr(model, "tts") else ()), ""
    else:
        prefixes, namespace = (
            (("thinker.", "talker."), "")
            if hasattr(model, "thinker")
            else (("thinker.",), "thinker.")
        )
    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        files = sorted(set(mapping.values()))
    elif (base / "model.safetensors").exists():
        files = ["model.safetensors"]
    else:
        raise OmniFactoryError("E_NATIVE_WEIGHTS", "native assembly requires safetensors weights")
    for file in files:
        if Path(file).name != file:
            raise OmniFactoryError("E_NATIVE_WEIGHTS", "shard paths must be local filenames")
    replacement = {
        namespace + k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
    }
    seen, weight_map, pending, size, total, number = set(), {}, {}, 0, 0, 0

    def flush():
        nonlocal pending, size, number
        if not pending:
            return
        name = f"model-{number:05d}.safetensors"
        save_file(pending, output / name, metadata={"format": "pt"})
        weight_map.update({key: name for key in pending})
        number += 1
        pending, size = {}, 0

    for file in files:
        with safe_open(base / file, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in seen:
                    raise OmniFactoryError("E_NATIVE_WEIGHTS", f"duplicate tensor: {key}")
                seen.add(key)
                if key.startswith(prefixes):
                    if key not in replacement or tuple(reader.get_slice(key).get_shape()) != tuple(
                        replacement[key].shape
                    ):
                        raise OmniFactoryError("E_NATIVE_SHAPE", key)
                    tensor = replacement[key]
                    # Preserve original checkpoint dtype for native serving.
                    dtype = reader.get_slice(key).get_dtype()
                    import torch

                    types = {
                        "F32": torch.float32,
                        "BF16": torch.bfloat16,
                        "F16": torch.float16,
                        "I64": torch.int64,
                        "I32": torch.int32,
                        "BOOL": torch.bool,
                    }
                    if dtype not in types:
                        raise OmniFactoryError("E_NATIVE_DTYPE", dtype)
                    tensor = tensor.to(types[dtype])
                else:
                    tensor = reader.get_tensor(key)
                amount = tensor.numel() * tensor.element_size()
                if pending and size + amount > shard_bytes:
                    flush()
                pending[key], size, total = tensor, size + amount, total + amount
    if set(replacement) - seen:
        raise OmniFactoryError(
            "E_NATIVE_KEYS", "trained understanding model has parameters absent from native base"
        )
    required = (
        ("tts.",) if mini else ("talker.", "code2wav.") if qwen3 else ("talker.", "token2wav.")
    )
    if any(not any(k.startswith(prefix) for k in seen) for prefix in required):
        raise OmniFactoryError("E_NATIVE_STAGES", f"native base must contain {required}")
    flush()
    for path in base.iterdir():
        if path.is_file() and path.suffix in (
            ".json",
            ".jinja",
            ".txt",
            ".model",
            ".tiktoken",
            ".pt",
            ".py",
        ):
            if path.name != "model.safetensors.index.json":
                shutil.copy2(path, output / path.name)
    if mini:
        # Native speech decoders may be external assets; preserve original subdirectories.
        for path in base.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                shutil.copytree(path, output / path.name)
        for name in ("LICENSE", "NOTICE"):
            if (base / name).is_file():
                shutil.copy2(base / name, output / name)
    write_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total}, "weight_map": weight_map},
    )
    return {"tensors": len(seen), "shards": number, "weight_bytes": total}


def export_builtin(checkpoint: Path, output: Path, *, format=None):
    if format not in (None, "merged", "understanding", "thinker", "vllm-omni"):
        raise OmniFactoryError("E_EXPORT_FORMAT", format)
    model, adapter, config, metadata = load_checkpoint(checkpoint, for_export=True)
    if config.adapter == "bagel_dit":
        if format not in (None, "merged", "vllm-omni"):
            raise OmniFactoryError("E_EXPORT_FORMAT", "BAGEL requires native merged export")
        return export_bagel(checkpoint, output, model, config, metadata, format or "merged")
    if config.adapter == "voxcpm2":
        if format not in (None, "merged", "vllm-omni"):
            raise OmniFactoryError("E_EXPORT_FORMAT", "VoxCPM2 requires a native merged export")
        return export_voxcpm2(checkpoint, output, model, config, metadata, format or "merged")
    if config.task == "flow_sft":
        if format not in (None, "merged"):
            raise OmniFactoryError(
                "E_EXPORT_FORMAT",
                "flow exports native stage assets; use merged, not a full Omni export",
            )
        return export_flow(checkpoint, output, model, config, metadata)
    if format is None:
        format = (
            "merged"
            if config.task == "speech_sft"
            else (
                "thinker" if config.adapter in ("qwen2_5_omni", "qwen3_omni") else "understanding"
            )
        )
    if config.task == "speech_sft" and format in ("thinker", "understanding"):
        raise OmniFactoryError(
            "E_EXPORT_FORMAT", "speech exports require merged or vllm-omni format"
        )
    if format == "thinker" and config.adapter not in ("qwen2_5_omni", "qwen3_omni"):
        raise OmniFactoryError("E_EXPORT_FORMAT", "use understanding for MiniCPM exports")
    model = merge_lora(model)
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    with new_directory(output) as root:
        details = {}
        qwen_speech = (
            config.adapter in ("qwen2_5_omni", "qwen3_omni") and config.task == "speech_sft"
        )
        if config.adapter == "mimo_audio":
            from omnifactory.adapters.mimo import MiMoAdapter
            from omnifactory.training.prepare import source_fingerprint

            # Native key layout is shared by global model, patch encoder and decoder.
            model.save_pretrained(root, safe_serialization=True)
            adapter.processor.save_pretrained(root)
            native_config = json.loads((root / "config.json").read_text())
            native_config["architectures"] = ["MiMoAudioModel"]
            write_json(root / "config.json", native_config)
            if config.speech_tokenizer is not None:
                codec_path = MiMoAdapter.from_config(config).codec_path
                if source_fingerprint(str(codec_path), None) != config.speech_tokenizer_sha256:
                    raise OmniFactoryError("E_CODEC_HASH", "codec changed before export")
                shutil.copytree(codec_path, root / "assets/mimo_codec")
        elif format in ("merged", "thinker", "understanding") and not qwen_speech:
            model.save_pretrained(root, safe_serialization=True)
            adapter.save_processor(root)
        else:
            base = Path(config.model)
            if not base.is_dir():
                from huggingface_hub import snapshot_download

                base = Path(snapshot_download(config.model, revision=config.revision))
            details = assemble_native(base, model, root, adapter=config.adapter)
        if config.task == "speech_sft" and config.speech_assets is not None:
            from omnifactory.adapters.token2wav import verify_assets

            verify_assets(config.speech_assets)
            shutil.copytree(config.speech_assets, root / "assets/token2wav", dirs_exist_ok=True)
        if config.duplex_schedule is not None:
            shutil.copy2(config.duplex_schedule, root / "duplex_schedule.json")
        result = {
            "timed_attention_required": config.duplex_schedule is not None,
            "format": format,
            "adapter": config.adapter,
            "inference_config": config.model_dump(mode="json"),
            "source_checkpoint": str(checkpoint.resolve()),
            "source_fingerprint": metadata["source_fingerprint"],
            "output": str(output.resolve()),
            "lora_merged": True,
            "vllm_deployment_verified": False,
            **details,
        }
        write_json(root / "omnifactory_export.json", result)
    return result


def export_voxcpm2(checkpoint, output, model, config, metadata, format):
    """Merged native model plus unchanged separate VAE, readable by upstream/vLLM."""
    from omnifactory.adapters.voxcpm2_model import model_root

    base = model_root(config)
    model = merge_lora(model)
    with new_directory(output) as root:
        state = {
            k: v.detach().cpu().contiguous()
            for k, v in model.state_dict().items()
            if not k.startswith("audio_vae.")
        }
        save_file(state, root / "model.safetensors")
        for path in base.iterdir():
            if (
                path.is_file()
                and (
                    path.suffix in (".json", ".model", ".txt", ".jinja", ".py")
                    or path.name in ("LICENSE", "NOTICE", "audiovae.safetensors", "audiovae.pth")
                )
                and path.name != "omnifactory_export.json"
            ):
                shutil.copy2(path, root / path.name)
        result = {
            "format": format,
            "adapter": config.adapter,
            "inference_config": config.model_dump(mode="json"),
            "source_checkpoint": str(checkpoint.resolve()),
            "source_fingerprint": metadata["source_fingerprint"],
            "output": str(output.resolve()),
            "lora_merged": True,
            "vllm_deployment_verified": False,
            "native_architecture": "voxcpm2",
            "audio_vae_unchanged": True,
        }
        write_json(root / "omnifactory_export.json", result)
    return result


def export_flow(checkpoint, output, model, config, metadata):
    """Native stage assets; drop-in Token2Wav replacement, not a complete Omni model."""
    import torch

    from omnifactory.adapters.speech import file_sha256
    from omnifactory.adapters.token2wav import ASSET_FILES, verify_assets

    original = verify_assets(config.model)
    model = merge_lora(model)
    with new_directory(output) as root:
        for name in ASSET_FILES:
            if name != "flow.pt":
                shutil.copy2(Path(config.model) / name, root / name)
        torch.save(
            {k: v.detach().cpu() for k, v in model.flow.state_dict().items()}, root / "flow.pt"
        )
        write_json(
            root / "omnifactory_assets.json",
            {
                **original,
                "base_sha256": original["sha256"],
                "sha256": {name: file_sha256(root / name) for name in ASSET_FILES},
                "fine_tuned": True,
            },
        )
        result = {
            "format": "token2wav-assets",
            "adapter": config.adapter,
            "source_checkpoint": str(checkpoint.resolve()),
            "source_fingerprint": metadata["source_fingerprint"],
            "output": str(output.resolve()),
            "vllm_deployment_verified": False,
        }
        write_json(root / "omnifactory_export.json", result)
    return result


def export_bagel(checkpoint, output, model, config, metadata, format):
    """Overlay the merged native generation model; preserve every other tensor."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    from omnifactory.adapters.bagel_model import model_root

    base = model_root(config)
    changed = {
        name for name, p in model.named_parameters() if p.requires_grad and ".lora_" not in name
    }
    changed.update(
        name + ".weight"
        for name, module in model.named_modules()
        if isinstance(module, BaseTunerLayer)
    )
    model = merge_lora(model)
    tuned = {
        k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in changed
    }
    with new_directory(output) as root:
        with safe_open(base / "ema.safetensors", framework="pt", device="cpu") as source:
            if set(tuned) - set(source.keys()):
                raise OmniFactoryError(
                    "E_NATIVE_WEIGHTS", "BAGEL export introduced non-native keys"
                )
            state = {
                key: tuned.get(key) if key in tuned else source.get_tensor(key)
                for key in source.keys()
            }
            save_file(state, root / "ema.safetensors")
            total_size = sum(t.numel() * t.element_size() for t in state.values())
        for path in base.iterdir():
            if path.is_file() and not path.name.startswith(".") and path.name != "ema.safetensors":
                shutil.copy2(path, root / path.name)
        index = root / "model.safetensors.index.json"
        weight_map = {key: "ema.safetensors" for key in state}
        # vLLM's indexed loader only visits shards named here. The frozen VAE is
        # a separate native shard, even though training only loads the MoT model.
        with safe_open(root / "ae.safetensors", framework="pt", device="cpu") as vae:
            for key in vae.keys():
                if key in weight_map:
                    raise OmniFactoryError("E_NATIVE_WEIGHTS", "BAGEL VAE/core key collision")
                weight_map[key] = "ae.safetensors"
                tensor = vae.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()
        write_json(index, {"metadata": {"total_size": total_size}, "weight_map": weight_map})
        result = dict(
            format=format,
            adapter=config.adapter,
            inference_config=config.model_dump(mode="json"),
            source_checkpoint=str(checkpoint.resolve()),
            source_fingerprint=metadata["source_fingerprint"],
            output=str(output.resolve()),
            lora_merged=True,
            vllm_deployment_verified=False,
            native_architecture="bagel",
            generation_only=True,
            updated_parameters=sorted(changed),
        )
        write_json(root / "omnifactory_export.json", result)
    return result


def export_checkpoint(checkpoint: Path, output: Path, *, format=None):
    from omnifactory.adapters.runtime import adapter_spec, resolve
    from omnifactory.artifacts import seal_export

    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    spec = adapter_spec(metadata["config"]["adapter"])
    result = resolve(spec.exporter)(checkpoint, output, format=format)
    try:
        seal_export(output)
    except Exception:
        # Failed validation must never leave a directory that appears deployable.
        shutil.rmtree(output)
        raise
    return result
