"""Native runtime constraints for built-in adapters, independent of torch imports."""

from omnifactory.planning.adaptation import ComponentPolicy


def validate_builtin(c):
    if c.personaplex_agent_only_depth:
        if c.adapter != "personaplex" or c.personaplex_user_loss_weight:
            raise ValueError("agent-only depth requires PersonaPlex with zero user loss weight")
        for name, policy in c.components.items():
            if policy.mode == "frozen" or name in ("temporal_model", "text_head"):
                continue
            if not (
                name in ("audio_projector", "speech_generator")
                and policy.mode == "lora"
                and set(policy.target_modules) <= {str(i) for i in range(8)}
            ):
                raise ValueError(
                    "agent-only depth requires frozen user depth parameters; "
                    "use agent projection/head LoRA targets 0..7"
                )
    if c.duplex_text_alignment is not None and c.adapter != "personaplex":
        raise ValueError("duplex_text_alignment currently belongs to PersonaPlex speech SFT")
    if c.adapter == "personaplex":
        import re

        if c.quantization != "none" or c.device == "mps" or c.attention != "sdpa":
            raise ValueError("PersonaPlex requires unquantized CPU/CUDA with native SDPA")
        if any(
            v is not None
            for v in (c.speech_codes, c.speech_assets, c.speech_tokenizer, c.speech_reference)
        ):
            raise ValueError("PersonaPlex uses bundled frozen Mimi and personaplex_voice")
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.pt", c.personaplex_voice):
            raise ValueError("personaplex_voice must name a bundled .pt file")
        safe = {
            "temporal_model": {"out_proj"},
            "text_head": {"text_linear"},
            "audio_projector": {str(i) for i in range(16)},
            "speech_generator": {str(i) for i in range(16)},
        }
        for name, policy in c.components.items():
            if policy.mode == "frozen":
                continue
            if name not in {*safe, "code_predictor"}:
                raise ValueError("PersonaPlex freezes Mimi and cached-voice input embeddings")
            if name == "text_head" and not c.text_loss_weight:
                raise ValueError("text_head tuning requires positive text_loss_weight")
            if policy.mode == "lora" and (
                name not in safe or not set(policy.target_modules) <= safe[name]
            ):
                raise ValueError(
                    "PersonaPlex LoRA must target executed projections; fused weight-only paths require full tuning"
                )
    if c.adapter == "qwen_image_edit":
        if c.quantization != "none" or c.device == "mps" or c.attention != "sdpa":
            raise ValueError("Qwen-Image-Edit requires unquantized CPU/CUDA with native SDPA")
        allowed = {
            "image_generator",
            "latent_projector",
            "text_projector",
            "timestep_embedder",
            "image_head",
        }
        if any(p.mode != "frozen" and k not in allowed for k, p in c.components.items()):
            raise ValueError("Qwen-Image-Edit prepared training freezes the VL encoder and VAE")
    if c.adapter == "cosyvoice3":
        import re

        if c.quantization != "none" or c.device == "mps" or c.attention != "sdpa":
            raise ValueError("CosyVoice3 requires unquantized CPU/CUDA with native SDPA")
        if any(v is not None for v in (c.speech_codes, c.speech_assets, c.speech_tokenizer)):
            raise ValueError("CosyVoice3 uses its bundled native frozen ONNX assets")
        if c.speech_reference is not None and not re.fullmatch(
            "[0-9a-f]{64}", c.speech_reference_sha256 or ""
        ):
            raise ValueError("fixed speech_reference requires its SHA-256")
        if c.speech_reference_text is not None and "<|" in c.speech_reference_text:
            raise ValueError("reference transcript must be literal text without control tokens")
    if c.adapter == "qwen3_tts":
        import re

        if c.quantization != "none" or c.device == "mps":
            raise ValueError("Qwen3-TTS requires unquantized CPU/CUDA speech SFT")
        if c.speech_reference is None or not re.fullmatch(
            "[0-9a-f]{64}", c.speech_reference_sha256 or ""
        ):
            raise ValueError(
                "Qwen3-TTS single-speaker SFT requires a fixed speech_reference and its SHA-256"
            )
        if not re.fullmatch("[a-z][a-z0-9_]{0,63}", c.speech_speaker):
            raise ValueError("Qwen3-TTS speaker must be a lowercase identifier")
        if any(v is not None for v in (c.speech_codes, c.speech_tokenizer, c.speech_assets)):
            raise ValueError("Qwen3-TTS uses the native 12 Hz tokenizer inside its checkpoint")
        if c.components.get("speaker_encoder", ComponentPolicy()).mode != "frozen":
            raise ValueError("Qwen3-TTS uses a fixed frozen speaker encoder")

    if c.adapter == "bagel_dit":
        if c.quantization != "none" or c.device == "mps" or c.attention != "sdpa":
            raise ValueError("BAGEL DiT requires unquantized CPU/CUDA with native SDPA")
        if c.device == "cuda" and c.dtype != "bfloat16":
            raise ValueError("BAGEL native CUDA attention requires BF16")
        allowed = {"image_generator", "latent_projector", "timestep_embedder", "image_head"}
        if any(p.mode != "frozen" and k not in allowed for k, p in c.components.items()):
            raise ValueError(
                "BAGEL DiT tunes only generation components; VAE/understanding stay frozen"
            )
    if c.adapter == "voxcpm2":
        if c.quantization != "none" or c.attention != "sdpa" or c.device == "mps":
            raise ValueError("VoxCPM2 requires unquantized CPU/CUDA speech SFT with native SDPA")
        if any(v is not None for v in (c.speech_codes, c.speech_tokenizer, c.speech_assets)):
            raise ValueError("VoxCPM2 encodes target waveforms with its checkpoint AudioVAE")
        if c.components.get("audio_vae", ComponentPolicy()).mode != "frozen":
            raise ValueError("VoxCPM2 prepared latent training requires a frozen AudioVAE")
    if c.adapter == "qwen3_omni":
        if c.quantization != "none":
            raise ValueError("Qwen3 requires unquantized understanding or speech SFT")
        if any(
            c.components.get(n, ComponentPolicy()).mode != "frozen"
            for n in ("language_router", "speech_router")
        ):
            raise ValueError(
                "Qwen3 router tuning needs a separately normalized auxiliary objective"
            )
    if c.feature_cache is not None:
        if c.adapter != "qwen2_5_omni" or c.task != "understanding_sft" or c.quantization != "none":
            raise ValueError("feature cache currently requires unquantized Qwen understanding SFT")
        if not c.feature_cache_components or len(set(c.feature_cache_components)) != len(
            c.feature_cache_components
        ):
            raise ValueError("feature_cache_components must be nonempty and unique")
    if c.packing != "none" and (
        c.adapter != "qwen2_5_omni"
        or c.task != "understanding_sft"
        or c.device != "cuda"
        or c.dtype != "bfloat16"
        or c.attention != "sdpa"
        or c.quantization != "none"
    ):
        raise ValueError(
            "packing requires unquantized Qwen Thinker understanding SFT, CUDA BF16 and SDPA encoders"
        )
    if c.duplex_schedule is not None and (
        c.adapter != "minicpm_o_4_5" or c.task != "understanding_sft"
    ):
        raise ValueError("timed native attention currently supports MiniCPM understanding SFT")
    if c.adapter == "minicpm_token2wav" and (
        c.attention != "sdpa" or c.quantization != "none" or c.device == "mps"
    ):
        raise ValueError("flow training supports unquantized CPU/CUDA execution with SDPA")
    if c.adapter == "mimo_audio":
        if c.task == "understanding_sft" and c.text_loss_weight == 0:
            raise ValueError("MiMo understanding SFT requires positive text_loss_weight")
        if c.quantization != "none":
            raise ValueError("MiMo NF4 is not verified; use unquantized component LoRA")
        if c.speech_assets is not None or c.speech_codes is not None:
            raise ValueError("MiMo uses speech_tokenizer directory, not S3 assets or Qwen labels")
    if c.task == "speech_sft":
        if c.adapter in ("qwen2_5_omni", "qwen3_omni"):
            if c.speech_codes is None:
                raise ValueError(
                    "Qwen speech_sft requires a native speech_codes manifest; arbitrary waveform encoding is unavailable"
                )
            if c.quantization != "none":
                raise ValueError(
                    "Qwen Talker NF4 training is not verified; use unquantized speech SFT"
                )
            if c.speech_tokenizer is not None or c.speech_assets is not None:
                raise ValueError(
                    "Qwen uses native code labels and checkpoint Token2Wav, not MiniCPM S3 assets"
                )
        elif c.adapter in ("minicpm_o_4_5", "mimo_audio") and (
            c.speech_tokenizer is None or c.speech_tokenizer_sha256 is None
        ):
            raise ValueError("speech_sft requires a local native tokenizer and its fingerprint")
