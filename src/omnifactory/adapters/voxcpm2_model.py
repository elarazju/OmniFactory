"""Pinned native VoxCPM2 CFM/stop objective and strict native weight loading."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint
from transformers import LlamaTokenizerFast

from omnifactory._vendor.voxcpm2.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
from omnifactory._vendor.voxcpm2.modules.audiovae import AudioVAEV2
from omnifactory._vendor.voxcpm2.modules.minicpm4.model import MiniCPMDecoderLayer
from omnifactory.errors import OmniFactoryError


class VoxCPMRuntimeConfig(VoxCPMConfig):
    def to_dict(self):
        """PEFT's non-Transformers model configuration protocol."""
        return self.model_dump()


def model_root(config):
    root = Path(config.model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(config.model, revision=config.revision))
    pointer = root / "voxcpm_processor.json"
    if pointer.exists():
        root = Path(json.loads(pointer.read_text())["source"])
    return root


def native_config(root):
    raw = json.loads((root / "config.json").read_text())
    if raw.get("architecture") != "voxcpm2":
        raise OmniFactoryError("E_NATIVE_CONFIG", "expected VoxCPM2 architecture")
    cfg = VoxCPMRuntimeConfig.model_validate(raw)
    if cfg.dit_config.dit_mean_mode or raw["dit_config"].get("mean_mode", False):
        raise OmniFactoryError("E_NATIVE_CONFIG", "VoxCPM2 mean-flow JVP training is not verified")
    if cfg.audio_vae_config is None or cfg.feat_dim != cfg.audio_vae_config.latent_dim:
        raise OmniFactoryError("E_NATIVE_CONFIG", "explicit matching AudioVAE V2 config required")
    if (
        cfg.dit_config.cfm_config.noise_cond_prob_range[0]
        != cfg.dit_config.cfm_config.noise_cond_prob_range[1]
    ):
        raise OmniFactoryError("E_NATIVE_CONFIG", "scheduled conditioning noise is not implemented")
    return cfg


def read_weights(root, stem, legacy):
    if (root / f"{stem}.safetensors").exists():
        return load_file(root / f"{stem}.safetensors")
    path = root / legacy
    if not path.exists():
        raise OmniFactoryError("E_NATIVE_WEIGHTS", f"missing {stem} weights")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state.get("state_dict", state)


def load_vae(root, cfg, *, device="cpu"):
    vae = AudioVAEV2(cfg.audio_vae_config)
    vae.load_state_dict(read_weights(root, "audiovae", "audiovae.pth"), strict=True)
    return vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


class VoxCPMTrainingModel(VoxCPM2Model):
    """Native parameter names and forward; checkpoint only the native decoder blocks."""

    def gradient_checkpointing_enable(self, **kwargs):
        for layer in self.modules():
            if not isinstance(layer, MiniCPMDecoderLayer) or hasattr(layer, "_of_forward"):
                continue
            layer._of_forward = layer.forward

            def forward(*args, _layer=layer, **kw):
                if _layer.training and torch.is_grad_enabled():
                    return checkpoint(_layer._of_forward, *args, use_reentrant=False, **kw)
                return _layer._of_forward(*args, **kw)

            layer.forward = forward


def load_voxcpm2(config):
    root = model_root(config)
    cfg = native_config(root)
    cfg.device, cfg.dtype = "cpu", "float32"
    # Cache is only needed for generation. Training uses independent causal examples.
    cfg.max_length = config.max_length
    tokenizer = LlamaTokenizerFast.from_pretrained(root, local_files_only=True)
    model = VoxCPMTrainingModel(cfg, tokenizer, AudioVAEV2(cfg.audio_vae_config), device="cpu")
    weights = read_weights(root, "model", "pytorch_model.bin")
    vae_weights = read_weights(root, "audiovae", "audiovae.pth")
    for key, value in vae_weights.items():
        full = "audio_vae." + key
        if full in weights and not torch.equal(weights[full], value):
            raise OmniFactoryError("E_NATIVE_WEIGHTS", f"conflicting VAE tensor: {key}")
        weights[full] = value
    model.load_state_dict(weights, strict=True)
    del weights, vae_weights
    # Never round the frozen FP32 VAE through BF16: preparation and decoding must
    # use the identical original encoder/decoder weights, including in BF16 SFT.
    vae = model.audio_vae
    del model.audio_vae
    model.to(device=config.device, dtype=getattr(torch, config.dtype))
    model.audio_vae = vae.to(device=config.device, dtype=torch.float32).eval().requires_grad_(False)
    model.device = config.device
    model.config.device, model.config.dtype = config.device, config.dtype
    model.base_lm.kv_cache = model.residual_lm.kv_cache = None
    return model


def generation_cache(model, length):
    if length > model.config.max_length:
        raise OmniFactoryError("E_GENERATION_LENGTH", "prompt and generation exceed max_length")
    for lm in (model.base_lm, model.residual_lm):
        lm.setup_cache(1, length, model.device, model._dtype())
