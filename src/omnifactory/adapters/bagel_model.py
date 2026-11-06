"""Native BAGEL generation MoT, retaining original parameter names for export."""

import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from omnifactory._vendor.bagel.modeling.autoencoder import AutoEncoder, AutoEncoderParams
from omnifactory._vendor.bagel.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from omnifactory._vendor.bagel.modeling.bagel.qwen2_navit import Qwen2MoTDecoderLayer
from omnifactory.errors import OmniFactoryError


def model_root(config):
    root = Path(config.model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(config.model, revision=config.revision))
    pointer = root / "bagel_processor.json"
    if pointer.exists():
        root = Path(json.loads(pointer.read_text())["source"])
    return root


def vae_config(root):
    fixture = root / "omnifactory_tiny_vae.json"
    if fixture.exists():
        return AutoEncoderParams(**json.loads(fixture.read_text()))
    return AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )


def load_vae(root, device="cpu"):
    with init_empty_weights():
        vae = AutoEncoder(vae_config(root))
    vae.load_state_dict(load_file(root / "ae.safetensors"), strict=True, assign=True)
    return vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def native_config(root):
    llm = Qwen2Config.from_json_file(root / "llm_config.json")
    llm.qk_norm, llm.tie_word_embeddings = True, False
    llm.layer_module = "Qwen2MoTDecoderLayer"
    llm.freeze_und = True
    # Custom attention is implemented by native packed MoT; HF must not probe FA2
    # at construction, particularly in CPU reference execution.
    llm._attn_implementation = "eager"
    raw = json.loads((root / "config.json").read_text())
    # Official config.json advertises 32, while the native app/checkpoint uses
    # a 64x64 table. Derive geometry from tensor metadata, without loading it.
    with safe_open(root / "ema.safetensors", framework="pt", device="cpu") as source:
        shape = source.get_slice("latent_pos_embed.pos_embed").get_shape()
    side = math.isqrt(shape[0])
    if len(shape) != 2 or side * side != shape[0] or shape[1] != llm.hidden_size:
        raise OmniFactoryError("E_NATIVE_WEIGHTS", "Invalid BAGEL latent position table geometry")
    return BagelConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm,
        vae_config=vae_config(root),
        latent_patch_size=raw.get("latent_patch_size", 2),
        max_latent_size=side,
        timestep_shift=raw.get("timestep_shift", 1.0),
        interpolate_pos=raw.get("interpolate_pos", False),
    )


@contextmanager
def native_training_dispatch(model):
    """Native forward uses train/eval for signature dispatch, not just dropout.

    Teacher-forced evaluation needs the training signature with gradients disabled.
    Native layers have no dropout; preserve every caller mode after execution.
    """
    modules = [(m, m.training) for m in model.modules()]
    try:
        for module, _ in modules:
            module.training = True
        yield
    finally:
        for module, training in modules:
            module.training = training


class BagelTrainingModel(Bagel):
    def gradient_checkpointing_enable(self, **kwargs):
        for layer in self.language_model.model.layers:
            if not isinstance(layer, Qwen2MoTDecoderLayer) or hasattr(layer, "_of_forward_train"):
                continue
            layer._of_forward_train = layer.forward_train

            def forward(*args, _layer=layer, **kw):
                if torch.is_grad_enabled():
                    return checkpoint(_layer._of_forward_train, *args, use_reentrant=False, **kw)
                return _layer._of_forward_train(*args, **kw)

            layer.forward_train = forward

    def forward(self, **kwargs):
        with native_training_dispatch(self):
            return super().forward(**kwargs)


def load_bagel(config):
    root = model_root(config)
    native = native_config(root)
    with init_empty_weights():
        model = BagelTrainingModel(Qwen2ForCausalLM(native.llm_config), None, native)
    filename = root / "ema.safetensors"
    if not filename.exists():
        raise OmniFactoryError("E_NATIVE_WEIGHTS", "BAGEL requires native ema.safetensors")
    keys = set(model.state_dict())
    with safe_open(filename, framework="pt", device="cpu") as source:
        if keys - set(source.keys()):
            raise OmniFactoryError(
                "E_NATIVE_WEIGHTS", "BAGEL checkpoint misses generation parameters"
            )
        state = {key: source.get_tensor(key).to(dtype=getattr(torch, config.dtype)) for key in keys}
    model.load_state_dict(state, strict=True, assign=True)
    model.to(config.device)
    return model
