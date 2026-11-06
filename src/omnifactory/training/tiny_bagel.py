"""Small original BAGEL MoT and VAE weights for offline correctness tests."""

from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from omnifactory._vendor.bagel.modeling.autoencoder import AutoEncoder, AutoEncoderParams
from omnifactory._vendor.bagel.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from omnifactory.training.artifacts import new_directory, write_json


def create_tiny_bagel(output: Path, seed=42):
    torch.manual_seed(seed)
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["A red circle. A blue square. A yellow flower on green grass."],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=special,
        pad_token=special[0],
        eos_token=special[2],
    )
    llm = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        layer_module="Qwen2MoTDecoderLayer",
        _attn_implementation="eager",
        freeze_und=True,
    )
    vae_cfg = AutoEncoderParams(
        resolution=32,
        in_channels=3,
        downsample=2,
        ch=32,
        out_ch=3,
        ch_mult=[1, 1],
        num_res_blocks=1,
        z_channels=4,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
    cfg = BagelConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm,
        vae_config=vae_cfg,
        latent_patch_size=2,
        max_latent_size=16,
    )
    model = Bagel(Qwen2ForCausalLM(llm), None, cfg)
    # Nonzero head exercises gradients into the transformer from the first step.
    torch.nn.init.normal_(model.llm2vae.weight, std=0.02)
    vae = AutoEncoder(vae_cfg)
    with new_directory(output) as root:
        tokenizer.save_pretrained(root)
        (root / "llm_config.json").write_text(llm.to_json_string())
        write_json(
            root / "config.json",
            {"architectures": ["Bagel"], "latent_patch_size": 2, "max_latent_size": 16},
        )
        write_json(root / "omnifactory_tiny_vae.json", asdict(vae_cfg))
        save_file(model.state_dict(), root / "ema.safetensors")
        save_file(vae.state_dict(), root / "ae.safetensors")
        write_json(
            root / "fixture.json",
            {"seed": seed, "random_initialization": True, "quality_benchmark": False},
        )
    return {"output": str(output.resolve()), "random_weights": True, "architecture": "bagel_dit"}
