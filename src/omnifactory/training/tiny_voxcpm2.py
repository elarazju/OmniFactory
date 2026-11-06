"""Small random native VoxCPM2 and AudioVAE V2 for architectural tests only."""

from pathlib import Path

from omnifactory.training.artifacts import new_directory, write_json


def create_tiny_voxcpm2(output: Path, seed=42):
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaTokenizerFast

    from omnifactory._vendor.voxcpm2.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from omnifactory._vendor.voxcpm2.modules.audiovae import AudioVAEV2

    torch.manual_seed(seed)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    specials = ["<unk>", "<s>", "</s>"] + [f"<reserved_{i}>" for i in range(3, 105)]
    backend.train_from_iterator(
        ["Hello world. Read this sentence. A short voice."],
        trainers.BpeTrainer(
            vocab_size=384,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = LlamaTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token=False,
        add_eos_token=False,
    )
    cfg = VoxCPMConfig.model_validate(
        {
            "lm_config": dict(
                bos_token_id=1,
                eos_token_id=2,
                hidden_size=32,
                kv_channels=8,
                intermediate_size=64,
                max_position_embeddings=1024,
                num_attention_heads=4,
                num_hidden_layers=2,
                num_key_value_heads=2,
                rms_norm_eps=1e-5,
                vocab_size=len(tokenizer),
                use_mup=False,
                scale_emb=1,
                dim_model_base=32,
                scale_depth=1,
                rope_theta=10000,
                rope_scaling=dict(
                    type="longrope",
                    long_factor=[1.0] * 4,
                    short_factor=[1.0] * 4,
                    original_max_position_embeddings=1024,
                ),
            ),
            "patch_size": 4,
            "feat_dim": 4,
            "residual_lm_num_layers": 1,
            "residual_lm_no_rope": True,
            "scalar_quantization_latent_dim": 8,
            "encoder_config": dict(
                hidden_dim=32, ffn_dim=64, num_heads=4, num_layers=1, kv_channels=8
            ),
            "dit_config": dict(
                kv_channels=8,
                hidden_dim=32,
                ffn_dim=64,
                num_heads=4,
                num_layers=1,
                cfm_config=dict(t_scheduler="log-norm", training_cfg_rate=0.1),
            ),
            "audio_vae_config": dict(
                encoder_dim=4,
                encoder_rates=[2, 5, 8, 8],
                latent_dim=4,
                decoder_dim=64,
                decoder_rates=[8, 6, 5, 2, 2, 2],
                sample_rate=16000,
                out_sample_rate=48000,
                cond_dim=8,
                sr_bin_boundaries=None,
            ),
            "max_length": 512,
            "device": "cpu",
            "dtype": "float32",
        }
    )
    model = VoxCPM2Model(cfg, tokenizer, AudioVAEV2(cfg.audio_vae_config), device="cpu")
    with new_directory(output) as root:
        tokenizer.save_pretrained(root)
        write_json(root / "config.json", {"architecture": "voxcpm2", **cfg.model_dump()})
        state = model.state_dict()
        save_file(
            {k: v.contiguous() for k, v in state.items() if not k.startswith("audio_vae.")},
            root / "model.safetensors",
        )
        save_file(
            {
                k.removeprefix("audio_vae."): v.contiguous()
                for k, v in state.items()
                if k.startswith("audio_vae.")
            },
            root / "audiovae.safetensors",
        )
        write_json(
            root / "omnifactory_fixture.json",
            {"seed": seed, "random_initialization": True, "quality_benchmark": False},
        )
    return {"model": str(output.resolve()), "random_initialization": True}
