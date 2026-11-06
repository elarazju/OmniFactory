"""Offline random fixture using OpenBMB's actual MiniCPM-o architecture."""

from pathlib import Path

from omnifactory.training.artifacts import new_directory, write_json


def create_tiny_minicpm(output: Path, seed=42, *, speech=False):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from omnifactory.adapters.minicpm import (
        MiniCPMAAudioProcessor,
        MiniCPMO,
        MiniCPMOConfig,
        MiniCPMOProcessor,
        MiniCPMOTokenizerFast,
        MiniCPMVImageProcessor,
        save_processor,
    )

    torch.manual_seed(seed)
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<unk>",
        "<image>",
        "</image>",
        "<slice>",
        "</slice>",
        "<image_id>",
        "</image_id>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|spk_bos|>",
        "<|spk_eos|>",
        "<|tts_bos|>",
        "<|tts_eos|>",
        "<|listen|>",
    ]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [
            "user assistant system\nWhat color is this image? red blue\n",
            "Is the tone low or high? low high\nRepeat 0123456789.\n",
        ],
        trainers.BpeTrainer(
            vocab_size=320,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = MiniCPMOTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<unk>",
        additional_special_tokens=specials,
    )
    tokenizer.chat_template = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if use_tts_template %}{{ '<|tts_bos|>' }}{% endif %}{% endif %}"""
    processor = MiniCPMOProcessor(
        MiniCPMVImageProcessor(
            scale_resolution=56,
            max_slice_nums=1,
            patch_size=14,
            image_feature_size=4,
            use_image_id=True,
            version=4.5,
        ),
        MiniCPMAAudioProcessor(feature_size=80, chunk_length=1),
        tokenizer,
    )
    config = MiniCPMOConfig(
        version="4.5",
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        query_num=4,
        image_size=56,
        drop_vision_last_layer=False,
        slice_config={"scale_resolution": 56, "max_slice_nums": 1, "patch_size": 14},
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 14,
            "image_size": 56,
        },
        audio_config={
            "num_mel_bins": 80,
            "d_model": 32,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 128,
            "max_source_positions": 100,
        },
        init_tts=speech,
        tts_config={
            "llm_dim": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "num_audio_tokens": 6562,
            "num_text_tokens": len(tokenizer),
            "audio_bos_token_id": tokenizer.convert_tokens_to_ids("<|tts_bos|>"),
            "text_eos_token_id": tokenizer.convert_tokens_to_ids("<|tts_eos|>"),
            "condition_type": "hidden_text_merge",
            "normalize_projected_hidden": True,
            "max_position_embeddings": 2048,
            "attention_type": "full_attention",
        },
        use_cache=False,
        attn_implementation="sdpa",
    )
    model = MiniCPMO(config)
    with new_directory(output) as root:
        model.save_pretrained(root)
        save_processor(processor, root)
        write_json(
            root / "omnifactory_fixture.json",
            {
                "adapter": "minicpm_o_4_5",
                "random_initialization": True,
                "quality_benchmark": False,
                "seed": seed,
                "parameters": model.num_parameters(),
            },
        )
    return {
        "model": str(output.resolve()),
        "parameters": model.num_parameters(),
        "random_initialization": True,
    }
