"""Offline, randomly initialized Qwen2.5-Omni Thinker for execution tests.

Uses the actual Transformers architecture and processor at reduced dimensions.
It is not a pretrained model, and its output has no quality significance.
"""

from pathlib import Path

from omnifactory.training.artifacts import new_directory, save_processor, write_json


def create_tiny(output: Path, seed: int = 42, *, speech: bool = False):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniThinkerForConditionalGeneration,
        Qwen2TokenizerFast,
        Qwen2VLImageProcessor,
        Qwen2VLVideoProcessor,
        WhisperFeatureExtractor,
    )

    torch.manual_seed(seed)
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|audio_bos|>",
        "<|audio_eos|>",
        "<|AUDIO|>",
        "<|audio_perception|>",
        "<|audio_sentinel|>",
        "<|tts_text_bos|>",
        "<|tts_text_eos|>",
        "<|tts_text_pad|>",
    ]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [
            "user assistant system\nWhat color is this image? red blue\n",
            "Is the tone low or high? low high\nRepeat the number 0123456789.\n",
        ],
        trainers.BpeTrainer(
            vocab_size=320,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        extra_special_tokens={
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
            "audio_token": "<|AUDIO|>",
            "vision_bos_token": "<|vision_start|>",
            "vision_eos_token": "<|vision_end|>",
            "audio_bos_token": "<|audio_bos|>",
            "audio_eos_token": "<|audio_eos|>",
        },
    )
    template = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}{% for block in message['content'] %}{% if block['type'] == 'text' %}{{ block['text'] }}{% elif block['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% elif block['type'] == 'video' %}{{ '<|vision_start|><|video_pad|><|vision_end|>' }}{% elif block['type'] == 'audio' %}{{ '<|audio_bos|><|AUDIO|><|audio_eos|>' }}{% endif %}{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
    processor = Qwen2_5OmniProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=56 * 56),
        video_processor=Qwen2VLVideoProcessor(),
        feature_extractor=WhisperFeatureExtractor(feature_size=8, chunk_length=1),
        tokenizer=tokenizer,
        chat_template=template,
    )
    token_id = tokenizer.convert_tokens_to_ids
    config = Qwen2_5OmniThinkerConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 2048,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
        },
        audio_config={
            "num_mel_bins": 8,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "d_model": 32,
            "output_dim": 32,
            "max_source_positions": 100,
            "n_window": 50,
        },
        image_token_index=token_id("<|image_pad|>"),
        video_token_index=token_id("<|video_pad|>"),
        audio_token_index=token_id("<|AUDIO|>"),
        vision_start_token_id=token_id("<|vision_start|>"),
        audio_start_token_id=token_id("<|audio_bos|>"),
        audio_end_token_id=token_id("<|audio_eos|>"),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = Qwen2_5OmniThinkerForConditionalGeneration(config)
    if speech:
        from transformers import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration

        native = Qwen2_5OmniConfig(
            thinker_config=config.to_dict(),
            talker_config={
                "vocab_size": 8448,
                "embedding_size": 32,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "max_position_embeddings": 2048,
                "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
                "tts_text_start_token_id": token_id("<|tts_text_bos|>"),
                "tts_text_end_token_id": token_id("<|tts_text_eos|>"),
                "tts_text_pad_token_id": token_id("<|tts_text_pad|>"),
                **{
                    k: getattr(config, k)
                    for k in (
                        "image_token_index",
                        "video_token_index",
                        "audio_token_index",
                        "vision_start_token_id",
                        "audio_start_token_id",
                        "audio_end_token_id",
                    )
                },
            },
            token2wav_config={
                "dit_config": {
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "head_dim": 16,
                    "emb_dim": 16,
                    "mel_dim": 8,
                    "enc_dim": 8,
                    "enc_emb_dim": 8,
                    "enc_channels": [16, 16, 16, 16, 48],
                    "enc_attention_channels": 8,
                    "enc_se_channels": 8,
                    "look_ahead_layers": [],
                    "look_backward_layers": [],
                    "dropout": 0,
                },
                "bigvgan_config": {
                    "mel_dim": 8,
                    "upsample_initial_channel": 128,
                    "resblock_kernel_sizes": [3],
                    "resblock_dilation_sizes": [[1, 3, 5]],
                },
            },
            enable_audio_output=True,
        )
        model = Qwen2_5OmniForConditionalGeneration(native)
    with new_directory(output) as root:
        if speech:
            torch.save(
                {
                    "Chelsie": {
                        "bos_token": token_id("<|tts_text_bos|>"),
                        "cond": torch.zeros(1, 8),
                        "ref_mel": torch.zeros(1, 32, 8),
                    }
                },
                root / "spk_dict.pt",
            )
        model.save_pretrained(root)
        save_processor(processor, root)
        write_json(
            root / "omnifactory_fixture.json",
            {
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
