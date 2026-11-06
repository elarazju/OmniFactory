"""Small random native MoE/deepstack model, never pretrained quality evidence."""

from pathlib import Path
from tempfile import TemporaryDirectory

from omnifactory.training.artifacts import new_directory, save_processor


def create_tiny_qwen3(output: Path, seed=42, *, speech=False):
    import torch
    from transformers import (
        Qwen2_5OmniProcessor,
        Qwen3OmniMoeProcessor,
        Qwen3OmniMoeThinkerConfig,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    from omnifactory.training.tiny import create_tiny

    # Reuse only the offline tokenizer and media fixtures, not the architecture.
    with TemporaryDirectory() as tmp:
        create_tiny(Path(tmp) / "tokenizer-fixture", seed=seed)
        old = Qwen2_5OmniProcessor.from_pretrained(Path(tmp) / "tokenizer-fixture")
        processor = Qwen3OmniMoeProcessor(
            image_processor=old.image_processor,
            video_processor=old.video_processor,
            feature_extractor=old.feature_extractor,
            tokenizer=old.tokenizer,
            chat_template=old.chat_template,
        )
    if speech:
        processor.tokenizer.add_tokens(["system", "user", "assistant"])
    torch.manual_seed(seed)
    token = processor.tokenizer.convert_tokens_to_ids
    config = Qwen3OmniMoeThinkerConfig(
        text_config=dict(
            vocab_size=len(processor.tokenizer),
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=2048,
            rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2]},
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=32,
            patch_size=14,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
        audio_config=dict(
            num_mel_bins=8,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            output_dim=32,
            downsample_hidden_size=8,
            max_source_positions=100,
            n_window=50,
            n_window_infer=100,
        ),
        image_token_id=token("<|image_pad|>"),
        video_token_id=token("<|video_pad|>"),
        audio_token_id=token("<|AUDIO|>"),
        vision_start_token_id=token("<|vision_start|>"),
        audio_start_token_id=token("<|audio_bos|>"),
        audio_end_token_id=token("<|audio_eos|>"),
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        position_id_per_seconds=13,
    )
    if speech:
        from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration

        talker = dict(
            spatial_merge_size=2,
            num_code_groups=16,
            thinker_hidden_size=32,
            accept_hidden_layer=1,
            codec_eos_token_id=2150,
            codec_nothink_id=2155,
            codec_think_bos_id=2156,
            codec_think_eos_id=2157,
            codec_pad_id=2148,
            codec_bos_id=2149,
            speaker_id={"ethan": 2145},
            position_id_per_seconds=13,
            image_token_id=token("<|image_pad|>"),
            video_token_id=token("<|video_pad|>"),
            audio_token_id=token("<|AUDIO|>"),
            audio_start_token_id=token("<|audio_bos|>"),
            vision_start_token_id=token("<|vision_start|>"),
            text_config=dict(
                vocab_size=3072,
                hidden_size=32,
                intermediate_size=64,
                moe_intermediate_size=24,
                shared_expert_intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                num_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=2048,
                rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2]},
            ),
            code_predictor_config=dict(
                vocab_size=2048,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                num_code_groups=16,
                max_position_embeddings=128,
            ),
        )
        config = Qwen3OmniMoeConfig(
            thinker_config=config.to_dict(),
            talker_config=talker,
            code2wav_config=dict(
                codebook_size=2048,
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=2,
                num_key_value_heads=1,
                num_hidden_layers=1,
                num_quantizers=16,
                upsample_rates=[2, 2],
                upsampling_ratios=[2, 2],
                decoder_dim=32,
            ),
            im_start_token_id=token("<|im_start|>"),
            im_end_token_id=token("<|im_end|>"),
            tts_pad_token_id=token("<|tts_text_pad|>"),
            tts_bos_token_id=token("<|tts_text_bos|>"),
            tts_eos_token_id=token("<|tts_text_eos|>"),
            system_token_id=token("system"),
            user_token_id=token("user"),
            assistant_token_id=token("assistant"),
        )
        model = Qwen3OmniMoeForConditionalGeneration(config)
    else:
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    with new_directory(output) as root:
        model.save_pretrained(root)
        save_processor(processor, root)
    return output
