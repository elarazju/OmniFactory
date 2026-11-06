"""Tiny native 16-codebook TTS, frozen speaker encoder and 12 Hz codec fixture."""

from pathlib import Path

from omnifactory.training.artifacts import new_directory, write_json


def create_tiny_qwen3_tts(output: Path, seed=42):
    import torch
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import EncodecFeatureExtractor, PreTrainedTokenizerFast

    from omnifactory._vendor.qwen3_tts.core.models import (
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
    )
    from omnifactory._vendor.qwen3_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Config,
    )
    from omnifactory._vendor.qwen3_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    torch.manual_seed(seed)
    vocabulary = [
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<|tts_pad|>",
        "<|tts_bos|>",
        "<|tts_eos|>",
        "assistant",
        "\n",
        "Hello",
        "world",
        ".",
        "Read",
        "this",
        "sentence",
        " ",
        "Test",
    ]
    backend = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(vocabulary)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"assistant|\n|\w+|[^\w\n]"), behavior="isolated"
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<|tts_pad|>",
        additional_special_tokens=vocabulary[1:6],
    )
    talker = dict(
        vocab_size=3072,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        text_hidden_size=24,
        text_vocab_size=len(tokenizer),
        num_code_groups=16,
        codec_eos_token_id=2150,
        codec_nothink_id=2155,
        codec_think_bos_id=2156,
        codec_think_eos_id=2157,
        codec_pad_id=2148,
        codec_bos_id=2149,
        spk_id={},
        spk_is_dialect={},
        codec_language_id={"english": 2050},
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "interleaved": True},
        code_predictor_config=dict(
            vocab_size=2048,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=8,
            num_code_groups=16,
        ),
    )
    cfg = Qwen3TTSConfig(
        talker_config=talker,
        tts_model_type="base",
        tts_model_size="tiny",
        tokenizer_type="qwen3_tts_tokenizer_12hz",
        im_start_token_id=1,
        im_end_token_id=2,
        tts_pad_token_id=3,
        tts_bos_token_id=4,
        tts_eos_token_id=5,
        speaker_encoder_config=dict(
            enc_dim=32,
            enc_channels=[8, 8, 8, 8, 24],
            enc_attention_channels=8,
            enc_res2net_scale=2,
            enc_se_channels=8,
        ),
    )
    model = Qwen3TTSForConditionalGeneration(cfg)
    encoder = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_filters=2,
        codebook_size=2048,
        codebook_dim=8,
        vector_quantization_hidden_dimension=8,
        upsample_groups=16,
        num_quantizers=16,
        upsampling_ratios=[8, 6, 5, 4],
        frame_rate=12.5,
    )
    decoder = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        latent_dim=16,
        codebook_dim=16,
        codebook_size=2048,
        decoder_dim=64,
        num_quantizers=16,
    )
    codec_cfg = Qwen3TTSTokenizerV2Config(encoder_config=encoder, decoder_config=decoder)
    codec = Qwen3TTSTokenizerV2Model(codec_cfg)
    with new_directory(output) as root:
        from omnifactory.adapters.qwen3_tts import save_native_model

        save_native_model(model, root)
        tokenizer.save_pretrained(root)
        codec.save_pretrained(root / "speech_tokenizer")
        EncodecFeatureExtractor(sampling_rate=24000, return_attention_mask=True).save_pretrained(
            root / "speech_tokenizer"
        )
        write_json(root / "generation_config.json", {"do_sample": False})
        write_json(
            root / "omnifactory_fixture.json",
            {"random_initialization": True, "seed": seed, "quality_benchmark": False},
        )
    return {"model": str(output.resolve()), "random_initialization": True}
