"""Reduced native MiMo global/local transformers and native RVQ codec; random weights."""

from pathlib import Path

from omnifactory.training.artifacts import new_directory, write_json


def create_tiny_mimo(output: Path, seed=42):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import Qwen2TokenizerFast

    from omnifactory._vendor.mimo_audio.codec import MiMoAudioTokenizer, MiMoAudioTokenizerConfig
    from omnifactory.adapters.mimo_model import MiMoAudioConfig, MiMoTrainingModel, model_args
    from omnifactory.training.prepare import source_fingerprint

    torch.manual_seed(seed)
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|sosp|>",
        "<|eosp|>",
        "<|sostm|>",
        "<|eostm|>",
        "<|eot|>",
        "<|empty|>",
    ]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["user assistant system\n<think>\n\n</think>\nSay hello. Hello. low high"],
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
        additional_special_tokens=specials,
    )
    config = MiMoAudioConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=2048,
        group_size=4,
        audio_channels=8,
        speech_vocab_size="17-17-9-9-9-9-9-9",
        speech_zeroemb_idx="16-16-8-8-8-8-8-8",
        delay_pattern="0-1-2-3-4-5-6-7",
        local_dim=16,
        local_layers=1,
        local_attn_heads=2,
        local_ffn_dim=32,
        local_attn_dropout=0.0,
        input_local_layers=1,
        input_local_dim=16,
        input_full_attention=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    model = MiMoTrainingModel(config, model_args(tokenizer, output))
    codec_config = MiMoAudioTokenizerConfig(
        max_audio_seconds=30,
        d_model=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=32,
        vocoder_dim=16,
        vocoder_intermediate_dim=32,
        vocoder_num_layers=1,
        vocoder_attention_heads=2,
        num_quantizers=20,
        codebook_size=[16, 16] + [8] * 18,
        avg_pooler=2,
    )
    codec = MiMoAudioTokenizer(codec_config).eval()
    # The native EMA codebooks are uninitialized in a random model. Initialize fixture
    # buffers explicitly; never perform data-dependent k-means during frozen preparation.
    for name, buffer in codec.named_buffers():
        if name.endswith("inited"):
            buffer.fill_(1)
        elif name.endswith("embed") or name.endswith("embed_avg"):
            buffer.normal_(std=0.1)
    with new_directory(output) as root:
        model.save_pretrained(root)
        tokenizer.save_pretrained(root)
        codec.save_pretrained(root / "assets/mimo_codec")
        fingerprint = source_fingerprint(str(root / "assets/mimo_codec"), None)
        write_json(
            root / "omnifactory_fixture.json",
            {
                "random_initialization": True,
                "quality_benchmark": False,
                "seed": seed,
                "parameters": model.num_parameters(),
                "codec_parameters": codec.num_parameters(),
                "codec_fingerprint": fingerprint,
            },
        )
    return {
        "model": str(output.resolve()),
        "parameters": model.num_parameters(),
        "codec_fingerprint": fingerprint,
        "random_initialization": True,
    }
