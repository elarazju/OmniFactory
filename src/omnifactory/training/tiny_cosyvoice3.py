"""Random native CosyVoice3 modules; frontend fixtures are supplied by CPU tests.

This does not manufacture a substitute for the pretrained ONNX speech tokenizer.
"""

from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config

from omnifactory._vendor.cosyvoice3.tokenizer.tokenizer import get_qwen_tokenizer
from omnifactory.adapters.cosyvoice3_model import (
    build_flow,
    build_hift,
    build_talker,
    native_config,
)
from omnifactory.training.artifacts import new_directory, write_json


def create_tiny_cosyvoice3(output, seed=42):
    torch.manual_seed(seed)
    words = [
        "<unk>",
        "<|endoftext|>",
        "Hello",
        "world",
        ".",
        "You",
        "are",
        "a",
        "helpful",
        "assistant",
    ]
    backend = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<|endoftext|>"
    )
    cfg = dict(
        sample_rate=24000,
        llm=dict(
            llm_input_size=32,
            llm_output_size=32,
            speech_token_size=32,
            length_normalized_loss=True,
            lsm_weight=0.0,
            mix_ratio=[5, 15],
            llm={},
            sampling={},
        ),
        flow=dict(
            input_size=80,
            output_size=80,
            spk_embed_dim=192,
            output_type="mel",
            vocab_size=32,
            input_frame_rate=25,
            only_mask_loss=True,
            token_mel_ratio=2,
            pre_lookahead_len=3,
            pre_lookahead_layer=dict(in_channels=80, channels=32, pre_lookahead_len=3),
            decoder=dict(
                in_channels=240,
                n_spks=1,
                spk_emb_dim=80,
                cfm_params=dict(
                    content=dict(
                        sigma_min=1e-6,
                        solver="euler",
                        t_scheduler="cosine",
                        training_cfg_rate=0.2,
                        inference_cfg_rate=0.7,
                        reg_loss_type="l1",
                    )
                ),
                estimator=dict(
                    dim=32,
                    depth=1,
                    heads=2,
                    dim_head=16,
                    ff_mult=2,
                    mel_dim=80,
                    mu_dim=80,
                    spk_dim=80,
                    out_channels=80,
                    static_chunk_size=6,
                    num_decoding_left_chunks=-1,
                ),
            ),
        ),
        hift=dict(
            in_channels=80,
            base_channels=32,
            nb_harmonics=8,
            sampling_rate=24000,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            istft_params=dict(n_fft=16, hop_len=4),
            f0_predictor=dict(num_class=1, in_channels=80, cond_channels=16),
        ),
        feat_extractor=dict(
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=None,
            center=False,
        ),
    )
    tags = dict(
        llm="cosyvoice.llm.llm.CosyVoice3LM",
        flow="cosyvoice.flow.flow.CausalMaskedDiffWithDiT",
        hift="cosyvoice.hifigan.generator.CausalHiFTGenerator",
    )
    with new_directory(Path(output)) as root:
        text = yaml.safe_dump(cfg, sort_keys=False)
        for key, tag in tags.items():
            text = text.replace(f"{key}:\n", f"{key}: !new:{tag}\n", 1)
        (root / "cosyvoice3.yaml").write_text(text)
        tokenizer.save_pretrained(root / "CosyVoice-BlankEN")
        native_tokenizer = get_qwen_tokenizer(str(root / "CosyVoice-BlankEN"), True, "cosyvoice3")
        native_tokenizer.tokenizer.save_pretrained(root / "CosyVoice-BlankEN")
        Qwen2Config(
            vocab_size=len(native_tokenizer.tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        ).save_pretrained(root / "CosyVoice-BlankEN")
        resolved = native_config(root)
        for name, builder, key in (
            ("llm.pt", lambda c: build_talker(root, c), "llm"),
            ("flow.pt", build_flow, "flow"),
            ("hift.pt", build_hift, "hift"),
        ):
            model = builder(resolved[key])
            torch.save(model.state_dict(), root / name)
        write_json(root / "config.json", {})
        write_json(
            root / "fixture.json",
            dict(random_native_modules=True, frontend_assets_supplied_by_tests=True),
        )
    return {
        "model": str(output),
        "scope": "random native CPU modules; not a speech tokenizer/quality fixture",
    }
