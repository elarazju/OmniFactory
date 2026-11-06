"""Random native editing pipeline for CPU contract tests, never quality evidence."""

from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from omnifactory.training.artifacts import new_directory, write_json


def create_tiny_qwen_image_edit(output: Path, seed=42):
    torch.manual_seed(seed)
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Change the red circle to blue. Keep the white background. user assistant system"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    with new_directory(output) as root:
        folder = root / "tokenizer"
        folder.mkdir()
        backend.model.save(str(folder))
        tokenizer = Qwen2Tokenizer(
            vocab_file=str(folder / "vocab.json"),
            merges_file=str(folder / "merges.txt"),
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
            additional_special_tokens=specials,
        )
        processor = Qwen2VLProcessor(
            image_processor=Qwen2VLImageProcessor(min_pixels=56**2, max_pixels=56**2),
            tokenizer=tokenizer,
            video_processor=Qwen2VLVideoProcessor(),
        )
        token_id = tokenizer.convert_tokens_to_ids
        config = Qwen2_5_VLConfig(
            text_config=dict(
                vocab_size=len(tokenizer),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
            ),
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                out_hidden_size=32,
                fullatt_block_indexes=[0],
            ),
            image_token_id=token_id("<|image_pad|>"),
            video_token_id=token_id("<|video_pad|>"),
            vision_start_token_id=token_id("<|vision_start|>"),
            vision_end_token_id=token_id("<|vision_end|>"),
        )
        encoder = Qwen2_5_VLForConditionalGeneration(config)
        transformer = QwenImageTransformer2DModel(
            num_layers=1,
            num_attention_heads=2,
            attention_head_dim=16,
            axes_dims_rope=(4, 6, 6),
            joint_attention_dim=32,
        )
        vae = AutoencoderKLQwenImage(
            base_dim=4,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
        )
        pipe = QwenImageEditPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=vae,
            text_encoder=encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
        )
        pipe.save_pretrained(root)
        write_json(
            root / "omnifactory_fixture.json",
            dict(
                random_initialization=True,
                quality_benchmark=False,
                seed=seed,
            ),
        )
    return {"output": str(output.resolve()), "random_weights": True}
