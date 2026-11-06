"""Qwen-Image-Edit native DiT and frozen, separately loaded conditioning pipeline."""

import json
from pathlib import Path

import torch
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions

from omnifactory.errors import OmniFactoryError


def model_root(config):
    root = Path(config.model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(config.model, revision=config.revision))
    pointer = root / "qwen_image_edit_processor.json"
    if pointer.exists():
        root = Path(json.loads(pointer.read_text())["source"])
    index = json.loads((root / "model_index.json").read_text())
    if index.get("_class_name") != "QwenImageEditPipeline":
        raise OmniFactoryError(
            "E_NATIVE_MODEL", "requires original QwenImageEditPipeline, not EditPlus/2509/2511"
        )
    return root


class QwenImageEditTrainingModel(QwenImageTransformer2DModel):
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        from functools import partial

        from torch.utils.checkpoint import checkpoint

        kwargs = {"use_reentrant": False, **(gradient_checkpointing_kwargs or {})}
        self.enable_gradient_checkpointing(partial(checkpoint, **kwargs))


def load_qwen_image_edit(config):
    # No text encoder or VAE is loaded in the optimizer process. Prepared tensors
    # are the frozen frontend's contract boundary, not a detached trainable layer.
    return QwenImageEditTrainingModel.from_pretrained(
        model_root(config) / "transformer",
        torch_dtype=getattr(torch, config.dtype),
        local_files_only=True,
    ).to(config.device)


def condition_shape(width, height):
    # Match the original native editing pipeline's one-megapixel condition path.
    w, h, _ = calculate_dimensions(1024 * 1024, width / height)
    if min(w, h) < 16:
        raise OmniFactoryError("E_IMAGE_SHAPE", "input aspect ratio produces an empty grid")
    return h, w


class FrozenEditFrontend:
    def __init__(self, root, config):
        self.config = config
        self.pipe = QwenImageEditPipeline.from_pretrained(
            root,
            transformer=None,
            torch_dtype=getattr(torch, config.dtype),
            local_files_only=True,
        ).to(config.device)
        self.pipe.text_encoder.eval().requires_grad_(False)
        self.pipe.vae.eval().requires_grad_(False)

    @torch.no_grad()
    def prompt(self, instruction, image, *, image_tokens):
        pipe = self.pipe
        # The native max_sequence_length argument does not truncate this VL
        # processor. Check the actual expanded sequence before encoder execution.
        inputs = pipe.processor(
            text=[pipe.prompt_template_encode.format(instruction)],
            images=image,
            padding=True,
            return_tensors="pt",
        )
        length = inputs.input_ids.shape[1]
        joint = image_tokens + length - pipe.prompt_template_encode_start_idx
        if (
            length > self.config.max_length
            or joint > self.config.max_length
            or max(length * length, joint * joint) > self.config.max_attention_pairs
        ):
            raise OmniFactoryError("E_SEQUENCE_LENGTH", "VL/DiT sequence exceeds budget")
        patches = int(inputs.image_grid_thw.prod(dim=-1).sum())
        if patches > self.config.budget.max_vision_patches:
            raise OmniFactoryError("E_IMAGE_BUDGET", "expanded vision patches exceed budget")
        # Use the native implementation for hidden-layer selection, prompt prefix
        # removal and masks; never encode the edited target in this path.
        embeds, mask = pipe.encode_prompt(instruction, image=image, device=self.config.device)
        return embeds.cpu(), None if mask is None else mask.cpu(), patches

    @torch.no_grad()
    def latent(self, image, shape):
        h, w = shape
        pixels = self.pipe.image_processor.preprocess(image, h, w).unsqueeze(2)
        pixels = pixels.to(device=self.config.device, dtype=self.pipe.vae.dtype)
        latent = self.pipe._encode_vae_image(pixels, generator=None)
        packed = self.pipe._pack_latents(
            latent,
            batch_size=1,
            num_channels_latents=latent.shape[1],
            height=latent.shape[-2],
            width=latent.shape[-1],
        )
        return packed.cpu()


def flow_prediction(model, *, clean, condition, prompt, mask, img_shapes, sigma, timestep, noise):
    """Only the target patches receive noise; clean source patches remain conditioning."""
    noisy = (1 - sigma) * clean + sigma * noise
    prediction = model(
        hidden_states=torch.cat((noisy, condition), dim=1),
        encoder_hidden_states=prompt,
        encoder_hidden_states_mask=mask,
        timestep=timestep / 1000,
        img_shapes=img_shapes,
        return_dict=False,
    )[0][:, : clean.shape[1]]
    return prediction, noise - clean
