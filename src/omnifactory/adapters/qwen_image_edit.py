"""Image-conditioned flow supervision for the original Qwen-Image-Edit model."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageOps

from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
from omnifactory.adapters.qwen_image_edit_model import (
    FrozenEditFrontend,
    condition_shape,
    flow_prediction,
    model_root,
)
from omnifactory.adapters.supervision import tensor_ref
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view
from omnifactory.data.supervision import (
    ContinuousRepresentation,
    LossObjective,
    Normalizer,
    ReductionMeasure,
    SupervisionPlan,
    SupervisionTarget,
)
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import SampleCost
from omnifactory.training.artifacts import new_directory, write_json
from omnifactory.training.losses import LossBundle, LossTerm, attach_losses


def edit_inputs(view):
    if len(view.messages) != 1 or view.messages[0].role != "user" or view.conditions:
        raise OmniFactoryError("E_IMAGE_INPUT", "expected one user edit instruction and image")
    blocks = view.messages[0].content
    images = [b for b in blocks if b.type == "image"]
    text = "".join(b.text for b in blocks if b.type == "text")
    if (
        len(images) != 1
        or any(b.type not in ("text", "image") for b in blocks)
        or not text.strip()
        or "<|" in text
    ):
        raise OmniFactoryError("E_IMAGE_INPUT", "one source image and literal instruction required")
    return text, images[0]


class QwenImageEditAdapter:
    def __init__(self, config):
        self.config = config
        self.root = model_root(config)
        self.resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.root / "scheduler", local_files_only=True
        )
        self.shape = (config.image_height, config.image_width)
        vae = json.loads((self.root / "vae/config.json").read_text())
        self.stride = 2 ** len(vae["temperal_downsample"]) * 2
        self.channels = vae["z_dim"] * 4
        if any(s % self.stride for s in self.shape):
            raise OmniFactoryError("E_IMAGE_SHAPE", "output dimensions must fit native VAE patches")
        if self.shape[0] * self.shape[1] > config.max_image_pixels:
            raise OmniFactoryError("E_IMAGE_BUDGET", "output image exceeds max_image_pixels")
        self._frontend = None
        # Processor assets are copied by native export; prepared storage retains a
        # source pointer so loading a training adapter does not instantiate VL/VAE.
        self.processor = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def frontend(self):
        if self._frontend is None:
            self._frontend = FrozenEditFrontend(self.root, self.config)
        return self._frontend

    def save_processor(self, path):
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "qwen_image_edit_processor.json", {"source": str(self.root.resolve())})

    def _image(self, asset):
        path = Path(self.resolver.resolve(asset).path)
        with Image.open(path) as image:
            pixels = image.width * image.height
            if pixels > self.config.max_image_pixels:
                raise OmniFactoryError("E_IMAGE_BUDGET", "source image exceeds max_image_pixels")
            if pixels * 16 > self.config.budget.max_decoded_media_bytes:
                raise OmniFactoryError("E_DECODE_BUDGET", "source image exceeds decode budget")
            return ImageOps.exif_transpose(image).convert("RGB"), path

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        import numpy as np

        view = inference_view(sample, prediction_id, ("image",))
        instruction, block = edit_inputs(view)
        source, _ = self._image(view.assets[block.asset_id])
        cond_shape = condition_shape(*source.size)
        if cond_shape[0] * cond_shape[1] > self.config.max_image_pixels:
            raise OmniFactoryError("E_IMAGE_BUDGET", "native condition resize exceeds pixel budget")
        shapes = [(1, h // self.stride, w // self.stride) for h, w in (self.shape, cond_shape)]
        image_tokens = sum(h * w for _, h, w in shapes)
        decoded_bytes = 16 * (
            source.width * source.height
            + cond_shape[0] * cond_shape[1]
            + self.shape[0] * self.shape[1]
        )
        if decoded_bytes > self.config.budget.max_decoded_media_bytes:
            raise OmniFactoryError("E_DECODE_BUDGET", "combined image decode exceeds budget")
        if (
            image_tokens > self.config.max_length
            or image_tokens**2 > self.config.max_attention_pairs
        ):
            raise OmniFactoryError("E_SEQUENCE_LENGTH", "latent sequence exceeds budget")
        latent_elements = image_tokens * self.channels
        if (
            self.config.budget.max_latent_elements is not None
            and latent_elements > self.config.budget.max_latent_elements
        ):
            raise OmniFactoryError("E_IMAGE_BUDGET", "condition and target latents exceed budget")
        frontend = self.frontend()
        resized = frontend.pipe.image_processor.resize(source, *cond_shape)
        prompt, mask, patches = frontend.prompt(instruction, resized, image_tokens=image_tokens)
        condition = frontend.latent(resized, cond_shape)
        payload = dict(
            prompt=prompt,
            prompt_mask=mask,
            condition=condition,
            img_shapes=[shapes],
        )
        cost = SampleCost(
            sample_id=sample.id,
            prediction_id=prediction_id,
            contract_id=contract_id,
            modalities=("text", "image"),
            llm_tokens=image_tokens + prompt.shape[1],
            vision_patches=patches,
            latent_elements=latent_elements,
            decoded_media_bytes=decoded_bytes,
        )
        if inference:
            negative, negative_mask, _ = frontend.prompt(" ", resized, image_tokens=image_tokens)
            payload.update(
                source_pixels=torch.from_numpy(np.array(source, copy=True)),
                negative_prompt=negative,
                negative_mask=negative_mask,
            )
            tensors, layout = pack_tree(payload)
            cost = cost.model_copy(
                update={
                    "llm_tokens": image_tokens + max(prompt.shape[1], negative.shape[1]),
                }
            )
            return EncodedExample(tensors, cost, layout=layout).validate(inference=True)
        prediction = next(p for p in sample.predictions if p.id == prediction_id)
        message = next(m for m in sample.messages if m.id == prediction.target_message_id)
        targets = [b for b in message.content if b.id in prediction.target_block_ids]
        if len(targets) != 1 or targets[0].type != "image":
            raise OmniFactoryError("E_IMAGE_TARGET", "select exactly one edited target image")
        target, _ = self._image(sample.assets[targets[0].asset_id])
        decoded_bytes += target.width * target.height * 16
        if decoded_bytes > self.config.budget.max_decoded_media_bytes:
            raise OmniFactoryError("E_DECODE_BUDGET", "target plus condition exceeds decode budget")
        # The target enters only the VAE, never the instruction/vision encoder.
        clean = frontend.latent(target, self.shape)
        payload["clean"] = clean
        tensors, layout = pack_tree(payload)
        tensors["target_mask"] = torch.ones_like(clean, dtype=torch.bool)
        plan = SupervisionPlan(
            targets=(
                SupervisionTarget(
                    id="edited_image",
                    modality="image",
                    representation=ContinuousRepresentation(
                        space="qwen_image_vae", encoder_id="native-frozen-vae-mode"
                    ),
                    values=tensor_ref("clean", clean, ("batch", "patch", "feature")),
                    mask=tensor_ref(
                        "target_mask", tensors["target_mask"], ("batch", "patch", "feature")
                    ),
                    measure=ReductionMeasure(unit="elements"),
                ),
            ),
            normalizers=(Normalizer(id="latent_elements", unit="elements"),),
            objectives=(
                LossObjective(
                    id="flow",
                    target_id="edited_image",
                    kind="flow_matching",
                    normalizer="latent_elements",
                ),
            ),
        )
        return EncodedExample(
            tensors,
            cost.model_copy(update={"decoded_media_bytes": decoded_bytes}),
            layout=layout,
            supervision=plan,
            metadata={
                "posterior": "mode",
                "condition_area": 1048576,
                "target_transform": "native_vae_resize",
                "timestep_sampling": "uniform",
            },
        ).validate()

    def collate(self, rows):
        # Native image/rotary layouts may differ. Budget group without padding or
        # sharing cross-example attention; common loss sums normalize the window.
        return rows[0] if len(rows) == 1 else {"examples": rows}

    @staticmethod
    def to_device(batch, model, config):
        return move_tree(batch, device=config.device, dtype=getattr(torch, config.dtype))

    def forward(self, model, batch):
        if "examples" in batch:
            return attach_losses(
                SimpleNamespace(),
                LossBundle.combine([self.forward(model, row).losses for row in batch["examples"]]),
            )
        clean = batch["clean"]
        noise = torch.randn_like(clean)
        # Uniform sampling with native scheduler sigmas, matching the unweighted
        # Diffusers Qwen training objective. All draws use checkpointed Torch RNG.
        index = (torch.rand(clean.shape[0]) * self.scheduler.config.num_train_timesteps).long()
        timestep = self.scheduler.timesteps[index].to(clean.device, clean.dtype)
        sigma = self.scheduler.sigmas[index].to(clean.device, clean.dtype).view(-1, 1, 1)
        prediction, target = flow_prediction(
            model,
            clean=clean,
            condition=batch["condition"],
            prompt=batch["prompt"],
            mask=batch["prompt_mask"],
            img_shapes=batch["img_shapes"],
            sigma=sigma,
            timestep=timestep,
            noise=noise,
        )
        numerator = (prediction.float() - target.float()).square().sum()
        return attach_losses(
            SimpleNamespace(),
            LossBundle(
                (LossTerm("flow", numerator, "latent_elements", kind="flow_matching"),),
                {"latent_elements": float(target.numel())},
                {"latent_elements": "elements"},
            ),
        )

    @torch.no_grad()
    def generate_image(self, model, batch, *, num_steps):
        import numpy as np

        if "examples" in batch:
            return [
                self.generate_image(model, row, num_steps=num_steps) for row in batch["examples"]
            ]
        pipe = self.frontend().pipe
        # Retain the actual inference-visible pixels, not a mutable file path:
        # the native VAE re-encode must see the same image as the cached VL prompt.
        image = Image.fromarray(batch["source_pixels"].cpu().numpy())
        # Native pipeline owns the schedule, CFG rescaling, rotary geometry and
        # VAE decode. Cached prompt tensors contain only inference-visible data.
        pipe.transformer = model
        try:
            images = pipe(
                image=image,
                prompt_embeds=batch["prompt"],
                prompt_embeds_mask=batch["prompt_mask"],
                negative_prompt_embeds=batch["negative_prompt"],
                negative_prompt_embeds_mask=batch["negative_mask"],
                height=self.shape[0],
                width=self.shape[1],
                true_cfg_scale=4.0,
                num_inference_steps=num_steps,
                output_type="np",
                generator=torch.Generator(device=self.config.device).manual_seed(self.config.seed),
            ).images
            if not np.isfinite(images).all():
                raise OmniFactoryError("E_IMAGE_OUTPUT", "nonfinite native decoded image")
            return pipe.image_processor.numpy_to_pil(images)[0]
        finally:
            pipe.transformer = None


def export_qwen_image_edit(checkpoint, output, *, format=None):
    from omnifactory.training.engine import load_checkpoint
    from omnifactory.training.policy import merge_lora

    if format not in (None, "merged", "vllm-omni"):
        raise OmniFactoryError(
            "E_EXPORT_FORMAT", "Qwen-Image-Edit requires a native pipeline export"
        )
    model, adapter, config, metadata = load_checkpoint(checkpoint, for_export=True)
    model = merge_lora(model)
    from safetensors import safe_open

    # Shared checkpoint loading merges in FP32. Return to the checkpoint's native
    # floating dtype instead of doubling the deployed DiT memory and file size.
    source_shard = sorted((adapter.root / "transformer").glob("*.safetensors"))[0]
    with safe_open(source_shard, framework="pt") as weights:
        dtype = weights.get_slice(next(iter(weights.keys()))).get_dtype()
    if dtype not in ("F32", "BF16", "F16"):
        raise OmniFactoryError("E_NATIVE_WEIGHTS", "unsupported source DiT dtype")
    model.to(dtype={"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}[dtype])
    with new_directory(output) as root:
        for name in ("text_encoder", "vae", "scheduler", "processor", "tokenizer"):
            shutil.copytree(
                adapter.root / name, root / name, ignore=shutil.ignore_patterns(".cache")
            )
        for name in ("model_index.json", "README.md", "LICENSE", "NOTICE"):
            if (adapter.root / name).is_file():
                shutil.copy2(adapter.root / name, root / name)
        model.save_pretrained(root / "transformer", safe_serialization=True, max_shard_size="2GB")
        # Native consumers require the upstream class, not our checkpointing shim.
        path = root / "transformer/config.json"
        native = json.loads(path.read_text())
        native["_class_name"] = "QwenImageTransformer2DModel"
        write_json(path, native)
        result = dict(
            format=format or "merged",
            adapter=config.adapter,
            inference_config=config.model_dump(mode="json"),
            source_checkpoint=str(Path(checkpoint).resolve()),
            source_fingerprint=metadata["source_fingerprint"],
            output=str(Path(output).resolve()),
            lora_merged=True,
            vllm_deployment_verified=False,
            native_architecture="QwenImageEditPipeline",
        )
        write_json(root / "omnifactory_export.json", result)
    return result


def validate_export(root, metadata):
    model_root(SimpleNamespace(model=str(root)))
    for name in ("text_encoder", "vae", "scheduler", "processor", "tokenizer", "transformer"):
        folder = root / name
        if not folder.is_dir() or not any(folder.iterdir()):
            raise OmniFactoryError("E_EXPORT_ASSET", f"missing native component: {name}")
    for name in ("transformer", "vae", "text_encoder"):
        if not (root / name / "config.json").is_file() or not list(
            (root / name).glob("*.safetensors")
        ):
            raise OmniFactoryError("E_EXPORT_ASSET", f"missing weights/config: {name}")


def build_edit_request(payload, *, view, resolver, metadata):
    """Return multipart fields/files using only the selected inference view."""
    import io

    instruction, block = edit_inputs(view)
    path = resolver.resolve(view.assets[block.asset_id]).path
    config = metadata["inference_config"]
    with Image.open(path) as source:
        if source.width * source.height > config["max_image_pixels"]:
            raise OmniFactoryError("E_IMAGE_BUDGET", "HTTP source exceeds image pixel budget")
        if source.width * source.height * 16 > config["budget"]["max_decoded_media_bytes"]:
            raise OmniFactoryError("E_DECODE_BUDGET", "HTTP source exceeds decode budget")
        image = ImageOps.exif_transpose(source).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    fields = {**payload, "prompt": instruction, "negative_prompt": " ", "true_cfg_scale": 4.0}
    return fields, {"image": ("source.png", "image/png", buffer.getvalue())}
