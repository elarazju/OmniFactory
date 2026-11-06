"""Text-to-image BAGEL: continuous VAE supervision with native MoT flow matching."""

import hashlib
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer

from omnifactory._vendor.bagel.data.data_utils import (
    add_special_tokens,
    patchify,
    pil_img2rgb,
    prepare_attention_mask_per_sample,
)
from omnifactory._vendor.bagel.modeling.bagel.qwen2_navit import NaiveCache
from omnifactory.adapters.bagel_model import load_vae, model_root, native_config
from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
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
from omnifactory.training.artifacts import write_json
from omnifactory.training.losses import LossBundle, LossTerm, attach_losses


class BagelAdapter:
    def __init__(self, config):
        self.config = config
        self.root = model_root(config)
        self.native = native_config(self.root)
        self.processor, self.special, added = add_special_tokens(
            AutoTokenizer.from_pretrained(self.root, local_files_only=True)
        )
        if max(self.special.values()) >= self.native.llm_config.vocab_size:
            raise OmniFactoryError("E_TOKENIZER", "image boundary tokens exceed native vocabulary")
        self.resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        self.vae = None
        stride = self.native.vae_config.downsample * self.native.latent_patch_size
        self.shape = (config.image_height, config.image_width)
        if any(s % stride or s // stride > self.native.max_latent_size for s in self.shape):
            raise OmniFactoryError(
                "E_IMAGE_SHAPE", "image dimensions must fit native patch/grid limits"
            )
        if config.image_height * config.image_width > config.max_image_pixels:
            raise OmniFactoryError("E_IMAGE_BUDGET", "output image exceeds max_image_pixels")
        self.grid = tuple(s // stride for s in self.shape)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def save_processor(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(path)
        write_json(path / "bagel_processor.json", {"source": str(self.root.resolve())})

    def _vae(self):
        if self.vae is None:
            self.vae = load_vae(self.root, self.config.device)
        return self.vae

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        view = inference_view(sample, prediction_id, ("image",))
        if (
            len(view.messages) != 1
            or view.messages[0].role != "user"
            or view.conditions
            or any(b.type != "text" for b in view.messages[0].content)
        ):
            raise OmniFactoryError(
                "E_IMAGE_INPUT",
                "BAGEL DiT currently expects one user text prompt without conditions",
            )
        text = "".join(b.text for b in view.messages[0].content)
        if not text.strip():
            raise OmniFactoryError("E_IMAGE_INPUT", "empty image prompt")
        prompt = [
            self.special["bos_token_id"],
            *self.processor.encode(text),
            self.special["eos_token_id"],
        ]
        n = self.grid[0] * self.grid[1]
        length = len(prompt) + n + 2
        if length > self.config.max_length or length * length > self.config.max_attention_pairs:
            raise OmniFactoryError(
                "E_SEQUENCE_LENGTH", "text/image sequence exceeds length or attention budget"
            )
        cost = dict(
            sample_id=sample.id,
            prediction_id=prediction_id,
            contract_id=contract_id,
            modalities=("text",),
            llm_tokens=length,
            latent_elements=n
            * self.native.vae_config.z_channels
            * self.native.latent_patch_size**2,
        )
        if inference:
            tensors, layout = pack_tree({"prompt": text, "image_shape": list(self.shape)})
            return EncodedExample(tensors, SampleCost(**cost), layout=layout)
        prediction = next(p for p in sample.predictions if p.id == prediction_id)
        message = next(m for m in sample.messages if m.id == prediction.target_message_id)
        targets = [b for b in message.content if b.id in prediction.target_block_ids]
        if len(targets) != 1 or targets[0].type != "image":
            raise OmniFactoryError("E_IMAGE_TARGET", "select exactly one target image")
        path = self.resolver.resolve(sample.assets[targets[0].asset_id]).path
        with Image.open(path) as image:
            if image.width * image.height > self.config.max_image_pixels:
                raise OmniFactoryError("E_IMAGE_BUDGET", "source image exceeds max_image_pixels")
            decoded_bytes = image.width * image.height * 4 + self.shape[0] * self.shape[1] * 12
            if decoded_bytes > self.config.budget.max_decoded_media_bytes:
                raise OmniFactoryError("E_DECODE_BUDGET", "image exceeds decoded memory budget")
            image = ImageOps.fit(
                pil_img2rgb(image), self.shape[::-1], method=Image.Resampling.BICUBIC
            )
            pixels = (
                torch.from_numpy(np.array(image, dtype=np.float32) / 127.5 - 1)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.config.device)
            )
        vae = self._vae()
        # Keep native posterior sampling, with a stable per-example draw. Restore
        # RNG so preparation order cannot influence training noise/timesteps.
        seed = int.from_bytes(
            hashlib.sha256((sample.id + ":" + str(self.config.seed)).encode()).digest()[:8],
            "little",
        ) % (2**63 - 1)
        devices = [torch.cuda.current_device()] if self.config.device == "cuda" else []
        with (
            torch.random.fork_rng(devices=devices),
            torch.no_grad(),
            torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False),
        ):
            torch.manual_seed(seed)
            latent = vae.encode(pixels).cpu()
        clean = patchify(latent[0], self.native.latent_patch_size)
        ids = torch.tensor(
            [*prompt, self.special["start_of_image"], self.special["end_of_image"]],
            dtype=torch.long,
        )
        indexes = torch.tensor([*range(len(prompt) + 1), length - 1])
        vae_indexes = torch.arange(len(prompt) + 1, length - 1)
        mask = torch.zeros(length, dtype=torch.bool)
        mask[vae_indexes] = True
        gh, gw = self.grid
        spatial = (
            torch.arange(gh)[:, None] * self.native.max_latent_size + torch.arange(gw)
        ).flatten()
        payload = dict(
            sequence_length=length,
            packed_text_ids=ids,
            packed_text_indexes=indexes,
            sample_lens=[length],
            packed_position_ids=torch.tensor([*range(len(prompt)), *([len(prompt)] * (n + 2))]),
            nested_attention_masks=[
                prepare_attention_mask_per_sample([len(prompt), n + 2], ["causal", "noise"])
            ],
            padded_latent=latent,
            patchified_vae_latent_shapes=[list(self.grid)],
            packed_latent_position_ids=spatial,
            packed_vae_token_indexes=vae_indexes,
            mse_loss_indexes=mask,
        )
        tensors, layout = pack_tree(payload)
        tensors["clean_patches"] = clean
        tensors["target_mask"] = torch.ones_like(clean, dtype=torch.bool)
        plan = SupervisionPlan(
            targets=(
                SupervisionTarget(
                    id="image_latent",
                    modality="image",
                    representation=ContinuousRepresentation(
                        space="bagel_flux_vae", encoder_id="native-frozen-vae"
                    ),
                    values=tensor_ref("clean_patches", clean, ("patch", "feature")),
                    mask=tensor_ref("target_mask", tensors["target_mask"], ("patch", "feature")),
                    measure=ReductionMeasure(unit="elements"),
                ),
            ),
            normalizers=(Normalizer(id="latent_elements", unit="elements"),),
            objectives=(
                LossObjective(
                    id="flow",
                    target_id="image_latent",
                    kind="flow_matching",
                    normalizer="latent_elements",
                ),
            ),
        )
        # The clean patch target is stored separately from native NCHW payload.
        cost["latent_elements"] *= 2
        cost["decoded_media_bytes"] = decoded_bytes
        return EncodedExample(
            tensors,
            SampleCost(**cost),
            layout=layout,
            supervision=plan,
            metadata={
                "image_transform": "bicubic_center_fit",
                "posterior": "native_sample_per_example_seed",
                "posterior_seed": seed,
            },
        ).validate()

    def collate(self, rows):
        return rows[0] if len(rows) == 1 else {"examples": rows}

    @staticmethod
    def to_device(batch, model, config):
        return move_tree(batch, device=config.device, dtype=torch.float32)

    def forward(self, model, batch):
        if "examples" in batch:
            return attach_losses(
                SimpleNamespace(),
                LossBundle.combine([self.forward(model, row).losses for row in batch["examples"]]),
            )
        kwargs = {k: v for k, v in batch.items() if k != "packing_lengths"}
        n = len(batch["packed_vae_token_indexes"])
        # One logit-normal noise level per target image, sampled at every visit.
        kwargs["packed_timesteps"] = torch.randn((), device=self.config.device).expand(n)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.config.device == "cuda" and self.config.dtype == "bfloat16"
            else nullcontext()
        )
        with context:
            result = model(**kwargs)
        mse = result["mse"].float()
        bundle = LossBundle(
            (LossTerm("flow", mse.sum(), "latent_elements", kind="flow_matching"),),
            {"latent_elements": float(mse.numel())},
            {"latent_elements": "elements"},
        )
        return attach_losses(SimpleNamespace(), bundle)

    @torch.no_grad()
    def generate_image(self, model, batch, *, num_steps):
        model.eval()
        device = next(model.parameters()).device

        def move(value):
            return move_tree(value, device=device, dtype=torch.float32)

        cache = NaiveCache(model.config.llm_config.num_hidden_layers)
        prompt, lengths, ropes = model.prepare_prompts(
            [0], [0], [batch["prompt"]], self.processor, self.special
        )
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        )
        with context:
            cache = model.forward_cache_update_text(cache, **move(prompt))
            inputs = model.prepare_vae_latent(
                lengths, ropes, [tuple(batch["image_shape"])], self.special
            )
            latent = model.generate_image(
                **move(inputs),
                past_key_values=cache,
                num_timesteps=num_steps,
                cfg_text_scale=1.0,
                cfg_img_scale=1.0,
            )[0]
        h, w = self.grid
        p, c = self.native.latent_patch_size, self.native.vae_config.z_channels
        latent = latent.reshape(h, w, p, p, c).permute(4, 0, 2, 1, 3).reshape(1, c, h * p, w * p)
        image = self._vae().decode(latent.to(device=self.config.device, dtype=torch.float32))
        if not torch.isfinite(image).all():
            raise OmniFactoryError("E_IMAGE_OUTPUT", "nonfinite native decoded image")
        pixels = image[0].clamp(-1, 1).add(1).mul(127.5).permute(1, 2, 0).byte().cpu().numpy()
        return Image.fromarray(pixels)

    def generate(self, model, batch, *, max_new_tokens):
        raise OmniFactoryError("E_IMAGE_OUTPUT", "BAGEL DiT requires image generation")
