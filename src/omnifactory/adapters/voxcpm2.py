"""Text/reference-audio → native continuous AudioVAE patches and stop supervision."""

from types import SimpleNamespace

import torch
from transformers import LlamaTokenizerFast

from omnifactory._vendor.voxcpm2.model.utils import mask_multichar_chinese_tokens
from omnifactory._vendor.voxcpm2.training.packers import AudioFeatureProcessingPacker
from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
from omnifactory.adapters.speech import load_target_audio
from omnifactory.adapters.supervision import tensor_ref
from omnifactory.adapters.voxcpm2_model import generation_cache, load_vae, model_root, native_config
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view
from omnifactory.data.supervision import (
    ContinuousRepresentation,
    DiscreteRepresentation,
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


class VoxCPM2Adapter:
    def __init__(self, config):
        self.config = config
        self.root = model_root(config)
        self.native = native_config(self.root)
        self.processor = LlamaTokenizerFast.from_pretrained(self.root, local_files_only=True)
        self.tokenize = mask_multichar_chinese_tokens(self.processor)
        self.resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        self.packer = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def save_processor(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(path)
        write_json(path / "voxcpm_processor.json", {"source": str(self.root.resolve())})

    def _packer(self):
        if self.packer is None:
            vae = load_vae(self.root, self.native, device=self.config.device)
            self.packer = AudioFeatureProcessingPacker(
                1, self.config.max_length, self.native.patch_size, self.native.feat_dim, vae
            )
        return self.packer

    def _wave(self, sample, media):
        wave, size = load_target_audio(
            self.resolver,
            sample.assets[media.asset_id],
            media,
            max_seconds=self.config.max_audio_seconds,
            max_bytes=self.config.budget.max_decoded_media_bytes,
            target_rate=self.native.audio_vae_config.sample_rate,
        )
        return torch.from_numpy(wave).to(self.config.device), size

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        # Native Snake activations use TorchScript. Its CUDA profiling executor
        # changes floating-point fusion after the first call, so cold/warm VAE
        # encodings otherwise differ and invalidate exact prepared-data hashes.
        # Keep the original FP32 operations for both targets and references;
        # restore the caller's execution setting when preparation finishes.
        with torch.jit.optimized_execution(False):
            return self._encode(sample, prediction_id, contract_id=contract_id, inference=inference)

    def _encode(self, sample, prediction_id, *, contract_id, inference=False):
        view = inference_view(sample, prediction_id, ("audio",))
        if (
            len(view.messages) != 1
            or view.messages[0].role != "user"
            or any(b.type != "text" for b in view.messages[0].content)
        ):
            raise OmniFactoryError(
                "E_TTS_INPUT", "VoxCPM2 expects one user text containing the exact utterance"
            )
        text = "".join(b.text for b in view.messages[0].content)
        if not text.strip():
            raise OmniFactoryError("E_TTS_INPUT", "empty utterance")
        if any(c.kind != "reference_audio" or c.transcript for c in view.conditions):
            raise OmniFactoryError(
                "E_TTS_CONDITION",
                "VoxCPM2 supports reference audio without continuation transcript",
            )
        tokens = torch.tensor(self.tokenize(text), dtype=torch.int64, device=self.config.device)
        size = 0
        ref = None
        if view.conditions:
            ref, size = self._wave(sample, view.conditions[0].media)
        if inference:
            # No access to target text/audio. Native reference-only prefix is an input.
            features = (
                self._packer().extract_audio_feats(ref)[0][0].cpu() if ref is not None else None
            )
            payload = {"text": text, "tokens": tokens.cpu(), "reference_features": features}
            length = len(tokens) + 1 + (len(features) + 2 if features is not None else 0)
            if length >= self.config.max_length:
                raise OmniFactoryError("E_SEQUENCE_LENGTH", "TTS prompt exceeds max_length")
            tensors, layout = pack_tree(payload)
            return EncodedExample(
                tensors,
                SampleCost(
                    sample_id=sample.id,
                    prediction_id=prediction_id,
                    contract_id=contract_id,
                    modalities=("text", "audio") if ref is not None else ("text",),
                    llm_tokens=length,
                    decoded_media_bytes=size,
                    latent_elements=features.numel() if features is not None else 0,
                    audio_frames=features.shape[0] * self.native.patch_size
                    if features is not None
                    else 0,
                ),
                layout=layout,
            )
        prediction = next(p for p in sample.predictions if p.id == prediction_id)
        message = next(m for m in sample.messages if m.id == prediction.target_message_id)
        targets = [b for b in message.content if b.id in prediction.target_block_ids]
        audio = [b for b in targets if b.type == "audio"]
        transcript = [b for b in targets if b.type == "text"]
        if (
            len(targets) != 2
            or len(audio) != 1
            or len(transcript) != 1
            or transcript[0].text != text
        ):
            raise OmniFactoryError(
                "E_TTS_TARGET",
                "select one target waveform and its transcript, identical to the input utterance",
            )
        if not any(
            p.audio_block_id == audio[0].id and p.text_block_id == transcript[0].id
            for p in sample.speech_pairs
        ):
            raise OmniFactoryError(
                "E_SPEECH_PAIR", "target audio requires an explicit utterance transcript pair"
            )
        wave, target_size = self._wave(sample, audio[0].media)
        size += target_size
        if size > self.config.budget.max_decoded_media_bytes:
            raise OmniFactoryError(
                "E_DECODE_BUDGET", "reference and target exceed combined audio budget"
            )
        packer = self._packer()
        packed = (
            packer.process_tts_data(wave, tokens)
            if ref is None
            else packer.process_tts_data_with_ref(ref, wave, tokens)
        )
        names = ("text_tokens", "audio_feats", "text_mask", "audio_mask", "loss_mask", "labels")
        payload = {k: v.detach().cpu().unsqueeze(0) for k, v in zip(names, packed[:6])}
        length = payload["text_tokens"].shape[1]
        if length > self.config.max_length:
            raise OmniFactoryError(
                "E_SEQUENCE_LENGTH",
                "VoxCPM2 text/audio exceeds max_length; split utterances explicitly",
            )
        payload["labels"] = payload["labels"].long()
        payload["loss_mask"] = payload["loss_mask"].bool()
        payload["position_ids"] = torch.arange(length).unsqueeze(0)
        tensors, layout = pack_tree(payload)
        tensors["latent_mask"] = (
            payload["loss_mask"].unsqueeze(-1).expand(-1, -1, self.native.patch_size).clone()
        )
        plan = SupervisionPlan(
            targets=(
                SupervisionTarget(
                    id="audio_latent",
                    modality="audio",
                    representation=ContinuousRepresentation(
                        space="voxcpm2_audiovae_v2", encoder_id="native-frozen-vae"
                    ),
                    values=tensor_ref(
                        "audio_feats",
                        tensors["audio_feats"],
                        ("batch", "patch", "frame", "feature"),
                    ),
                    mask=tensor_ref(
                        "latent_mask", tensors["latent_mask"], ("batch", "patch", "frame")
                    ),
                    measure=ReductionMeasure(unit="frames", feature_reduction="mean"),
                ),
                SupervisionTarget(
                    id="stop",
                    modality="audio",
                    representation=DiscreteRepresentation(vocabulary_sizes=(2,)),
                    values=tensor_ref("labels", tensors["labels"], ("batch", "patch")),
                    mask=tensor_ref("loss_mask", tensors["loss_mask"], ("batch", "patch")),
                    measure=ReductionMeasure(unit="symbols"),
                ),
            ),
            normalizers=(
                Normalizer(id="latent_frames", unit="frames"),
                Normalizer(id="stop_symbols", unit="symbols"),
            ),
            objectives=(
                LossObjective(
                    id="flow",
                    target_id="audio_latent",
                    kind="flow_matching",
                    normalizer="latent_frames",
                ),
                LossObjective(
                    id="stop", target_id="stop", kind="cross_entropy", normalizer="stop_symbols"
                ),
            ),
        )
        return EncodedExample(
            tensors,
            SampleCost(
                sample_id=sample.id,
                prediction_id=prediction_id,
                contract_id=contract_id,
                modalities=("text", "audio"),
                llm_tokens=length,
                audio_frames=int(payload["audio_mask"].sum()) * self.native.patch_size,
                latent_elements=payload["audio_feats"].numel(),
                decoded_media_bytes=size,
            ),
            answer=None,
            layout=layout,
            supervision=plan,
            metadata={
                "target_representation": "continuous",
                "native_patch_size": self.native.patch_size,
                "last_patch": "native waveform right-padding",
                "encoder_execution": "torchscript_unoptimized_fp32",
            },
        ).validate()

    def collate(self, rows):
        # Native MiniCPM4 has no per-example attention mask in its forward contract.
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
        # Match upstream mixed-precision training: native MSE/CE reductions stay
        # FP32 under autocast, even when the native transformer weights are BF16.
        with torch.autocast(
            device_type=self.config.device,
            dtype=torch.bfloat16,
            enabled=self.config.device == "cuda" and self.config.dtype == "bfloat16",
        ):
            result = model(**batch)
        patches = float(batch["loss_mask"].sum())
        frames = patches * model.patch_size
        bundle = LossBundle(
            (
                LossTerm(
                    "flow", result["loss/diff"] * frames, "latent_frames", kind="flow_matching"
                ),
                LossTerm(
                    "stop", result["loss/stop"] * patches, "stop_symbols", kind="cross_entropy"
                ),
            ),
            {"latent_frames": frames, "stop_symbols": patches},
            {"latent_frames": "frames", "stop_symbols": "symbols"},
        )
        return attach_losses(SimpleNamespace(native=result), bundle)

    def generate(self, model, batch, *, max_new_tokens):
        raise OmniFactoryError("E_TTS_OUTPUT", "VoxCPM2 generates audio; use generate_audio")

    @torch.no_grad()
    def generate_speech(self, model, batch, *, max_new_tokens, max_codec_tokens):
        tokens = torch.cat((batch["tokens"], batch["tokens"].new_tensor([101])))
        features = torch.zeros(
            (len(tokens), model.patch_size, model.feat_dim),
            device=model.device,
            dtype=model._dtype(),
        )
        text_mask = torch.ones_like(tokens)
        audio_mask = torch.zeros_like(tokens)
        ref = batch["reference_features"]
        if ref is not None:
            rt, rf, tm, am = model._make_ref_prefix(ref, tokens.device)
            tokens, features = torch.cat((rt, tokens)), torch.cat((rf, features))
            text_mask, audio_mask = torch.cat((tm, text_mask)), torch.cat((am, audio_mask))
        generation_cache(model, len(tokens) + max_codec_tokens)
        features = features.to(model._dtype())
        stops = []
        hook = model.stop_head.register_forward_hook(
            lambda m, i, o: stops.append(int(o.argmax(-1)[0]))
        )
        try:
            _, latents = model.inference(
                tokens[None],
                text_mask[None],
                features[None],
                audio_mask[None],
                max_len=max_codec_tokens,
                inference_timesteps=10,
            )
        finally:
            hook.remove()
        # Native [patch, frame, feature], decoded separately without discretization.
        return {
            "text": batch["text"],
            "audio_latents": latents.float().reshape(-1, model.feat_dim).tolist(),
            "latent_space": "voxcpm2_audiovae_v2",
            "speech_finished": len(stops) > 3 and stops[-1] == 1,
        }

    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None):
        if speech_assets is not None or reference_audio is not None:
            raise OmniFactoryError(
                "E_TTS_CONDITION",
                "VoxCPM2 uses its native AudioVAE and per-sample reference_audio condition",
            )

        def decode(values):
            latent = torch.tensor(values, dtype=torch.float32, device=model.device)
            if (
                latent.ndim != 2
                or latent.shape[-1] != model.feat_dim
                or not torch.isfinite(latent).all()
            ):
                raise OmniFactoryError("E_LATENT", "invalid AudioVAE latent shape or values")
            wave = model.audio_vae.decode(latent.T[None]).flatten().detach().cpu().numpy()
            import numpy as np

            if not wave.size or not np.isfinite(wave).all():
                raise OmniFactoryError(
                    "E_AUDIO", "native AudioVAE produced empty/nonfinite waveform"
                )
            return wave, model.sample_rate

        return decode
