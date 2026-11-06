"""Train the actual MiniCPM Token2Wav DiT using conditional flow matching.

The public upstream wrapper exposes inference only. This module supplies an
explicit offline CFM objective; it does not claim to reproduce private chunk training.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
from omnifactory.adapters.speech import S3Tokenizer, load_target_audio
from omnifactory.adapters.supervision import normalization_counts, tensor_ref
from omnifactory.adapters.token2wav import ASSET_FILES, load_native_flow, verify_assets
from omnifactory.data.media import LocalMediaResolver
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


def flow_plan(tensors):
    return SupervisionPlan(
        targets=(
            SupervisionTarget(
                id="mel",
                modality="audio",
                representation=ContinuousRepresentation(
                    space="minicpm_log_mel_24khz_80", encoder_id="stepaudio2-minicpmo-0.1.1"
                ),
                values=tensor_ref("mel", tensors["mel"], ("batch", "frame", "feature")),
                mask=tensor_ref("loss_mask", tensors["loss_mask"], ("batch", "frame")),
                measure=ReductionMeasure(unit="frames", feature_reduction="mean"),
            ),
        ),
        normalizers=(Normalizer(id="mel_frames", unit="frames"),),
        objectives=(
            LossObjective(
                id="velocity", target_id="mel", kind="flow_matching", normalizer="mel_frames"
            ),
        ),
    )


def flow_path(clean, noise, time, sigma_min):
    """Gaussian-to-data conditional path and its exact time derivative."""
    t = time[:, None, None]
    return (1 - (1 - sigma_min) * t) * noise + t * clean, clean - (1 - sigma_min) * noise


class MiniCPMFlowModel(torch.nn.Module):
    def __init__(self, flow, config):
        super().__init__()
        self.flow = flow
        self.recipe = config
        self.checkpointing = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing = True

    def forward(self, batch, *, noise=None, time=None):
        flow = self.flow
        mel = batch["mel"].transpose(1, 2)
        codes = batch["codes"]
        # Frozen encoders stay deterministic during decoder-only adaptation.
        if not any(p.requires_grad for p in flow.encoder.parameters()):
            flow.encoder.eval()
        encoded, _ = flow.encoder(flow.input_embedding(codes), batch["code_lengths"])
        mu = flow.encoder_proj(encoded).transpose(1, 2)
        if mu.shape != mel.shape:
            raise OmniFactoryError("E_FLOW_LENGTH", "native upsampling and mel alignment differ")
        speaker = flow.spk_embed_affine_layer(F.normalize(batch["speaker"], dim=1))
        valid = batch["valid"].unsqueeze(1).to(mel.dtype)
        cond = batch["condition"].transpose(1, 2)
        generator = None
        if not self.training:
            generator = torch.Generator(device=mel.device).manual_seed(self.recipe.seed)
        if noise is None:
            noise = torch.randn(mel.shape, device=mel.device, dtype=mel.dtype, generator=generator)
        if time is None:
            time = torch.rand(mel.shape[0], device=mel.device, dtype=mel.dtype, generator=generator)
            time = 1 - torch.cos(time * torch.pi / 2)
        if (
            noise.shape != mel.shape
            or time.shape != (mel.shape[0],)
            or not torch.isfinite(noise).all()
            or not torch.isfinite(time).all()
            or ((time < 0) | (time > 1)).any()
        ):
            raise OmniFactoryError("E_FLOW_NOISE", "invalid explicit noise or time")
        noisy, velocity = flow_path(mel, noise, time, self.recipe.flow_sigma_min)
        if self.training and self.recipe.flow_cfg_dropout:
            keep = (torch.rand(mel.shape[0], device=mel.device) >= self.recipe.flow_cfg_dropout).to(
                mel.dtype
            )
            mu, speaker, cond = (
                mu * keep[:, None, None],
                speaker * keep[:, None],
                cond * keep[:, None, None],
            )
        args = (noisy * valid, valid, mu * valid, time, speaker, cond * valid)
        estimate = (
            checkpoint(flow.decoder.estimator, *args, use_reentrant=False)
            if self.training and self.checkpointing
            else flow.decoder.estimator(*args)
        )
        mask = batch["loss_mask"]
        per_frame = (
            (estimate.transpose(1, 2)[mask].float() - velocity.transpose(1, 2)[mask].float())
            .square()
            .mean(-1)
        )
        bundle = LossBundle(
            (LossTerm("velocity", per_frame.sum(), "mel_frames", kind="flow_matching"),),
            {"mel_frames": float(mask.sum())},
            {"mel_frames": "frames"},
        )
        return attach_losses(
            SimpleNamespace(velocity=estimate, target_velocity=velocity, time=time), bundle
        )


def load_flow_model(config):
    verify_assets(config.model)
    flow = load_native_flow(config.model)
    # Streaming CUDA-graph caches are not used by offline differentiable forward.
    for module in flow.modules():
        for name in ("att_cache_buffer", "cnn_cache_buffer"):
            if name in module._buffers:
                module._buffers[name] = torch.empty(0)
    return MiniCPMFlowModel(flow, config).to(
        device=config.device, dtype=getattr(torch, config.dtype)
    )


class MiniCPMFlowAdapter:
    def __init__(self, config):
        import onnxruntime as ort

        self.config = config
        self.resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        self.assets = Path(config.speech_assets or config.model)
        if (self.assets / "flow_processor.json").exists():
            import json

            self.assets = Path(
                json.loads((self.assets / "flow_processor.json").read_text())["assets"]
            )
        info = verify_assets(self.assets)
        self.codec = S3Tokenizer(self.assets / ASSET_FILES[0], info["sha256"][ASSET_FILES[0]])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.speaker = ort.InferenceSession(
            str(self.assets / "campplus.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.processor = self

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def save_processor(self, path):
        path.mkdir(parents=True, exist_ok=True)
        write_json(
            path / "flow_processor.json",
            {"assets": str(self.assets.resolve()), "features": "stepaudio2-minicpmo-0.1.1"},
        )

    def _features(self, sample, media):
        import torchaudio
        from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

        wave, decoded = load_target_audio(
            self.resolver,
            sample.assets[media.asset_id],
            media,
            max_seconds=self.config.max_audio_seconds,
            max_bytes=self.config.budget.max_decoded_media_bytes,
            target_rate=24000,
        )
        wav = torch.from_numpy(wave).unsqueeze(0)
        wav16 = torchaudio.functional.resample(wav, 24000, 16000)
        codes = self.codec.encode(wav16[0].numpy())
        mel = mel_spectrogram(wav).transpose(1, 2)
        # Native codec is 25 Hz and flow upsamples to 50 Hz. Retain the measured
        # feature count; codec rounding padding never contributes to the loss.
        frames = min(mel.shape[1], len(codes) * 2)
        if abs(mel.shape[1] - len(codes) * 2) > 2:
            raise OmniFactoryError("E_FLOW_ALIGNMENT", "unexpected codec/mel length mismatch")
        mel = F.pad(mel[:, :frames], (0, 0, 0, len(codes) * 2 - frames))
        return codes, mel, frames, wav16, decoded

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        if inference:
            raise OmniFactoryError(
                "E_FLOW_STAGE", "flow decoding needs explicit codec tokens; use flow-decode"
            )
        import torchaudio.compliance.kaldi as kaldi

        prediction = next(p for p in sample.predictions if p.id == prediction_id)
        target_message = next(m for m in sample.messages if m.id == prediction.target_message_id)
        targets = [
            b
            for b in target_message.content
            if b.id in prediction.target_block_ids and b.type == "audio"
        ]
        references = [c for c in prediction.conditions if c.kind == "reference_audio"]
        if len(targets) != 1 or len(references) != 1:
            raise OmniFactoryError(
                "E_FLOW_DATA",
                "flow_sft needs one target audio and a separate reference_audio condition",
            )
        rc, rm, rn, rw, rb = self._features(sample, references[0].media)
        tc, tm, tn, _, tb = self._features(sample, targets[0].media)
        if (
            rb + tb > self.config.budget.max_decoded_media_bytes
            or len(rc) + len(tc) > self.config.max_codec_tokens
        ):
            raise OmniFactoryError(
                "E_FLOW_BUDGET", "reference plus target exceeds decode/codec budget"
            )
        features = kaldi.fbank(rw, num_mel_bins=80, dither=0, sample_frequency=16000)
        features -= features.mean(0, keepdim=True)
        speaker = torch.from_numpy(
            self.speaker.run(
                None, {self.speaker.get_inputs()[0].name: features.unsqueeze(0).numpy()}
            )[0]
        )
        mel = torch.cat([rm, tm], 1)
        prefix, length = rm.shape[1], mel.shape[1]
        condition = torch.zeros_like(mel)
        condition[:, :rn] = rm[:, :rn]
        valid = torch.zeros(1, length, dtype=torch.bool)
        valid[:, :rn] = True
        valid[:, prefix : prefix + tn] = True
        loss_mask = torch.zeros_like(valid)
        loss_mask[:, prefix : prefix + tn] = True
        tensors = {
            "mel": mel,
            "condition": condition,
            "valid": valid,
            "loss_mask": loss_mask,
            "codes": torch.cat([rc, tc]).unsqueeze(0),
            "code_lengths": torch.tensor([len(rc) + len(tc)]),
            "speaker": speaker,
        }
        flat, layout = pack_tree(tensors)
        encoded = EncodedExample(
            flat,
            SampleCost(
                sample_id=sample.id,
                prediction_id=prediction_id,
                contract_id=contract_id,
                modalities=("audio",),
                latent_elements=mel.numel() + condition.numel(),
                codec_frames=len(rc) + len(tc),
                audio_frames=length,
                decoded_media_bytes=rb + tb,
            ),
            layout=layout,
            supervision=flow_plan(tensors),
            metadata={
                "objective": "offline_conditional_flow_matching",
                "reference_frames": rn,
                "target_frames": tn,
            },
        )
        assert normalization_counts(encoded.supervision, tensors)["mel_frames"] == tn
        return encoded.validate()

    def collate(self, rows):
        # Native conformer padding changes boundary convolutions. Keep singleton
        # conditioning exact; the common loss bundle combines measured sums.
        return {"examples": rows}

    @staticmethod
    def to_device(batch, model, config):
        return move_tree(batch, device=config.device, dtype=getattr(torch, config.dtype))

    def forward(self, model, batch):
        if "examples" in batch:
            outputs = [model(row) for row in batch["examples"]]
            return attach_losses(SimpleNamespace(), LossBundle.combine([o.losses for o in outputs]))
        return model(batch)
