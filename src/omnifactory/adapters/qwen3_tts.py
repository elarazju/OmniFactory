"""Native Qwen3-TTS 12 Hz SFT with explicit frame/codebook causal alignment.

Training packs the same non-streaming prompt as native generation. Losses use
one time shift for codebook zero and no shift within already-aligned residual
codebook logits. This deliberately avoids the extra HF loss shift in the
upstream example trainer. Native parameter names and inference are preserved.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from omnifactory._vendor.qwen3_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
)
from omnifactory._vendor.qwen3_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSSpeakerEncoder,
    mel_spectrogram,
)
from omnifactory._vendor.qwen3_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
from omnifactory.adapters.speech import load_target_audio
from omnifactory.adapters.supervision import tensor_ref
from omnifactory.artifacts import sha256
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import MediaAsset, MediaSlice, inference_view
from omnifactory.data.supervision import (
    DiscreteRepresentation,
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


def model_root(config):
    root = Path(config.model)
    if root.is_dir():
        pointer = root / "qwen3_tts_processor.json"
        return Path(json.loads(pointer.read_text())["source"]) if pointer.exists() else root
    from huggingface_hub import snapshot_download

    from omnifactory.training.prepare import source_fingerprint

    source_fingerprint(config.model, config.revision)
    return Path(snapshot_download(config.model, revision=config.revision))


def native_config(root):
    if json.loads((Path(root) / "config.json").read_text()).get("model_type") != "qwen3_tts":
        raise OmniFactoryError("E_NATIVE_CONFIG", "expected Qwen3-TTS architecture")
    cfg = Qwen3TTSConfig.from_pretrained(root, local_files_only=True)
    if (
        cfg.model_type != "qwen3_tts"
        or cfg.tokenizer_type != "qwen3_tts_tokenizer_12hz"
        or cfg.tts_model_type not in ("base", "custom_voice")
        or cfg.talker_config.num_code_groups != 16
    ):
        raise OmniFactoryError(
            "E_NATIVE_CONFIG", "expected native Qwen3-TTS 12 Hz Base/CustomVoice"
        )
    return cfg


def save_native_model(model, root):
    """Save an upstream-readable speaker config without HF-only metadata.

    The native speaker constructor does not accept serialization metadata and
    its missing base initialization breaks HF's recursive diff after loading.
    Serialize a clean native configuration, then restore the runtime object.
    """
    import inspect

    from omnifactory._vendor.qwen3_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSSpeakerEncoderConfig,
    )

    speaker = model.config.speaker_encoder_config
    names = inspect.signature(Qwen3TTSSpeakerEncoderConfig).parameters
    model.config.speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(
        **{key: getattr(speaker, key) for key in names}
    )
    try:
        model.save_pretrained(root, safe_serialization=True)
        config = json.loads((Path(root) / "config.json").read_text())
        config["speaker_encoder_config"].pop("model_type", None)
        write_json(Path(root) / "config.json", config)
    finally:
        model.config.speaker_encoder_config = speaker


def load_qwen3_tts(config):
    root = model_root(config)
    cfg = native_config(root)
    # Bypass the upstream convenience loader, which eagerly loads the frozen
    # waveform codec. Preparation and decoding load that asset only when needed.
    model, info = super(
        Qwen3TTSForConditionalGeneration, Qwen3TTSForConditionalGeneration
    ).from_pretrained(
        root,
        config=cfg,
        local_files_only=True,
        torch_dtype=getattr(torch, config.dtype),
        attn_implementation=config.attention,
        output_loading_info=True,
    )
    if any(
        info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise OmniFactoryError("E_NATIVE_WEIGHTS", str(info))
    model.to(config.device)
    model.config.use_cache = False
    model.talker.config.use_cache = False
    model.talker.code_predictor.config.use_cache = False
    return model


class Qwen3TTSAdapter:
    def __init__(self, config):
        self.config = config
        self.root = model_root(config)
        self.native = native_config(self.root)
        self.processor = AutoTokenizer.from_pretrained(
            self.root, local_files_only=True, fix_mistral_regex=True
        )
        self.resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        self.codec = None
        self._speaker = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def save_processor(self, path):
        self.processor.save_pretrained(path)
        write_json(path / "qwen3_tts_processor.json", {"source": str(self.root.resolve())})

    def _codec(self):
        if self.codec is None:
            self.codec = Qwen3TTSTokenizer.from_pretrained(
                str(self.root / "speech_tokenizer"),
                device_map=self.config.device,
                torch_dtype=torch.float32,
                attn_implementation=self.config.attention,
            )
            self.codec.model.eval().requires_grad_(False)
        return self.codec

    def speaker_embedding(self):
        if self._speaker is not None:
            return self._speaker
        path = self.config.speech_reference
        if path is None or sha256(path) != self.config.speech_reference_sha256:
            raise OmniFactoryError(
                "E_SPEAKER_REFERENCE", "fixed reference recording is missing or changed"
            )
        wave, _ = load_target_audio(
            LocalMediaResolver(path.parent),
            MediaAsset(uri=path.name, modality="audio"),
            MediaSlice(asset_id="reference"),
            max_seconds=self.config.max_audio_seconds,
            max_bytes=self.config.budget.max_decoded_media_bytes,
            target_rate=24000,
        )
        if len(wave) < 1024:
            raise OmniFactoryError(
                "E_SPEAKER_REFERENCE", "reference must have at least 1024 samples"
            )
        encoder = Qwen3TTSSpeakerEncoder(self.native.speaker_encoder_config).eval()
        index = self.root / "model.safetensors.index.json"
        names = (
            set(json.loads(index.read_text())["weight_map"].values())
            if index.exists()
            else {"model.safetensors"}
        )
        weights = {}
        for name in names:
            if Path(name).name != name:
                raise OmniFactoryError("E_NATIVE_WEIGHTS", "invalid shard path")
            with safe_open(self.root / name, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith("speaker_encoder."):
                        weights[key.removeprefix("speaker_encoder.")] = handle.get_tensor(key)
        encoder.load_state_dict(weights, strict=True)
        with torch.no_grad():
            mel = mel_spectrogram(
                torch.from_numpy(wave)[None],
                n_fft=1024,
                num_mels=128,
                sampling_rate=24000,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000,
            ).transpose(1, 2)
            self._speaker = encoder(mel).float().detach()
        return self._speaker

    def _text(self, text):
        if not text.strip() or any(t in text for t in ("<|im_start|>", "<|im_end|>", "<|tts_")):
            raise OmniFactoryError(
                "E_TTS_INPUT", "literal nonempty text without control tokens required"
            )
        raw = self.processor(
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        # Native prompt construction depends on this exact role/suffix layout.
        prefix = self.processor("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        suffix = self.processor(
            "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        ).input_ids
        if (
            len(prefix) != 3
            or len(suffix) != 5
            or raw[0, :3].tolist() != prefix
            or raw[0, -5:].tolist() != suffix
        ):
            raise OmniFactoryError(
                "E_TTS_TEMPLATE", "tokenizer does not match native role boundaries"
            )
        return raw

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        view = inference_view(sample, prediction_id, ("audio",))
        if (
            len(view.messages) != 1
            or view.messages[0].role != "user"
            or any(b.type != "text" for b in view.messages[0].content)
        ):
            raise OmniFactoryError(
                "E_TTS_INPUT", "one user message containing the literal utterance required"
            )
        text = "".join(b.text for b in view.messages[0].content)
        ids = self._text(text)
        if view.conditions:
            if len(view.conditions) != 1:
                raise OmniFactoryError("E_TTS_CONDITION", "single fixed speaker reference required")
            condition = view.conditions[0]
            if (
                condition.kind != "reference_audio"
                or condition.transcript
                or condition.media.start_us
                or condition.media.end_us is not None
                or self.resolver.resolve(view.assets[condition.media.asset_id]).sha256
                != self.config.speech_reference_sha256
            ):
                raise OmniFactoryError(
                    "E_TTS_CONDITION", "reference must match the full fixed recording"
                )
        if inference:
            payload = {"text": text, "text_ids": ids}
            length = ids.shape[1] + 2
            if length >= self.config.max_length:
                raise OmniFactoryError("E_SEQUENCE_LENGTH", "TTS prompt exceeds max_length")
            tensors, layout = pack_tree(payload)
            return EncodedExample(
                tensors,
                SampleCost(
                    sample_id=sample.id,
                    prediction_id=prediction_id,
                    contract_id=contract_id,
                    modalities=("text",),
                    llm_tokens=length,
                ),
                layout=layout,
            )
        if self.native.tts_model_type != "base":
            raise OmniFactoryError(
                "E_TTS_TRAINING_BASE", "single-speaker training starts from Base"
            )
        prediction = next(p for p in sample.predictions if p.id == prediction_id)
        message = next(m for m in sample.messages if m.id == prediction.target_message_id)
        targets = [b for b in message.content if b.id in prediction.target_block_ids]
        audios = [b for b in targets if b.type == "audio"]
        texts = [b for b in targets if b.type == "text"]
        if len(targets) != 2 or len(audios) != 1 or len(texts) != 1 or texts[0].text != text:
            raise OmniFactoryError(
                "E_TTS_TARGET", "select target audio and its exact utterance transcript"
            )
        audio = audios[0]
        if not any(
            p.audio_block_id == audio.id and p.text_block_id == texts[0].id
            for p in sample.speech_pairs
        ):
            raise OmniFactoryError("E_SPEECH_PAIR", "explicit transcript/audio pairing required")
        if (
            self.resolver.resolve(sample.assets[audio.media.asset_id]).sha256
            == self.config.speech_reference_sha256
        ):
            raise OmniFactoryError(
                "E_SPEAKER_REFERENCE", "target and fixed reference must be separate recordings"
            )
        speaker = self.speaker_embedding()
        wave, size = load_target_audio(
            self.resolver,
            sample.assets[audio.media.asset_id],
            audio.media,
            max_seconds=self.config.max_audio_seconds,
            max_bytes=self.config.budget.max_decoded_media_bytes,
            target_rate=24000,
        )
        with torch.no_grad():
            codes = self._codec().encode(wave, sr=24000).audio_codes[0].long().cpu()
        n = len(codes)
        vocab = self.native.talker_config.code_predictor_config.vocab_size
        if codes.shape != (n, 16) or not n or (codes < 0).any() or (codes >= vocab).any():
            raise OmniFactoryError("E_CODEC_SHAPE", "expected nonempty native 16-codebook labels")
        if n > self.config.max_codec_tokens:
            raise OmniFactoryError("E_CODEC_LENGTH", "split utterances to fit max_codec_tokens")
        payload = pack_training(ids[:, :-5], codes, self.native)
        payload["speaker_embedding"] = speaker
        length = payload["input_ids"].shape[1]
        if length > self.config.max_length:
            raise OmniFactoryError("E_SEQUENCE_LENGTH", "text/audio exceeds max_length")
        tensors, layout = pack_tree(payload)
        targets_spec = []
        for name, axes, vs in [
            ("primary", ("batch", "time"), (self.native.talker_config.vocab_size,)),
            ("residual", ("frame", "codebook"), (vocab,) * 15),
        ]:
            key = name + "_labels"
            mask = name + "_mask"
            tensors[mask] = tensors[key] != -100
            targets_spec.append(
                SupervisionTarget(
                    id=name,
                    modality="audio",
                    representation=DiscreteRepresentation(
                        vocabulary_sizes=vs,
                        codebook_axis="codebook" if name == "residual" else None,
                    ),
                    values=tensor_ref(key, tensors[key], axes),
                    mask=tensor_ref(mask, tensors[mask], axes),
                    measure=ReductionMeasure(unit="symbols"),
                )
            )
        plan = SupervisionPlan(
            targets=tuple(targets_spec),
            normalizers=tuple(
                Normalizer(id=k, unit="symbols") for k in ("primary_symbols", "residual_symbols")
            ),
            objectives=tuple(
                LossObjective(
                    id=k + "_ce",
                    target_id=k,
                    kind="cross_entropy",
                    normalizer=k + "_symbols",
                    weight=self.config.speech_loss_weight * (0.3 if k == "residual" else 1),
                )
                for k in ("primary", "residual")
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
                codec_frames=n,
                decoded_media_bytes=size,
            ),
            layout=layout,
            supervision=plan,
            metadata={
                "codec": "qwen3_tts_tokenizer_12hz",
                "label_alignment": "one_time_shift; within_frame_residual",
                "reference_sha256": self.config.speech_reference_sha256,
            },
        ).validate()

    def collate(self, rows):
        return rows[0] if len(rows) == 1 else {"examples": rows}

    @staticmethod
    def to_device(batch, model, config):
        return move_tree(batch, device=config.device, dtype=getattr(torch, config.dtype))

    def forward(self, model, batch):
        if "examples" in batch:
            return attach_losses(
                SimpleNamespace(),
                LossBundle.combine([self.forward(model, b).losses for b in batch["examples"]]),
            )
        talker = model.talker
        embeds = talker.text_projection(talker.get_text_embeddings()(batch["input_ids"][:, :, 0]))
        embeds = embeds * batch["text_mask"][:, :, None]
        codec = (
            talker.get_input_embeddings()(batch["input_ids"][:, :, 1])
            * batch["codec_mask"][:, :, None]
        )
        codec = codec.clone()
        codec[:, 6] = batch["speaker_embedding"]
        embeds = embeds + codec
        for i in range(1, 16):
            embeds = (
                embeds
                + talker.code_predictor.get_input_embeddings()[i - 1](batch["codec_ids"][:, :, i])
                * batch["frame_mask"][:, :, None]
            )
        result = talker(
            inputs_embeds=embeds[:, :-1],
            attention_mask=batch["attention_mask"][:, :-1],
            output_hidden_states=True,
            use_cache=False,
        )
        primary = result.logits.float()
        labels = batch["primary_labels"][:, 1:]
        primary_sum = F.cross_entropy(
            primary.reshape(-1, primary.shape[-1]), labels.reshape(-1), reduction="sum"
        )
        # Residual frame t is conditioned on the query BEFORE frame t, just as generation.
        hidden = result.hidden_states[0][-1][batch["frame_mask"][:, 1:]]
        codes = batch["codec_ids"][batch["frame_mask"]]
        seq = [hidden[:, None], talker.get_input_embeddings()(codes[:, :1])]
        seq.extend(
            talker.code_predictor.get_input_embeddings()[i - 1](codes[:, i : i + 1])
            for i in range(1, 15)
        )
        residual = talker.code_predictor.forward_finetune(
            inputs_embeds=torch.cat(seq, dim=1), use_cache=False
        ).logits.float()
        residual_sum = F.cross_entropy(
            residual.reshape(-1, residual.shape[-1]),
            batch["residual_labels"].reshape(-1),
            reduction="sum",
        )
        counts = {
            "primary_symbols": float((labels != -100).sum()),
            "residual_symbols": float(batch["residual_labels"].numel()),
        }
        bundle = LossBundle(
            (
                LossTerm(
                    "primary_ce",
                    primary_sum,
                    "primary_symbols",
                    weight=self.config.speech_loss_weight,
                    kind="cross_entropy",
                ),
                LossTerm(
                    "residual_ce",
                    residual_sum,
                    "residual_symbols",
                    weight=0.3 * self.config.speech_loss_weight,
                    kind="cross_entropy",
                ),
            ),
            counts,
            {k: "symbols" for k in counts},
        )
        return attach_losses(SimpleNamespace(logits=primary, residual_logits=residual), bundle)

    def generate(self, model, batch, *, max_new_tokens):
        raise OmniFactoryError("E_TTS_OUTPUT", "Qwen3-TTS generates audio; use generate_audio")

    @torch.no_grad()
    def generate_speech(self, model, batch, *, max_new_tokens, max_codec_tokens):
        if self.native.tts_model_type != "custom_voice":
            raise OmniFactoryError(
                "E_TTS_EXPORT", "generate speech from the exported CustomVoice model"
            )
        if (
            max_codec_tokens < 2
            or batch["text_ids"].shape[1] + 2 + max_codec_tokens > self.config.max_length
        ):
            raise OmniFactoryError(
                "E_SEQUENCE_LENGTH",
                "audio generation bound must fit the sequence budget and be >=2",
            )
        speaker = self.config.speech_speaker.lower()
        if speaker not in model.config.talker_config.spk_id:
            raise OmniFactoryError("E_SPEAKER", speaker)
        original = model.talker.generate
        captured = {}

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured["sequences"] = result.sequences
            return result

        model.talker.generate = capture
        try:
            codes, _ = model.generate(
                input_ids=[batch["text_ids"]],
                languages=["Auto"],
                speakers=[speaker],
                non_streaming_mode=True,
                max_new_tokens=max_codec_tokens,
                # Preserve native Talker and residual-codebook sampling defaults.
                # Forcing greedy decoding can repeat frames without reaching EOS.
            )
        finally:
            model.talker.generate = original
        finished = bool(
            (captured["sequences"][0] == model.config.talker_config.codec_eos_token_id).any()
        )
        return {
            "text": batch["text"],
            "codec_tokens": codes[0].cpu().tolist(),
            "speech_finished": finished,
        }

    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None):
        if speech_assets is not None or reference_audio is not None:
            raise OmniFactoryError(
                "E_TTS_CONDITION", "CustomVoice exports use their embedded speaker"
            )

        def decode(codes):
            wave, rate = self._codec().decode(
                {"audio_codes": torch.tensor(codes, dtype=torch.long, device=self.config.device)}
            )
            import numpy as np

            if not len(wave[0]) or not np.isfinite(wave[0]).all():
                raise OmniFactoryError(
                    "E_AUDIO_DECODE", "native codec returned empty/nonfinite audio"
                )
            return wave[0], rate

        return decode


def pack_training(ids, codes, cfg):
    """Unpadded native non-streaming prompt followed by all frames and a single EOS."""
    text_length, n = ids.shape[1], codes.shape[0]
    length = text_length + n + 8
    start = text_length + 7
    t = cfg.talker_config
    pairs = torch.zeros((1, length, 2), dtype=torch.long)
    pairs[0, :3, 0] = ids[0, :3]
    pairs[0, 3:7, 0] = cfg.tts_pad_token_id
    pairs[0, 7, 0] = cfg.tts_bos_token_id
    pairs[0, 8 : text_length + 5, 0] = ids[0, 3:]
    pairs[0, text_length + 5, 0] = cfg.tts_eos_token_id
    pairs[0, text_length + 6 :, 0] = cfg.tts_pad_token_id
    pairs[0, 3:8, 1] = torch.tensor(
        [t.codec_nothink_id, t.codec_think_bos_id, t.codec_think_eos_id, 0, t.codec_pad_id]
    )
    pairs[0, 8 : text_length + 6, 1] = t.codec_pad_id
    pairs[0, text_length + 6, 1] = t.codec_bos_id
    pairs[0, start : start + n, 1] = codes[:, 0]
    pairs[0, -1, 1] = t.codec_eos_token_id
    codec_ids = torch.zeros((1, length, 16), dtype=torch.long)
    codec_ids[0, start : start + n] = codes
    frame_mask = torch.zeros((1, length), dtype=torch.bool)
    frame_mask[:, start : start + n] = True
    primary = torch.full((1, length), -100, dtype=torch.long)
    primary[:, start:] = pairs[:, start:, 1]
    codec_mask = torch.ones((1, length), dtype=torch.bool)
    codec_mask[:, :3] = False
    codec_mask[:, 6] = False
    return dict(
        input_ids=pairs,
        codec_ids=codec_ids,
        text_mask=torch.ones_like(codec_mask),
        codec_mask=codec_mask,
        frame_mask=frame_mask,
        attention_mask=torch.ones_like(primary),
        primary_labels=primary,
        residual_labels=codes[:, 1:].clone(),
    )


def export_qwen3_tts(checkpoint, output, *, format=None):
    from omnifactory.training.engine import load_checkpoint
    from omnifactory.training.policy import merge_lora

    if format not in (None, "merged", "vllm-omni", "sglang-omni"):
        raise OmniFactoryError(
            "E_EXPORT_FORMAT", "Qwen3-TTS requires a native merged speech export"
        )
    model, adapter, config, metadata = load_checkpoint(checkpoint, for_export=True)
    speaker = adapter.speaker_embedding()[0]
    model = merge_lora(model)
    if model.config.tts_model_type != "base":
        raise OmniFactoryError("E_TTS_TRAINING_BASE", "export expects the trained Base model")
    speaker_id = 3000
    if model.config.talker_config.vocab_size <= speaker_id:
        raise OmniFactoryError("E_NATIVE_CONFIG", "native speaker slot 3000 is missing")
    model.config.tts_model_type = "custom_voice"
    model.config.talker_config.spk_id = {config.speech_speaker.lower(): speaker_id}
    model.config.talker_config.spk_is_dialect = {config.speech_speaker.lower(): False}
    del model.speaker_encoder
    with torch.no_grad():
        model.talker.get_input_embeddings().weight[speaker_id].copy_(speaker)
    with new_directory(output) as root:
        save_native_model(model, root)
        adapter.processor.save_pretrained(root)
        shutil.copytree(adapter.root / "speech_tokenizer", root / "speech_tokenizer")
        for name in ("preprocessor_config.json", "generation_config.json", "LICENSE", "NOTICE"):
            if (adapter.root / name).is_file():
                shutil.copy2(adapter.root / name, root / name)
        result = dict(
            format=format or "merged",
            adapter=config.adapter,
            inference_config=config.model_dump(mode="json"),
            source_checkpoint=str(Path(checkpoint).resolve()),
            source_fingerprint=metadata["source_fingerprint"],
            output=str(Path(output).resolve()),
            lora_merged=True,
            vllm_deployment_verified=False,
            sglang_deployment_verified=False,
            native_architecture="Qwen3TTSForConditionalGeneration",
            speaker=config.speech_speaker.lower(),
            speaker_reference_sha256=config.speech_reference_sha256,
        )
        write_json(root / "omnifactory_export.json", result)
    return result


def validate_export(root, metadata):
    from omnifactory.artifacts import validate_native

    validate_native(root, metadata)
    cfg = native_config(root)
    if cfg.tts_model_type != "custom_voice" or metadata["speaker"] not in cfg.talker_config.spk_id:
        raise OmniFactoryError("E_EXPORT_ASSET", "missing exported custom speaker")
    for name in ("config.json", "preprocessor_config.json"):
        if not (root / "speech_tokenizer" / name).is_file():
            raise OmniFactoryError("E_EXPORT_ASSET", "missing native speech tokenizer asset")
    if not list((root / "speech_tokenizer").glob("*.safetensors")):
        raise OmniFactoryError("E_EXPORT_ASSET", "missing speech tokenizer weights")


def build_speech_request(payload, *, view, resolver, metadata):
    """CustomVoice embeds its fixed speaker; do not send a Base clone request."""
    if len(view.conditions) > 1:
        raise OmniFactoryError("E_TTS_CONDITION", "single fixed speaker reference required")
    for condition in view.conditions:
        if (
            condition.kind != "reference_audio"
            or condition.transcript
            or condition.media.start_us
            or condition.media.end_us is not None
            or resolver.resolve(view.assets[condition.media.asset_id]).sha256
            != metadata["speaker_reference_sha256"]
        ):
            raise OmniFactoryError("E_TTS_CONDITION", "reference differs from exported speaker")
    return {**payload, "voice": metadata["speaker"], "task_type": "CustomVoice", "language": "Auto"}
