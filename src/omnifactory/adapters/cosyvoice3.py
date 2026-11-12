"""Literal TTS and native mel flow supervision for CosyVoice3."""

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from omnifactory._vendor.cosyvoice3.tokenizer.tokenizer import get_qwen_tokenizer
from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
from omnifactory.adapters.cosyvoice3_features import CosyVoice3Features, pack_sequence
from omnifactory.adapters.cosyvoice3_model import (
    build_flow,
    build_hift,
    load_weights,
    model_root,
    native_config,
)
from omnifactory.adapters.supervision import tensor_ref
from omnifactory.artifacts import sha256
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import MediaAsset, MediaSlice, inference_view
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
from omnifactory.training.artifacts import new_directory, write_json
from omnifactory.training.losses import LossBundle, LossTerm, attach_losses


def text_inputs(view):
    if (
        not 1 <= len(view.messages) <= 2
        or view.messages[-1].role != "user"
        or (len(view.messages) == 2 and view.messages[0].role != "system")
        or any(b.type != "text" for m in view.messages for b in m.content)
    ):
        raise OmniFactoryError(
            "E_TTS_INPUT", "one literal user utterance with optional system instruction"
        )
    text = "".join(b.text for b in view.messages[-1].content)
    instruction = (
        "".join(b.text for b in view.messages[0].content)
        if len(view.messages) == 2
        else "You are a helpful assistant."
    )
    if not text.strip() or any("<|" in value for value in (text, instruction)):
        raise OmniFactoryError(
            "E_TTS_INPUT", "nonempty literal text without control tokens required"
        )
    return text, instruction + "<|endofprompt|>"


class CosyVoice3Adapter:
    def __init__(self, config):
        self.config = config
        self.root = model_root(config)
        self.native = native_config(self.root)
        self.processor = get_qwen_tokenizer(
            str(self.root / "CosyVoice-BlankEN"), True, "cosyvoice3"
        )
        self.resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        self.features = CosyVoice3Features(self.root, config, self.native)
        self._fixed_reference = None
        self._decode_reference = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def save_processor(self, path):
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "cosyvoice3_processor.json", {"source": str(self.root.resolve())})

    def _ids(self, text):
        return torch.tensor([self.processor.encode(text)], dtype=torch.long)

    def fixed_reference(self):
        if self._fixed_reference is not None:
            return self._fixed_reference
        bundled = self.root / "omnifactory_reference.safetensors"
        if bundled.exists():
            self._fixed_reference = load_file(bundled)
            return self._fixed_reference
        path = self.config.speech_reference
        if path is None or sha256(path) != self.config.speech_reference_sha256:
            raise OmniFactoryError(
                "E_SPEAKER_REFERENCE",
                "provide a reference condition or fixed reference with matching SHA-256",
            )
        self._fixed_reference, _ = self.features.extract(
            LocalMediaResolver(path.parent),
            MediaAsset(uri=path.name, modality="audio"),
            MediaSlice(asset_id="reference"),
        )
        return self._fixed_reference

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        view = inference_view(sample, prediction_id, ("audio",))
        text, instruction = text_inputs(view)
        text_ids, instruction_ids = self._ids(text), self._ids(instruction)
        payload = dict(text=text, text_token=text_ids, instruct_token=instruction_ids)
        size = latent_elements = frames = 0
        length = text_ids.numel() + instruction_ids.numel() + 2
        if inference:
            if len(view.conditions) > 1:
                raise OmniFactoryError("E_TTS_CONDITION", "one reference recording supported")
            if view.conditions:
                condition = view.conditions[0]
                if condition.kind != "reference_audio":
                    raise OmniFactoryError("E_TTS_CONDITION", "audio reference required")
                reference, size = self.features.extract(
                    self.resolver, view.assets[condition.media.asset_id], condition.media
                )
                transcript = condition.transcript or ""
                if "<|" in transcript:
                    raise OmniFactoryError(
                        "E_TTS_INPUT", "reference transcript must be literal text"
                    )
            else:
                reference = self.fixed_reference()
                transcript = self.config.speech_reference_text or ""
            payload.update(
                reference=reference,
                reference_text=self._ids(transcript)
                if transcript
                else torch.empty(1, 0, dtype=torch.long),
            )
            frames = reference["speech_token"].numel()
            latent_elements = reference["speech_feat"].numel()
            length += payload["reference_text"].numel() + (frames if transcript else 0)
            plan = SupervisionPlan()
        else:
            if view.conditions:
                raise OmniFactoryError(
                    "E_TTS_CONDITION",
                    "native SFT derives targets from recordings; reference conditions are for inference",
                )
            prediction = next(p for p in sample.predictions if p.id == prediction_id)
            message = next(m for m in sample.messages if m.id == prediction.target_message_id)
            targets = [b for b in message.content if b.id in prediction.target_block_ids]
            audios, texts = ([b for b in targets if b.type == kind] for kind in ("audio", "text"))
            if (
                len(targets) != 2
                or len(audios) != 1
                or len(texts) != 1
                or texts[0].text != text
                or not any(
                    p.audio_block_id == audios[0].id and p.text_block_id == texts[0].id
                    for p in sample.speech_pairs
                )
            ):
                raise OmniFactoryError(
                    "E_TTS_TARGET", "select a recording and its paired exact transcript"
                )
            features, size = self.features.extract(
                self.resolver,
                sample.assets[audios[0].media.asset_id],
                audios[0].media,
                flow=self.config.task == "flow_sft",
            )
            frames = features["speech_token"].numel()
            if self.config.task == "speech_sft":
                mode = self.config.cosyvoice_sequence_mode
                digest = hashlib.sha256(
                    f"{self.config.seed}:{sample.id}:{prediction_id}".encode()
                ).digest()
                bistream = mode == "bistream" or (mode == "mixed" and digest[0] < 128)
                payload = pack_sequence(
                    text_ids,
                    instruction_ids,
                    features["speech_token"],
                    vocabulary_size=self.native["llm"]["speech_token_size"],
                    bistream=bistream,
                )
                length = payload["input_ids"].numel()
                key, axes, unit, kind = "labels", ("batch", "time"), "symbols", "cross_entropy"
                representation = DiscreteRepresentation(
                    vocabulary_sizes=(self.native["llm"]["speech_token_size"] + 200,),
                    ignore_index=-1,
                )
                name = "codec_ce"
                mask = payload[key] != -1
            else:
                payload = features
                key, axes, unit, kind = (
                    "speech_feat",
                    ("batch", "frame", "feature"),
                    "elements",
                    "flow_matching",
                )
                representation = ContinuousRepresentation(
                    space="cosyvoice3_mel", encoder_id="native_mel_24khz"
                )
                name = "mel_flow"
                mask = torch.ones_like(payload[key], dtype=torch.bool)
                latent_elements = payload[key].numel()
            tensors, layout = pack_tree(payload)
            tensors["target_mask"] = mask
            plan = SupervisionPlan(
                targets=(
                    SupervisionTarget(
                        id="speech",
                        modality="audio",
                        representation=representation,
                        values=tensor_ref(key, tensors[key], axes),
                        mask=tensor_ref("target_mask", mask, axes),
                        measure=ReductionMeasure(unit=unit),
                    ),
                ),
                normalizers=(Normalizer(id="speech_units", unit=unit),),
                objectives=(
                    LossObjective(
                        id=name,
                        target_id="speech",
                        kind=kind,
                        normalizer="speech_units",
                        weight=self.config.speech_loss_weight,
                    ),
                ),
            )
        if length > self.config.max_length:
            raise OmniFactoryError("E_SEQUENCE_LENGTH", "CosyVoice3 sequence exceeds max_length")
        if inference:
            tensors, layout = pack_tree(payload)
        return EncodedExample(
            tensors,
            SampleCost(
                sample_id=sample.id,
                prediction_id=prediction_id,
                contract_id=contract_id,
                modalities=("text", "audio"),
                llm_tokens=length,
                codec_frames=frames,
                audio_frames=frames * 2 if self.config.task == "flow_sft" else 0,
                latent_elements=latent_elements,
                decoded_media_bytes=size,
            ),
            layout=layout,
            supervision=plan,
        ).validate(inference=inference)

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
        if self.config.task == "speech_sft":
            ids, text_mask = batch["input_ids"], batch["text_mask"]
            talker = model.talker
            text = talker.llm.model.model.embed_tokens(ids.masked_fill(~text_mask, 0))
            speech = talker.speech_embedding(ids.masked_fill(text_mask, 0))
            embeds = torch.where(text_mask[..., None], text, speech)
            hidden, _ = talker.llm(embeds, torch.tensor([ids.shape[1]], device=ids.device))
            logits = talker.llm_decoder(hidden).float()
            loss_sum = F.cross_entropy(
                logits.flatten(0, 1), batch["labels"].flatten(), ignore_index=-1, reduction="sum"
            )
            count, kind, unit, name = (
                float((batch["labels"] != -1).sum()),
                "cross_entropy",
                "symbols",
                "codec_ce",
            )
        else:
            flow = model.flow
            codes, target = batch["speech_token"], batch["speech_feat"]
            embedding = flow.spk_embed_affine_layer(F.normalize(batch["embedding"], dim=1))
            h = flow.pre_lookahead_layer(flow.input_embedding(codes)).repeat_interleave(2, dim=1)
            cond = torch.zeros_like(target)
            # Same native Bernoulli policies, using checkpointed torch RNGs only.
            streaming = bool(torch.rand(()) < 0.5)
            if bool(torch.rand(()) >= 0.5):
                end = int(torch.randint(int(0.3 * target.shape[1]) + 1, ()))
                cond[:, :end] = target[:, :end]
            loss, _ = flow.decoder.compute_loss(
                target.transpose(1, 2).contiguous(),
                torch.ones(1, 1, target.shape[1], device=target.device, dtype=target.dtype),
                h.transpose(1, 2).contiguous(),
                embedding,
                cond=cond.transpose(1, 2),
                streaming=streaming,
            )
            count, kind, unit, name = float(target.numel()), "flow_matching", "elements", "mel_flow"
            loss_sum, logits = loss * count, None
        bundle = LossBundle(
            (
                LossTerm(
                    name, loss_sum, "speech_units", weight=self.config.speech_loss_weight, kind=kind
                ),
            ),
            {"speech_units": count},
            {"speech_units": unit},
        )
        return attach_losses(SimpleNamespace(logits=logits), bundle)

    def generate(self, model, batch, *, max_new_tokens):
        raise OmniFactoryError("E_TTS_OUTPUT", "CosyVoice3 requires generate_audio")

    @torch.no_grad()
    def generate_speech(self, model, batch, *, max_new_tokens, max_codec_tokens):
        talker = model.talker
        ref_text = batch["reference_text"]
        text = torch.cat([batch["instruct_token"], ref_text, batch["text_token"]], dim=1)
        prefix = [
            talker.speech_embedding.weight[talker.sos][None, None],
            talker.llm.model.model.embed_tokens(text),
            talker.speech_embedding.weight[talker.task_id][None, None],
        ]
        if ref_text.numel():
            prefix.append(talker.speech_embedding(batch["reference"]["speech_token"]))
        prefix = torch.cat(prefix, dim=1)
        if prefix.shape[1] + max_codec_tokens > self.config.max_length:
            raise OmniFactoryError(
                "E_SEQUENCE_LENGTH", "prompt plus audio bound exceeds max_length"
            )
        captured = {}
        original = talker.sampling_ids

        def capture(*args, **kwargs):
            token = original(*args, **kwargs)
            captured["last"] = int(token)
            return token

        talker.sampling_ids = capture
        try:
            codes = list(
                talker.inference_wrapper(
                    prefix,
                    25,
                    min(2 * batch["text_token"].numel(), max_codec_tokens - 1),
                    max_codec_tokens,
                    "omnifactory",
                )
            )
        finally:
            talker.sampling_ids = original
        self._decode_reference = batch["reference"]
        stop = captured.get("last")
        return dict(
            text=batch["text"],
            codec_tokens=codes,
            speech_finished=stop == talker.eos_token,
            speech_stopped=stop in talker.stop_token_ids,
            speech_eos=stop == talker.eos_token,
            stop_token_id=stop if stop in talker.stop_token_ids else None,
        )

    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None):
        if speech_assets is not None or reference_audio is not None:
            raise OmniFactoryError(
                "E_TTS_CONDITION", "use reference conditions or speech_reference in the recipe"
            )
        # Frozen decoding stages stay FP32; never round their weights through BF16.
        flow = (
            load_weights(build_flow(self.native["flow"]), self.root / "flow.pt")
            .to(self.config.device)
            .eval()
        )
        hift = (
            load_weights(build_hift(self.native["hift"]), self.root / "hift.pt")
            .to(self.config.device)
            .eval()
        )

        @torch.no_grad()
        def decode(codes):
            if not codes or self._decode_reference is None:
                raise OmniFactoryError("E_AUDIO_DECODE", "missing generated codes/reference")
            ref = move_tree(self._decode_reference, device=self.config.device, dtype=torch.float32)
            token = torch.tensor([codes], device=self.config.device, dtype=torch.long)
            mel, _ = flow.inference(
                token,
                torch.tensor([len(codes)]),
                ref["speech_token"],
                torch.tensor([ref["speech_token"].shape[1]]),
                ref["speech_feat"],
                torch.tensor([ref["speech_feat"].shape[1]]),
                ref["embedding"],
                streaming=False,
                finalize=True,
            )
            waveform, _ = hift.inference(mel.float(), finalize=True)
            wave = waveform.float().cpu().reshape(-1).numpy()
            if not len(wave) or not torch.tensor(wave).isfinite().all():
                raise OmniFactoryError("E_AUDIO_DECODE", "empty/nonfinite native waveform")
            return wave, 24000

        return decode


def export_cosyvoice3(checkpoint, output, *, format=None):
    from omnifactory.training.engine import load_checkpoint
    from omnifactory.training.policy import merge_lora

    if format not in (None, "merged", "vllm-omni", "sglang-omni"):
        raise OmniFactoryError("E_EXPORT_FORMAT", "CosyVoice3 requires native pipeline export")
    model, adapter, config, metadata = load_checkpoint(checkpoint, for_export=True)
    model = merge_lora(model)
    stage, module = (
        ("llm.pt", model.talker) if config.task == "speech_sft" else ("flow.pt", model.flow)
    )
    with new_directory(output) as root:
        for name in (
            "CosyVoice-BlankEN",
            "cosyvoice3.yaml",
            "config.json",
            "configuration.json",
            "llm.pt",
            "flow.pt",
            "hift.pt",
            "campplus.onnx",
            "speech_tokenizer_v3.onnx",
            "README.md",
            "LICENSE",
        ):
            source = adapter.root / name
            if name == stage or not source.exists():
                continue
            if source.is_dir():
                shutil.copytree(source, root / name, ignore=shutil.ignore_patterns(".cache"))
            else:
                shutil.copy2(source, root / name)
        torch.save(module.state_dict(), root / stage)
        reference_file = None
        if (
            config.speech_reference is not None
            or (adapter.root / "omnifactory_reference.safetensors").exists()
        ):
            save_file(
                {k: v.contiguous() for k, v in adapter.fixed_reference().items()},
                root / "omnifactory_reference.safetensors",
            )
            source = config.speech_reference
            if source is not None and source.is_file():
                reference_file = "reference" + source.suffix
                shutil.copy2(source, root / reference_file)
        result = dict(
            format=format or "merged",
            adapter=config.adapter,
            inference_config={**config.model_dump(mode="json"), "task": "speech_sft"},
            trained_stage=stage,
            source_checkpoint=str(Path(checkpoint).resolve()),
            source_fingerprint=metadata["source_fingerprint"],
            output=str(Path(output).resolve()),
            lora_merged=True,
            vllm_deployment_verified=False,
            reference_file=reference_file,
            native_architecture="CosyVoice3Model",
        )
        write_json(root / "omnifactory_export.json", result)
    return result


def validate_export(root, metadata):
    native_config(root)
    for name in (
        "llm.pt",
        "flow.pt",
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v3.onnx",
        "CosyVoice-BlankEN/config.json",
        "CosyVoice-BlankEN/tokenizer_config.json",
    ):
        if not (root / name).is_file():
            raise OmniFactoryError("E_EXPORT_ASSET", f"missing native CosyVoice3 asset: {name}")


def build_speech_request(payload, *, view, resolver, metadata):
    """Build native zero-shot HTTP input from inference-visible conditioning only."""
    import base64
    import io

    import soundfile as sf

    from omnifactory.adapters.speech import load_target_audio

    text, instruction = text_inputs(view)
    config = metadata["inference_config"]
    if len(view.conditions) > 1:
        raise OmniFactoryError("E_SERVE_CONDITION", "one reference recording supported")
    if view.conditions:
        condition = view.conditions[0]
        if condition.kind != "reference_audio":
            raise OmniFactoryError("E_SERVE_CONDITION", "audio reference required")
        asset, media = view.assets[condition.media.asset_id], condition.media
        transcript = condition.transcript
    else:
        name = metadata.get("reference_file")
        if not name or Path(name).name != name or not metadata.get("_export_root"):
            raise OmniFactoryError("E_SERVE_CONDITION", "export has no bundled reference recording")
        root = Path(metadata["_export_root"])
        resolver = LocalMediaResolver(root)
        asset, media = MediaAsset(uri=name, modality="audio"), MediaSlice(asset_id="reference")
        if sha256(resolver.resolve(asset).path) != config.get("speech_reference_sha256"):
            raise OmniFactoryError("E_SPEAKER_REFERENCE", "bundled reference hash differs")
        transcript = config.get("speech_reference_text")
    if not transcript or not transcript.strip() or "<|" in transcript:
        raise OmniFactoryError(
            "E_SERVE_CONDITION",
            "CosyVoice3 serving requires the reference recording's literal transcript",
        )
    wave, _ = load_target_audio(
        resolver,
        asset,
        media,
        max_seconds=config["max_audio_seconds"],
        max_bytes=config["budget"]["max_decoded_media_bytes"],
        target_rate=16000,
    )
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format="WAV", subtype="FLOAT")
    sglang = metadata.get("_serving_backend") == "sglang-omni"
    return {
        # SGLang rejects separate instructions together with a transcript. Its
        # zero-shot builder preserves this complete native prompt verbatim.
        **{k: v for k, v in payload.items() if not (sglang and k == "instructions")},
        "input": text,
        "ref_text": instruction + transcript,
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode(),
    }
