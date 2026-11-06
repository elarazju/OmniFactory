"""Native Qwen2.5 Thinker–Talker training with provenance-checked codec labels.

The waveform-to-code encoder is not published with Qwen2.5-Omni. Labels must
come from a compatible native source (e.g. explicit teacher synthesis), never
from MiniCPM S3 or Qwen3 tokenizers. Token2Wav is loaded only for synthesis.
"""

import hashlib
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig, Qwen2_5OmniConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniPreTrainedModel,
    Qwen2_5OmniTalkerForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniToken2WavModel,
)
from transformers.utils import cached_file

from omnifactory.adapters.speech import file_sha256
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import digest

CODEC = "qwen2_5_omni_v1"
CODEBOOK_SIZE = 8192
CODEC_EOS = 8294
CODEC_STOPS = (8292, 8294)


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def speech_target(sample, prediction_id):
    prediction = next(p for p in sample.predictions if p.id == prediction_id)
    message = next(m for m in sample.messages if m.id == prediction.target_message_id)
    blocks = [b for b in message.content if b.id in prediction.target_block_ids]
    texts = [b for b in blocks if b.type == "text"]
    audio = [b for b in blocks if b.type == "audio"]
    if len(blocks) != 2 or len(texts) != 1 or len(audio) != 1:
        raise OmniFactoryError(
            "E_SPEECH_TARGET", "select exactly one transcript and one audio block"
        )
    if audio[0].placement is not None or not any(
        p.text_block_id == texts[0].id and p.audio_block_id == audio[0].id
        for p in sample.speech_pairs
    ):
        raise OmniFactoryError(
            "E_SPEECH_TARGET", "speech target needs an utterance text/audio pair without placement"
        )
    return texts[0].text, audio[0]


class NativeCodeLabels:
    def __init__(self, path):
        self.rows = {}
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["sample_id"], row["prediction_id"])
                if key in self.rows:
                    raise OmniFactoryError("E_CODEC_MANIFEST", f"duplicate code labels: {key}")
                self.rows[key] = row

    def get(self, sample, prediction_id, resolver, speaker, max_tokens):
        answer, block = speech_target(sample, prediction_id)
        row = self.rows.get((sample.id, prediction_id), {})
        asset = sample.assets[block.media.asset_id]
        expected = {
            "codec": CODEC,
            "audio_block_id": block.id,
            "audio_sha256": file_sha256(resolver.resolve(asset).path),
            "transcript_sha256": text_hash(answer),
            "sample_sha256": digest(sample.model_dump(mode="json")),
            "speaker": speaker,
            "start_us": block.media.start_us,
            "end_us": block.media.end_us,
        }
        if any(row.get(k) != v for k, v in expected.items()):
            raise OmniFactoryError(
                "E_CODEC_MANIFEST",
                "native codec identity, sample, transcript, speaker or audio hash mismatch",
            )
        if row.get("origin") not in ("native_teacher", "native_encoder") or not row.get(
            "source_fingerprint"
        ):
            raise OmniFactoryError("E_CODEC_MANIFEST", "native label provenance is required")
        if type(row.get("speech_finished")) is not bool:
            raise OmniFactoryError(
                "E_CODEC_MANIFEST", "speech_finished must explicitly mark EOS versus truncation"
            )
        eos = row.get("eos_token_id")
        if (row["speech_finished"] and (type(eos) is not int or eos not in CODEC_STOPS)) or (
            not row["speech_finished"] and eos is not None
        ):
            raise OmniFactoryError(
                "E_CODEC_MANIFEST", "EOS identity must match native completion status"
            )
        codes = row.get("codes", [])
        if (
            not codes
            or len(codes) + int(row["speech_finished"]) > max_tokens
            or any(type(c) is not int or not 0 <= c < CODEBOOK_SIZE for c in codes)
        ):
            raise OmniFactoryError(
                "E_CODEC_TOKENS", "expected bounded native Qwen codes in [0,8191], without EOS"
            )
        return torch.tensor(codes + ([eos] if row["speech_finished"] else [])), {
            **expected,
            "origin": row["origin"],
            "source_fingerprint": row["source_fingerprint"],
            "speech_finished": row.get("speech_finished"),
        }


class QwenSpeechModel(Qwen2_5OmniPreTrainedModel):
    """Load only trainable graph stages; keep the frozen decoder out of training RAM."""

    config_class = Qwen2_5OmniConfig
    base_model_prefix = ""
    _keys_to_ignore_on_load_unexpected = [r"token2wav\..*"]

    def __init__(self, config):
        super().__init__(config)
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration(config.talker_config)
        self.post_init()


def load_speakers(model, revision=None):
    path = cached_file(model, "spk_dict.pt", revision=revision)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_qwen_speech(config):
    if config.quantization != "none":
        raise OmniFactoryError("E_QUANTIZATION", "native Talker NF4 is not verified")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise OmniFactoryError("E_DEVICE", "CUDA requested but unavailable")
    raw, _ = PretrainedConfig.get_config_dict(config.model, revision=config.revision)
    if raw.get("model_type") != "qwen2_5_omni" or not all(
        key in raw for key in ("thinker_config", "talker_config", "token2wav_config")
    ):
        raise OmniFactoryError(
            "E_MODEL_TYPE", "speech training requires a full native Qwen2.5-Omni checkpoint"
        )
    native = Qwen2_5OmniConfig.from_dict(raw)
    if not native.enable_audio_output or native.talker_config.tts_codec_end_token_id != CODEC_EOS:
        raise OmniFactoryError(
            "E_NATIVE_STAGES", "requires native Qwen2.5 Thinker/Talker configuration"
        )
    model, info = QwenSpeechModel.from_pretrained(
        config.model,
        config=native,
        revision=config.revision,
        dtype=getattr(torch, config.dtype),
        attn_implementation=config.attention,
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise OmniFactoryError("E_CHECKPOINT_INCOMPLETE", str(info))
    speakers = load_speakers(config.model, config.revision)
    if config.speech_speaker not in speakers:
        raise OmniFactoryError("E_SPEAKER", f"available speakers: {list(speakers)}")
    model.speech_speaker = config.speech_speaker
    model.speaker_bos = int(speakers[config.speech_speaker]["bos_token"])
    model.to(config.device)
    model.thinker.config.use_cache = False
    model.thinker.config.text_config.use_cache = False
    model.talker.config.use_cache = False
    return model


def talker_condition(model, batch, output, start, end):
    """Match native dual-track conditioning and multimodal RoPE for one unpadded example."""
    if not 0 < start < end <= batch["input_ids"].shape[1]:
        raise OmniFactoryError("E_SPEECH_SPAN", "speech requires a nonempty transcript")
    thinker, talker = model.thinker, model.talker
    ids = batch["input_ids"]
    embed = thinker.get_input_embeddings()
    token = embed(ids)
    # The native wrapper removes raw media embeddings from Talker while retaining
    # contextual Thinker states. Reconstruct this without mutating autograd tensors.
    media = torch.zeros_like(ids, dtype=torch.bool)
    for name in ("audio_token_index", "image_token_index", "video_token_index"):
        media |= ids == getattr(thinker.config, name)
    token = token.masked_fill(media.unsqueeze(-1), 0)
    hidden = output.hidden_states[-1]
    context = hidden[:, :start] + token[:, :start]
    reply = hidden[:, start:end] + token[:, start:end]

    def special(value):
        return torch.tensor([[value]], device=ids.device)

    bos = special(model.speaker_bos)
    codec_pad = talker.get_input_embeddings()(special(talker.codec_pad_token))
    codec_bos = talker.get_input_embeddings()(special(talker.codec_bos_token))
    prefix = torch.cat([context, embed(bos) + codec_pad, reply[:, :1] + codec_bos], dim=1)
    remaining = torch.cat(
        [
            reply[:, 1:],
            embed(special(talker.text_eos_token)),
            embed(special(talker.text_pad_token)),
        ],
        dim=1,
    )
    text_ids = torch.cat([ids[:, :start], bos, ids[:, start : start + 1]], dim=1)
    positions, _ = talker.get_rope_index(
        text_ids,
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        attention_mask=torch.ones_like(text_ids),
        use_audio_in_video=batch.get("use_audio_in_video", False),
        audio_seqlens=batch["feature_attention_mask"].sum(-1)
        if "feature_attention_mask" in batch
        else None,
        second_per_grids=batch.get("video_second_per_grid"),
    )
    return prefix, remaining, positions


def talker_logits(model, condition, codes):
    """One causal shift: audio BOS predicts code[0], previous code predicts next."""
    prefix, remaining, positions = condition
    previous = codes[:-1]
    if len(previous):
        index = torch.arange(len(previous), device=prefix.device).clamp(max=remaining.shape[1] - 1)
        tail = model.talker.get_input_embeddings()(previous.unsqueeze(0)) + remaining[:, index]
        embeds = torch.cat([prefix, tail], dim=1)
        extra = torch.arange(1, len(previous) + 1, device=prefix.device) + positions.max()
        positions = torch.cat([positions, extra.view(1, 1, -1).expand(3, 1, -1)], dim=-1)
    else:
        embeds = prefix
    result = model.talker(
        inputs_embeds=embeds,
        position_ids=positions,
        attention_mask=torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long),
        use_cache=False,
        return_dict=True,
    )
    return result.logits[:, prefix.shape[1] - 1 :]


@torch.no_grad()
def generate_talker(model, condition, max_tokens, *, do_sample=False):
    if max_tokens < 1:
        raise OmniFactoryError("E_GENERATION", "max_codec_tokens must be positive")
    prefix, remaining, positions = condition
    result = model.talker(
        inputs_embeds=prefix, position_ids=positions, use_cache=True, return_dict=True
    )
    codes, end_token = [], None
    if do_sample:
        from transformers import (
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        # Defaults of the pinned native Qwen2.5 wrapper. Greedy codec decoding
        # can repeat indefinitely even for a short supplied transcript.
        repetition = RepetitionPenaltyLogitsProcessor(1.05)
        warpers = (TemperatureLogitsWarper(0.9), TopKLogitsWarper(40), TopPLogitsWarper(0.8))
    for index in range(max_tokens):
        logits = result.logits[0, -1].float().clone()
        logits[CODEBOOK_SIZE:] = -torch.inf
        if codes:
            for stop in CODEC_STOPS:
                logits[stop] = result.logits[0, -1, stop]
        if do_sample:
            history = torch.tensor([codes], device=prefix.device, dtype=torch.long)
            scores = repetition(history, logits.unsqueeze(0)) if codes else logits.unsqueeze(0)
            for warper in warpers:
                scores = warper(history, scores)
            code = int(torch.multinomial(scores.softmax(-1), 1))
        else:
            code = int(logits.argmax())
        if code in CODEC_STOPS:
            end_token = code
            break
        codes.append(code)
        if index + 1 == max_tokens:
            break
        embedding = model.talker.get_input_embeddings()(
            torch.tensor([[code]], device=prefix.device)
        )
        embedding = (
            embedding
            + remaining[
                :, min(index, remaining.shape[1] - 1) : min(index, remaining.shape[1] - 1) + 1
            ]
        )
        pos = (positions.max() + index + 1).expand(3, 1, 1)
        result = model.talker(
            inputs_embeds=embedding,
            position_ids=pos,
            past_key_values=result.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    return codes, end_token


class QwenNativeDecoder(Qwen2_5OmniToken2WavModel):
    base_model_prefix = "token2wav"


def make_decoder(config, *, speaker=None):
    native = Qwen2_5OmniConfig.from_pretrained(config.model, revision=config.revision)
    decoder, info = QwenNativeDecoder.from_pretrained(
        config.model,
        config=native.token2wav_config,
        revision=config.revision,
        dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise OmniFactoryError("E_CHECKPOINT_INCOMPLETE", str(info))
    decoder.to(config.device).eval()
    voice = load_speakers(config.model, config.revision)[speaker or config.speech_speaker]

    @torch.no_grad()
    def decode(codes):
        if not codes or any(not 0 <= c < CODEBOOK_SIZE for c in codes):
            raise OmniFactoryError(
                "E_CODEC_TOKENS", "decoder requires native Qwen codes without EOS"
            )
        waveform = decoder(
            torch.tensor([codes], device=config.device),
            conditioning=voice["cond"].to(device=config.device, dtype=torch.float32),
            reference_mel=voice["ref_mel"].to(device=config.device, dtype=torch.float32),
        ).reshape(-1)
        if not waveform.numel() or not torch.isfinite(waveform).all():
            raise OmniFactoryError("E_WAVEFORM", "native Qwen decoder produced invalid audio")
        return waveform.cpu().numpy(), 24000

    return decode
