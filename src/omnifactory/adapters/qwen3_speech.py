"""Native Qwen3 16-codebook teacher forcing; Code2Wav stays outside training.

Targets keep [frame, codebook] axes. EOS supervises only the primary head;
residual heads predict within-frame codes, independently for each frame.
"""

import torch
from transformers import PretrainedConfig, Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoePreTrainedModel,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from omnifactory.adapters.qwen_speech import NativeCodeLabels, speech_target, text_hash
from omnifactory.adapters.speech import file_sha256
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import digest

CODEC = "qwen3_omni_16x2048_v1"
CODEBOOKS = 16
CODEBOOK_SIZE = 2048
CODEC_EOS = 2150
PRIMARY_VOCAB = 3072


class Qwen3CodeLabels(NativeCodeLabels):
    def get(self, sample, prediction_id, resolver, speaker, max_tokens):
        answer, block = speech_target(sample, prediction_id)
        row = self.rows.get((sample.id, prediction_id), {})
        expected = {
            "codec": CODEC,
            "audio_block_id": block.id,
            "audio_sha256": file_sha256(resolver.resolve(sample.assets[block.media.asset_id]).path),
            "transcript_sha256": text_hash(answer),
            "sample_sha256": digest(sample.model_dump(mode="json")),
            "speaker": speaker,
            "start_us": block.media.start_us,
            "end_us": block.media.end_us,
        }
        if any(row.get(k) != v for k, v in expected.items()):
            raise OmniFactoryError(
                "E_CODEC_MANIFEST", "Qwen3 native codec/sample/audio identity mismatch"
            )
        if row.get("origin") not in ("native_teacher", "native_encoder") or not row.get(
            "source_fingerprint"
        ):
            raise OmniFactoryError("E_CODEC_MANIFEST", "native label provenance is required")
        if (
            row.get("speech_finished") is not True
            or type(row.get("eos_token_id")) is not int
            or row["eos_token_id"] != CODEC_EOS
        ):
            raise OmniFactoryError(
                "E_CODEC_MANIFEST", "Qwen3 training requires complete native speech with EOS 2150"
            )
        codes = row.get("codes")
        if (
            not isinstance(codes, list)
            or not codes
            or len(codes) + 1 > max_tokens
            or any(
                not isinstance(frame, list)
                or len(frame) != CODEBOOKS
                or any(type(c) is not int or not 0 <= c < CODEBOOK_SIZE for c in frame)
                for frame in codes
            )
        ):
            raise OmniFactoryError(
                "E_CODEC_TOKENS",
                "expected bounded Qwen3 [frames,16] codes in [0,2047], without EOS",
            )
        labels = torch.tensor(codes + [[CODEC_EOS] + [-100] * (CODEBOOKS - 1)])
        return labels, {
            **expected,
            "origin": row["origin"],
            "source_fingerprint": row["source_fingerprint"],
            "speech_finished": True,
        }


class Qwen3SpeechModel(Qwen3OmniMoePreTrainedModel):
    config_class = Qwen3OmniMoeConfig
    base_model_prefix = ""
    _keys_to_ignore_on_load_unexpected = [r"code2wav\..*"]
    _get_talker_user_parts = Qwen3OmniMoeForConditionalGeneration._get_talker_user_parts
    _get_talker_assistant_parts = Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts

    def __init__(self, config):
        super().__init__(config)
        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration._from_config(
            config.thinker_config
        )
        self.talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(config.talker_config)
        self.post_init()


def load_qwen3_speech(config):
    if config.quantization != "none":
        raise OmniFactoryError("E_QUANTIZATION", "Qwen3 speech quantization is not verified")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise OmniFactoryError("E_DEVICE", "CUDA requested but unavailable")
    raw, _ = PretrainedConfig.get_config_dict(config.model, revision=config.revision)
    if raw.get("model_type") != "qwen3_omni_moe" or not raw.get("enable_audio_output", True):
        raise OmniFactoryError("E_MODEL_TYPE", "speech requires a full Qwen3-Omni checkpoint")
    native = Qwen3OmniMoeConfig.from_dict(raw)
    talker = native.talker_config
    if (
        talker.num_code_groups != CODEBOOKS
        or talker.code_predictor_config.num_code_groups != CODEBOOKS
        or talker.text_config.vocab_size != PRIMARY_VOCAB
        or talker.code_predictor_config.vocab_size != CODEBOOK_SIZE
        or talker.codec_eos_token_id != CODEC_EOS
        or native.code2wav_config.num_quantizers != CODEBOOKS
        or native.code2wav_config.codebook_size != CODEBOOK_SIZE
    ):
        raise OmniFactoryError(
            "E_NATIVE_STAGES", "requires Qwen3 native 16x2048 codec and primary EOS 2150"
        )
    for cfg in (native.thinker_config.text_config, talker.text_config):
        if cfg.num_hidden_layers > 48 or cfg.output_router_logits:
            raise OmniFactoryError(
                "E_NATIVE_STAGES", "unmapped depth or unsupported router objective"
            )
    if config.speech_speaker.lower() not in (talker.speaker_id or {}):
        raise OmniFactoryError("E_SPEAKER", f"available speakers: {list(talker.speaker_id or {})}")
    model, info = Qwen3SpeechModel.from_pretrained(
        config.model,
        config=native,
        revision=config.revision,
        dtype=getattr(torch, config.dtype),
        attn_implementation=config.attention,
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise OmniFactoryError("E_CHECKPOINT_INCOMPLETE", str(info))
    model.to(config.device)
    model.speech_speaker = config.speech_speaker
    for cfg in (
        model.thinker.config,
        model.thinker.config.text_config,
        model.talker.config.text_config,
        model.talker.code_predictor.config,
    ):
        cfg.use_cache = False
    return model


def talker_condition(model, batch, output, start, end):
    """Same ChatML segmentation and native projection helpers as HF generation."""
    ids = batch["input_ids"]
    if ids.shape[0] != 1 or not 0 < start < end <= ids.shape[1]:
        raise OmniFactoryError(
            "E_SPEECH_SPAN", "Qwen3 conditioning requires one unpadded nonempty transcript"
        )
    cfg = model.config
    starts = (ids[0, :start] == cfg.im_start_token_id).nonzero().flatten().tolist()
    if (
        not starts
        or starts[-1] + 3 != start
        or int(ids[0, starts[-1] + 1]) != cfg.assistant_token_id
    ):
        raise OmniFactoryError("E_SPEECH_SPAN", "native three-token assistant header required")
    # HF generation's hidden states do not contain the final sampled text EOS.
    embed = output.hidden_states[0][:, :end]
    hidden = output.hidden_states[cfg.talker_config.accept_hidden_layer][:, :end]
    mm = sum(
        (ids[:, :end] == getattr(cfg.thinker_config, k)).int()
        for k in ("audio_token_id", "image_token_id", "video_token_id")
    ).bool()
    specials = torch.tensor(
        [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]], device=ids.device
    )
    bos, eos, pad = model.talker.text_projection(
        model.thinker.get_input_embeddings()(specials)
    ).chunk(3, dim=1)
    pieces, token_parts, trailing = [], [], None
    for i, begin in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else end
        role = int(ids[0, begin + 1])
        if role == cfg.system_token_id or (role == cfg.assistant_token_id and i != len(starts) - 1):
            continue
        if role == cfg.user_token_id:
            pieces.append(model._get_talker_user_parts(begin, stop, mm, hidden, embed))
            token_parts.append(ids[:, begin:stop])
        elif role == cfg.assistant_token_id and i == len(starts) - 1:
            part, tokens, trailing = model._get_talker_assistant_parts(
                begin,
                stop,
                cfg.talker_config.speaker_id[model.speech_speaker.lower()],
                embed,
                pad,
                bos,
                eos,
            )
            pieces.append(part)
            token_parts.append(tokens)
        else:
            raise OmniFactoryError("E_SPEECH_ROLE", "unsupported Qwen3 conversation role")
    if trailing is None:
        raise OmniFactoryError("E_SPEECH_SPAN", "missing assistant conditioning")
    prefix, tokens = torch.cat(pieces, 1), torch.cat(token_parts, 1)
    media = dict(
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        use_audio_in_video=batch.get("use_audio_in_video", False),
        audio_seqlens=batch["feature_attention_mask"].sum(-1)
        if "feature_attention_mask" in batch
        else None,
        second_per_grids=batch.get("video_second_per_grid"),
    )
    positions, _ = model.talker.get_rope_index(
        tokens, attention_mask=torch.ones_like(tokens), **media
    )
    return prefix, trailing, pad, positions, tokens, media


def talker_logits(model, condition, labels):
    """Primary frame causal attention followed by within-frame residual attention."""
    prefix, trailing, pad, positions, _, _ = condition
    if (
        labels.ndim != 2
        or labels.shape[1] != CODEBOOKS
        or len(labels) < 2
        or int(labels[-1, 0]) != CODEC_EOS
        or (labels[-1, 1:] != -100).any()
        or (labels[:-1] < 0).any()
        or (labels[:-1] >= CODEBOOK_SIZE).any()
    ):
        raise OmniFactoryError(
            "E_CODEC_TOKENS", "expected complete frames plus primary-only EOS row"
        )
    frames = labels[:-1]
    predictor = model.talker.code_predictor
    # Previous frame's full residual embedding sum predicts the next primary.
    previous = model.talker.get_input_embeddings()(frames[:, 0])
    for i, embedding in enumerate(predictor.get_input_embeddings()):
        previous = previous + embedding(frames[:, i + 1])
    index = torch.arange(len(frames), device=prefix.device)
    text = trailing[:, index.clamp(max=trailing.shape[1] - 1)]
    text = torch.where((index < trailing.shape[1])[None, :, None], text, pad)
    inputs = torch.cat([prefix, previous.unsqueeze(0) + text], dim=1)
    extra = positions.max() + torch.arange(1, len(frames) + 1, device=prefix.device)
    positions = torch.cat([positions, extra.view(1, 1, -1).expand(3, 1, -1)], dim=-1)
    # Call the native backbone with explicit positions: Talker.forward's attention
    # mask branch recomputes prefix positions for generation, not teacher forcing.
    hidden = model.talker.model(
        inputs_embeds=inputs,
        position_ids=positions,
        attention_mask=torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[:, prefix.shape[1] - 1 :]
    primary = model.talker.codec_head(hidden)[0]
    residual_inputs = [hidden[0, :-1], model.talker.get_input_embeddings()(frames[:, 0])]
    residual_inputs += [
        embedding(frames[:, i + 1])
        for i, embedding in enumerate(predictor.get_input_embeddings()[:-1])
    ]
    residual_hidden = predictor.model(
        inputs_embeds=torch.stack(residual_inputs, 1), use_cache=False, return_dict=True
    ).last_hidden_state
    residual = torch.stack(
        [head(residual_hidden[:, i + 1]) for i, head in enumerate(predictor.lm_head)], 1
    )
    return primary, residual


@torch.no_grad()
def generate_talker(model, condition, max_tokens, *, do_sample=True):
    prefix, trailing, pad, _, tokens, media = condition
    if max_tokens < 2:
        raise OmniFactoryError("E_GENERATION", "at least one frame and EOS required")
    result = model.talker.generate(
        inputs_embeds=prefix,
        trailing_text_hidden=trailing,
        tts_pad_embed=pad,
        talker_input_ids=tokens,
        image_grid_thw=media["image_grid_thw"],
        video_grid_thw=media["video_grid_thw"],
        use_audio_in_video=media["use_audio_in_video"],
        audio_feature_lengths=media["audio_seqlens"],
        video_second_per_grid=media["second_per_grids"],
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        eos_token_id=CODEC_EOS,
        repetition_penalty=1.05,
        suppress_tokens=[i for i in range(CODEBOOK_SIZE, PRIMARY_VOCAB) if i != CODEC_EOS],
        output_hidden_states=True,
        return_dict_in_generate=True,
        use_cache=True,
    )
    frames = [h[-1] for h in result.hidden_states if h[-1] is not None]
    codes = torch.cat(frames, 0).tolist() if frames else []
    primary = result.sequences[0].tolist()
    finished = bool(primary and primary[-1] == CODEC_EOS)
    if finished and (len(codes) + 1 != len(primary) or [c[0] for c in codes] != primary[:-1]):
        raise OmniFactoryError(
            "E_CODEC_ALIGNMENT", "native primary/residual generation alignment changed"
        )
    return codes, CODEC_EOS if finished else None


class Qwen3NativeDecoder(Qwen3OmniMoeCode2Wav):
    base_model_prefix = "code2wav"


def make_decoder(config):
    native = Qwen3OmniMoeConfig.from_pretrained(config.model, revision=config.revision)
    decoder, info = Qwen3NativeDecoder.from_pretrained(
        config.model,
        config=native.code2wav_config,
        revision=config.revision,
        dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise OmniFactoryError("E_CHECKPOINT_INCOMPLETE", str(info))
    decoder.to(config.device).eval()

    @torch.no_grad()
    def decode(codes):
        values = torch.tensor(codes, device=config.device, dtype=torch.long)
        if (
            values.ndim != 2
            or not len(values)
            or values.shape[1] != CODEBOOKS
            or (values < 0).any()
            or (values >= CODEBOOK_SIZE).any()
        ):
            raise OmniFactoryError(
                "E_CODEC_TOKENS", "Code2Wav requires [frames,16] native codes, without EOS"
            )
        waveform = (
            decoder.chunked_decode(values.T.unsqueeze(0), chunk_size=300, left_context_size=25)
            .float()
            .reshape(-1)
        )
        if not waveform.numel() or not torch.isfinite(waveform).all():
            raise OmniFactoryError("E_WAVEFORM", "native Qwen3 decoder produced invalid audio")
        return waveform.cpu().numpy(), 24000

    return decode
