"""Native MiniCPM S3 targets and differentiable, causally shifted TTS supervision."""

import hashlib

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from omnifactory.errors import OmniFactoryError


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_target_audio(resolver, asset, media, *, max_seconds, max_bytes, target_rate=16000):
    path = resolver.resolve(asset).path
    info = sf.info(path)
    if asset.sample_rate_hz and asset.sample_rate_hz != info.samplerate:
        raise OmniFactoryError("E_AUDIO_METADATA", "target sampling rate differs from file")
    start = round(media.start_us * info.samplerate / 1e6)
    end = round(media.end_us * info.samplerate / 1e6) if media.end_us is not None else info.frames
    duration = (end - start) / info.samplerate
    required = (end - start) * info.channels * 4 + round(duration * target_rate) * 4
    if not 0 <= start < end <= info.frames:
        raise OmniFactoryError("E_AUDIO_SLICE", "target audio slice exceeds file")
    if duration > max_seconds or required > max_bytes:
        raise OmniFactoryError("E_DECODE_BUDGET", "speech target exceeds duration/byte budget")
    audio, sr = sf.read(path, start=start, stop=end, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_rate)
    if len(audio) < round(target_rate * 0.04) or not np.isfinite(audio).all():
        raise OmniFactoryError("E_AUDIO", "target audio must be finite and at least 40 ms")
    return audio, required


class S3Tokenizer:
    """Frozen official 25 Hz ONNX quantizer, checked by content hash, CPU execution."""

    def __init__(self, path, sha256):
        if file_sha256(path) != sha256:
            raise OmniFactoryError("E_CODEC_HASH", "native speech tokenizer hash differs")
        try:
            import onnxruntime as ort
            import s3tokenizer
        except ImportError as exc:
            raise OmniFactoryError(
                "E_SPEECH_INSTALL", "install with uv sync --extra train --extra speech"
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.mel = s3tokenizer.log_mel_spectrogram
        if [x.name for x in self.session.get_inputs()] != ["feats", "feats_length"]:
            raise OmniFactoryError(
                "E_CODEC_TYPE", "expected native speech_tokenizer_v2_25hz ONNX inputs"
            )

    def encode(self, waveform):
        with torch.no_grad():
            features = self.mel(torch.from_numpy(waveform)).unsqueeze(0).numpy()
        codes = self.session.run(
            None,
            {
                "feats": features.astype(np.float32),
                "feats_length": np.array([features.shape[-1]], dtype=np.int32),
            },
        )[0]
        codes = torch.from_numpy(codes.astype(np.int64)).reshape(-1)
        if not len(codes) or (codes < 0).any() or (codes >= 6561).any():
            raise OmniFactoryError("E_CODEC_TOKENS", "invalid native S3 token range")
        return codes


def speech_condition(tts, text_ids, hidden):
    if tts.condition_type != "hidden_text_merge" or tts.num_vq != 1:
        raise OmniFactoryError(
            "E_TTS_CONFIG", "requires native 4.5 hidden_text_merge, single S3 codebook"
        )
    semantic = tts.projector_semantic(hidden)
    if tts.config.normalize_projected_hidden:
        semantic = F.normalize(semantic, p=2, dim=-1)
    text = tts.emb_text(text_ids) + semantic
    suffix = torch.tensor(
        [tts.config.text_eos_token_id, tts.audio_bos_token_id], device=text_ids.device
    )
    return torch.cat([text, tts.emb_text(suffix)], dim=0)


def speech_logits(tts, condition, targets):
    """BOS predicts code[0]; code[t-1] predicts code[t], including terminal EOS."""
    prefix = condition.shape[0]
    teacher = tts.emb_code[0](targets[:-1])
    inputs = torch.cat([condition, teacher], dim=0).unsqueeze(0)
    if inputs.shape[1] > tts.config.max_position_embeddings:
        raise OmniFactoryError("E_TTS_LENGTH", "TTS sequence exceeds native position limit")
    output = tts.model(inputs_embeds=inputs, use_cache=False)
    return tts.head_code[0](output.last_hidden_state[0, prefix - 1 :]).float()


def speech_loss(model, batch, hidden, *, text_loss, text_count, text_weight, speech_weight):
    total = text_loss * text_count * text_weight if text_weight else hidden.sum() * 0
    count = text_count if text_weight else 0
    codec_sum = hidden.new_zeros(())
    codec_count = 0
    for i, (span, targets) in enumerate(zip(batch["speech_spans"], batch["codec_labels"])):
        start, end = map(int, span)
        condition = speech_condition(
            model.tts, batch["input_ids"][i, start:end], hidden[i, start:end]
        )
        logits = speech_logits(model.tts, condition, targets)
        codec_sum = codec_sum + F.cross_entropy(logits, targets, reduction="sum")
        codec_count += len(targets)
    total = total + codec_sum * speech_weight
    return total / (count + codec_count), codec_sum / codec_count


@torch.no_grad()
def generate_codes(tts, condition, max_new_tokens):
    from omnifactory._vendor.minicpm_o45.modeling_minicpmo import gen_logits
    from omnifactory._vendor.minicpm_o45.utils import TTSSamplingParams

    if condition.shape[0] + max_new_tokens > tts.config.max_position_embeddings:
        raise OmniFactoryError("E_TTS_LENGTH", "speech generation exceeds native position limit")
    inputs = condition.unsqueeze(0)
    cache = None
    codes = []
    params = TTSSamplingParams()
    warpers, processors = gen_logits(
        num_code=tts.num_audio_tokens,
        top_p=params.top_p,
        top_k=params.top_k,
        repetition_penalty=params.repetition_penalty,
    )
    for step in range(max_new_tokens):
        output = tts.model(inputs_embeds=inputs, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        scores = tts.head_code[0](output.last_hidden_state[:, -1]).float() / params.temperature
        # Match native non-streaming TTS: no history filtering on the first
        # code, a 16-code repetition window thereafter, and at least 50 codes.
        if codes:
            history = torch.tensor([codes], device=condition.device, dtype=torch.long)
            for transform in [*processors, *warpers]:
                scores = transform(history, scores)
        if step < 50:
            scores[:, tts.num_audio_tokens - 1] = -torch.inf
        token = int(torch.multinomial(scores.softmax(-1), num_samples=1))
        if token == tts.num_audio_tokens - 1:
            return codes, True
        codes.append(token)
        inputs = tts.emb_code[0](torch.tensor([[token]], device=condition.device))
    return codes, False
