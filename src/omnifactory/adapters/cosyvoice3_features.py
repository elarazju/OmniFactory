"""Frozen native ONNX quantizer, speaker embedding, and aligned 80-bin mel."""

import numpy as np
import torch
import torchaudio

from omnifactory._vendor.cosyvoice3.audio import mel_spectrogram
from omnifactory.adapters.speech import load_target_audio
from omnifactory.errors import OmniFactoryError


class CosyVoice3Features:
    def __init__(self, root, config, native):
        self.root, self.config, self.native = root, config, native
        self.sessions = {}

    def session(self, name):
        import onnxruntime as ort

        if name not in self.sessions:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self.sessions[name] = ort.InferenceSession(
                str(self.root / name), sess_options=options, providers=["CPUExecutionProvider"]
            )
        return self.sessions[name]

    def extract(self, resolver, asset, media, *, flow=True):
        import s3tokenizer
        import soundfile as sf

        source_rate = sf.info(resolver.resolve(asset).path).samplerate
        wave, size = load_target_audio(
            resolver,
            asset,
            media,
            max_seconds=self.config.max_audio_seconds,
            max_bytes=self.config.budget.max_decoded_media_bytes,
            target_rate=source_rate,
        )
        extra = round(len(wave) / source_rate * (16000 + 24000) * 4)
        if size + extra > self.config.budget.max_decoded_media_bytes:
            raise OmniFactoryError("E_DECODE_BUDGET", "native resampled buffers exceed budget")
        wave = torch.from_numpy(wave)[None]
        w16 = torchaudio.functional.resample(wave, source_rate, 16000)
        codec = self.session("speech_tokenizer_v3.onnx")
        features = s3tokenizer.log_mel_spectrogram(w16, n_mels=128)
        inputs = codec.get_inputs()
        length_dtype = np.int64 if inputs[1].type == "tensor(int64)" else np.int32
        values = codec.run(
            None,
            {
                inputs[0].name: features.numpy(),
                inputs[1].name: np.array([features.shape[-1]], dtype=length_dtype),
            },
        )[0]
        codes = torch.tensor(values, dtype=torch.long).reshape(1, -1)
        if (
            not codes.numel()
            or (codes < 0).any()
            or (codes >= self.native["llm"]["speech_token_size"]).any()
        ):
            raise OmniFactoryError("E_CODEC_TOKENS", "invalid CosyVoice3 speech token range")
        payload = {"speech_token": codes}
        if flow:
            w24 = torchaudio.functional.resample(wave, source_rate, 24000)
            mel = mel_spectrogram(w24, **self.native["feat_extractor"]).transpose(1, 2)
            n = min(codes.shape[1], mel.shape[1] // 2)
            if not n:
                raise OmniFactoryError("E_AUDIO", "audio has no aligned codec/mel frames")
            # Exactly the native 24 kHz frontend alignment: two mel frames per token.
            payload.update(speech_token=codes[:, :n], speech_feat=mel[:, : n * 2])
            fbank = torchaudio.compliance.kaldi.fbank(
                w16, num_mel_bins=80, dither=0, sample_frequency=16000
            )
            fbank -= fbank.mean(dim=0, keepdim=True)
            speaker = self.session("campplus.onnx")
            embedding = speaker.run(None, {speaker.get_inputs()[0].name: fbank[None].numpy()})[0]
            embedding = torch.tensor(embedding, dtype=torch.float32).reshape(1, -1)
            if (
                embedding.shape[1] != self.native["flow"]["spk_embed_dim"]
                or not embedding.isfinite().all()
            ):
                raise OmniFactoryError("E_SPEAKER", "invalid native speaker embedding")
            payload["embedding"] = embedding
        if payload["speech_token"].shape[1] > self.config.max_codec_tokens:
            raise OmniFactoryError("E_CODEC_LENGTH", "split recordings to fit max_codec_tokens")
        return payload, size + extra


def pack_sequence(text, instruction, speech, *, vocabulary_size, bistream=False):
    """Native CosyVoice3 unistream/5:15 layout with already-aligned CE labels.

    The layout is chosen during preparation so its explicit supervision measure
    stays stable under accumulation, caching and exact checkpoint resume.
    """
    sos, eos, task, fill = (vocabulary_size + i for i in range(4))
    text, instruction, speech = (
        text.flatten().tolist(),
        instruction.flatten().tolist(),
        speech.flatten().tolist(),
    )
    ids, is_text, labels = [sos], [False], [-1]
    ids += instruction
    is_text += [True] * len(instruction)
    labels += [-1] * len(instruction)
    if bistream and len(speech) / len(text) > 3:
        for j in range((len(text) + 5) // 5):
            t = text[j * 5 : (j + 1) * 5]
            if len(t) == 5:
                s = speech[j * 15 : (j + 1) * 15]
                if len(s) != 15:
                    raise OmniFactoryError("E_NATIVE_LAYOUT", "incomplete bistream speech block")
                ids += t + s
                is_text += [True] * len(t) + [False] * len(s)
                labels += [-1] * 4 + s + [fill]
            else:
                s = speech[j * 15 :]
                ids += t + [task] + s
                is_text += [True] * len(t) + [False] * (1 + len(s))
                labels += [-1] * len(t) + s + [eos]
    else:
        ids += text + [task] + speech
        is_text += [True] * len(text) + [False] * (1 + len(speech))
        labels += [-1] * len(text) + speech + [eos]
    if len(ids) != len(labels):
        raise OmniFactoryError("E_NATIVE_LAYOUT", "native inputs and CE labels differ in length")
    return dict(
        input_ids=torch.tensor([ids]),
        text_mask=torch.tensor([is_text]),
        labels=torch.tensor([labels]),
    )
