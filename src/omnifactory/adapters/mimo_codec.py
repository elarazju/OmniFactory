"""Frozen MiMo codec, kept outside the trainable model and loaded only when needed."""

import numpy as np
import torch

from omnifactory.errors import OmniFactoryError


class MiMoCodec:
    def __init__(self, path, channels=8):
        from omnifactory._vendor.mimo_audio.codec import MiMoAudioTokenizer

        self.model = MiMoAudioTokenizer.from_pretrained(path).float().eval()
        self.model.requires_grad_(False)
        self.config = self.model.config
        self.channels = channels

    @torch.no_grad()
    def encode(self, waveform, rate):
        import librosa

        c = self.config
        if rate != c.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=rate, target_sr=c.sampling_rate)
        from torchaudio.transforms import MelSpectrogram

        transform = MelSpectrogram(
            sample_rate=c.sampling_rate,
            n_fft=c.nfft,
            hop_length=c.hop_length,
            win_length=c.window_size,
            f_min=c.fmin,
            f_max=c.fmax,
            n_mels=c.n_mels,
            power=1.0,
            center=True,
        )
        wav = torch.from_numpy(waveform.copy()).float()
        mel = transform(wav[None]).clamp_min(1e-7).log().squeeze(0).T.contiguous()
        lengths = [3000] * (len(mel) // 3000)
        if len(mel) % 3000:
            lengths.append(len(mel) % 3000)
        codes, _ = self.model.encoder.encode(
            mel, input_lens=torch.tensor(lengths), return_codes_only=True
        )
        return codes[: self.channels].T.long().contiguous()

    @torch.no_grad()
    def __call__(self, codes):
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2 or not len(codes):
            raise OmniFactoryError("E_CODEC_SHAPE", "MiMo decoding requires [frames, codebooks]")
        sizes = self.config.codebook_size
        if len(sizes) == 1:
            sizes = sizes * self.config.num_quantizers
        if codes.shape[1] != self.channels or self.channels > self.config.num_quantizers:
            raise OmniFactoryError("E_CODEC_SHAPE", "codebook count differs from codec")
        if any(
            (codes[:, i] < 0).any() or (codes[:, i] >= size).any()
            for i, size in enumerate(sizes[: self.channels])
        ):
            raise OmniFactoryError("E_CODEC_RANGE", "invalid codebook index")
        waveform = self.model.decode(codes.T.contiguous()).float().reshape(-1).cpu().numpy()
        if not np.isfinite(waveform).all():
            raise OmniFactoryError("E_CODEC_WAVEFORM", "nonfinite decoded waveform")
        return waveform, self.config.sampling_rate
