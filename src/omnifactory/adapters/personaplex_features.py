"""Frozen streaming Mimi and explicitly timed SentencePiece targets."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Annotated

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator

from omnifactory._vendor.personaplex.models.loaders import MIMI_NAME, get_mimi
from omnifactory._vendor.personaplex.utils.compile import no_compile
from omnifactory.adapters.speech import load_target_audio
from omnifactory.artifacts import sha256
from omnifactory.data.schema import Record
from omnifactory.errors import OmniFactoryError

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class TimedText(Record):
    sample_id: str
    prediction_id: str
    tokenizer_sha256: Digest
    audio_sha256: Digest
    text_sha256: Digest
    # One explicitly annotated frame for each token in the exact transcript.
    token_ids: tuple[Annotated[int, Field(ge=0)], ...]
    frames: tuple[Annotated[int, Field(ge=0)], ...]

    @model_validator(mode="after")
    def ordered(self):
        if len(self.token_ids) != len(self.frames) or not self.frames:
            raise ValueError("E_TEXT_ALIGNMENT: one frame per transcript token required")
        if any(a >= b for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("E_TEXT_ALIGNMENT: frame indices must strictly increase")
        return self


def load_alignments(path):
    if path is None:
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "training requires duplex_text_alignment")
    rows = [TimedText.model_validate(row) for row in json.loads(Path(path).read_text())]
    result = {(row.sample_id, row.prediction_id): row for row in rows}
    if len(result) != len(rows):
        raise OmniFactoryError("E_TEXT_ALIGNMENT", "duplicate sample/prediction alignment")
    return result


def aligned_text(row, text, tokenizer, tokenizer_path, audio_path, *, frames):
    if (
        row.tokenizer_sha256 != sha256(tokenizer_path)
        or row.audio_sha256 != sha256(audio_path)
        or row.text_sha256 != hashlib.sha256(text.encode()).hexdigest()
        or row.token_ids != tuple(tokenizer.encode(text))
        or row.frames[-1] >= frames
    ):
        raise OmniFactoryError(
            "E_TEXT_ALIGNMENT", "timing does not match transcript/tokenizer/audio"
        )
    result = torch.full((1, 1, frames), 3, dtype=torch.long)
    result[0, 0, list(row.frames)] = torch.tensor(row.token_ids)
    return result


def read_voice(root, name):
    if Path(name).name != name or not name.endswith(".pt"):
        raise OmniFactoryError("E_VOICE_STATE", "voice must be a bundled .pt basename")
    root = Path(root)
    path = root / "voices" / name
    if path.is_file():
        if path.stat().st_size > 8 * 1024 * 1024:
            raise OmniFactoryError("E_VOICE_STATE", "voice bundle exceeds size limit")
        data = path.read_bytes()
    else:
        with tarfile.open(root / "voices.tgz") as archive:
            matches = [m for m in archive if m.isfile() and Path(m.name).name == name]
            if len(matches) != 1 or matches[0].size > 8 * 1024 * 1024:
                raise OmniFactoryError("E_VOICE_STATE", "missing, duplicate or oversized voice")
            with archive.extractfile(matches[0]) as stream:
                data = stream.read()
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"embeddings", "cache"}:
        raise OmniFactoryError("E_VOICE_STATE", "expected native embeddings and cache")
    return state


class MimiFeatures:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self._codec = None

    @property
    def codec(self):
        if self._codec is None:
            self._codec = get_mimi(self.root / MIMI_NAME, self.config.device).requires_grad_(False)
        return self._codec

    def audio(self, resolver, asset, block):
        import soundfile as sf

        info = sf.info(resolver.resolve(asset).path)
        if info.channels != 1:
            raise OmniFactoryError("E_DUPLEX_TRACK", "provide separate mono user and agent tracks")
        wave, size = load_target_audio(
            resolver,
            asset,
            block.media,
            max_seconds=self.config.max_audio_seconds,
            max_bytes=self.config.budget.max_decoded_media_bytes,
            target_rate=24000,
        )
        if (len(wave) + 1919) // 1920 > self.config.max_codec_tokens:
            raise OmniFactoryError("E_CODEC_LENGTH", "split the recording at an explicit boundary")
        return torch.from_numpy(wave), size

    @torch.no_grad()
    def encode(self, wave):
        # Each channel starts with an independent codec state, including silence.
        frame_size = 1920
        wave = F.pad(wave, (0, (-len(wave)) % frame_size)).reshape(1, 1, -1)
        outputs = []
        with no_compile(), self.codec.streaming(1):
            for chunk in wave.split(frame_size, -1):
                outputs.append(self.codec.encode(chunk.to(self.config.device)).cpu())
        result = torch.cat(outputs, -1)
        if result.shape != (1, 8, wave.shape[-1] // frame_size):
            raise OmniFactoryError("E_CODEC_LAYOUT", "Mimi must return eight codes per 80 ms frame")
        return result.long()

    @torch.no_grad()
    def decode(self, frames):
        value = torch.tensor(frames, dtype=torch.long, device=self.config.device).T[None]
        if value.ndim != 3 or value.shape[1] != 8 or not value.shape[-1]:
            raise OmniFactoryError("E_CODEC_LAYOUT", "expected frame-major eight-codebook audio")
        outputs = []
        with no_compile(), self.codec.streaming(1):
            for frame in value.split(1, -1):
                outputs.append(self.codec.decode(frame).float().cpu())
        wave = torch.cat(outputs, -1).reshape(-1)
        if not torch.isfinite(wave).all() or not wave.abs().max():
            raise OmniFactoryError("E_AUDIO_DECODE", "empty or nonfinite Mimi waveform")
        return wave.numpy(), 24000
