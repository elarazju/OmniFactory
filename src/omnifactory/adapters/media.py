"""Shared bounded local media loading; chat formatting belongs to each adapter."""

import numpy as np
import soundfile as sf
from PIL import Image

from omnifactory.data.schema import inference_view
from omnifactory.errors import OmniFactoryError


class MediaInputs:
    def _messages(self, sample, prediction_id):
        view = inference_view(sample, prediction_id, requested_outputs=("text",))
        if view.conditions:
            raise OmniFactoryError(
                "E_RUNTIME_CONDITION", "understanding SFT does not support generation conditions"
            )
        messages, images, audio, decoded = [], [], [], 0
        modalities = set()
        self.last_media_metadata = []
        self.last_videos = []
        self.last_video_audio_modes = []
        for message in view.messages:
            content = []
            for block in message.content:
                modalities.add(block.type)
                if block.type == "text":
                    # Literal control tokens could create ambiguous supervision boundaries.
                    if any(
                        token in block.text
                        for token in self.processor.tokenizer.all_special_tokens
                        + list(getattr(self, "reserved_content_tokens", ()))
                    ):
                        raise OmniFactoryError(
                            "E_CONTROL_TOKEN", "literal special tokens in content are unsupported"
                        )
                    content.append({"type": "text", "text": block.text})
                    continue
                if (
                    block.type == "video"
                    and hasattr(self, "video_options")
                    and block.placement is None
                ):
                    from omnifactory.adapters.video import decode_video

                    path = self.resolver.resolve(view.assets[block.media.asset_id]).path
                    frames, clips, size, timeline = decode_video(
                        path,
                        block,
                        **self.video_options,
                        max_pixels=self.max_image_pixels,
                        max_bytes=self.max_decoded_bytes - decoded,
                    )
                    if any(len(clip) > self.max_audio_seconds * 16000 for clip in clips):
                        raise OmniFactoryError(
                            "E_DECODE_BUDGET",
                            "video audio segment exceeds processor duration budget",
                        )
                    decoded += size
                    if clips:
                        modalities.add("audio")
                    if getattr(self, "native_video", False):
                        self.last_videos.append(np.stack([np.asarray(frame) for frame in frames]))
                        self.last_video_audio_modes.append(bool(clips))
                        content.append({"type": "video", "video": path})
                        if clips:
                            # Remove per-frame minimum padding before joining the native soundtrack.
                            waveform = np.concatenate(
                                [
                                    clip[: round((unit["end_us"] - unit["start_us"]) * 0.016)]
                                    for clip, unit in zip(clips, timeline)
                                ]
                            )
                            if len(waveform) > self.max_audio_seconds * 16000:
                                raise OmniFactoryError(
                                    "E_DECODE_BUDGET",
                                    "video soundtrack exceeds processor duration budget",
                                )
                            audio.append(waveform)
                    else:
                        images.extend(frames)
                        audio.extend(clips)
                        for i, frame in enumerate(frames):
                            content.append({"type": "image", "image": frame})
                            if clips:
                                content.append({"type": "audio", "audio": path})
                    self.last_media_metadata.append(
                        {
                            "block_id": block.id,
                            "audio_mode": block.audio_mode,
                            "frames": timeline,
                            "fps": self.video_options["fps"],
                        }
                    )
                    continue
                if block.type == "video" or getattr(block, "placement", None) is not None:
                    raise OmniFactoryError(
                        "E_RUNTIME_MODALITY",
                        "video and synchronized timeline execution are not implemented",
                    )
                asset_id = block.asset_id if block.type == "image" else block.media.asset_id
                asset = view.assets[asset_id]
                path = self.resolver.resolve(asset).path
                if block.type == "image":
                    with Image.open(path) as image:
                        pixels = image.width * image.height
                        if (
                            pixels > self.max_image_pixels
                            or decoded + pixels * 3 > self.max_decoded_bytes
                        ):
                            raise OmniFactoryError(
                                "E_DECODE_BUDGET", "image exceeds decoded pixel/byte budget"
                            )
                        images.append(image.convert("RGB"))
                        decoded += pixels * 3
                    content.append({"type": "image", "image": path})
                else:
                    info = sf.info(path)
                    if asset.sample_rate_hz and asset.sample_rate_hz != info.samplerate:
                        raise OmniFactoryError(
                            "E_AUDIO_METADATA", "declared sampling rate differs from file"
                        )
                    start = round(block.media.start_us * info.samplerate / 1e6)
                    end = (
                        round(block.media.end_us * info.samplerate / 1e6)
                        if block.media.end_us
                        else info.frames
                    )
                    if not 0 <= start < end <= info.frames:
                        raise OmniFactoryError(
                            "E_AUDIO_SLICE", "slice exceeds decoded audio duration"
                        )
                    seconds = (end - start) / info.samplerate
                    # Include source channels and resampled destination before allocating either.
                    required = (end - start) * info.channels * 4 + int(seconds * 16000 + 1) * 4
                    if (
                        seconds > self.max_audio_seconds
                        or decoded + required > self.max_decoded_bytes
                    ):
                        raise OmniFactoryError(
                            "E_DECODE_BUDGET", "audio exceeds duration/decoded byte budget"
                        )
                    waveform, rate = sf.read(
                        path, start=start, stop=end, dtype="float32", always_2d=True
                    )
                    waveform = waveform.mean(axis=1)
                    if rate != 16000:
                        import librosa

                        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=16000)
                    if not np.isfinite(waveform).all() or len(waveform) < 640:
                        raise OmniFactoryError("E_AUDIO", "audio must be finite and at least 40 ms")
                    audio.append(waveform)
                    decoded += required
                    content.append({"type": "audio", "audio": path})
            messages.append({"role": message.role, "content": content})
        return messages, images, audio, decoded, tuple(sorted(modalities))
