"""Bounded offline video sampling with explicit source time and soundtrack policy."""

import json
import math
import shutil
import subprocess

import numpy as np
from PIL import Image

from omnifactory.errors import OmniFactoryError


def run_media(args):
    try:
        return subprocess.run(args, check=True, capture_output=True, timeout=120).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        raise OmniFactoryError("E_VIDEO_DECODE", str(exc)) from exc


def decode_video(path, block, *, fps, max_frames, max_seconds, max_pixels, max_bytes):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise OmniFactoryError("E_VIDEO_INSTALL", "install FFmpeg and ffprobe for video inputs")
    info = json.loads(
        run_media(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        )
    )
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise OmniFactoryError("E_VIDEO", "asset has no video stream")
    duration = float(info.get("format", {}).get("duration", video.get("duration", 0)))
    start = block.media.start_us / 1e6
    end = block.media.end_us / 1e6 if block.media.end_us is not None else duration
    width, height = int(video["width"]), int(video["height"])
    frames = math.ceil((end - start) * fps)
    audio = block.audio_mode == "include"
    required = (2 * frames + 1) * width * height * 3 + (
        math.ceil((end - start) * 16000) * 8 if audio else 0
    )
    if not 0 <= start < end <= duration + 1e-4:
        raise OmniFactoryError("E_VIDEO_SLICE", "slice exceeds video duration")
    if (
        end - start > max_seconds
        or frames > max_frames
        or width * height > max_pixels
        or required > max_bytes
    ):
        raise OmniFactoryError(
            "E_DECODE_BUDGET", "video exceeds duration/frame/pixel/decoded byte budget"
        )
    if audio and not any(s["codec_type"] == "audio" for s in info["streams"]):
        raise OmniFactoryError("E_VIDEO_AUDIO", "audio_mode=include requires an audio stream")
    common = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-noautorotate",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(path),
    ]
    # A fixed sampling grid defines each audiovisual unit. Preserve source PTS in metadata.
    raw = run_media(
        common
        + [
            "-t",
            f"{end - start:.6f}",
            "-map",
            "0:v:0",
            "-vf",
            f"fps=fps={fps}:start_time=0:round=up",
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    size = width * height * 3
    if not raw or len(raw) % size or len(raw) // size != frames:
        raise OmniFactoryError(
            "E_VIDEO_FRAMES", "decoder did not produce the requested sampling grid"
        )
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frames, height, width, 3)
    images = [Image.fromarray(x).copy() for x in pixels]
    clips = []
    if audio:
        expected = round((end - start) * 16000)
        # Apply the audio limit on the resampled sample grid. Muxer time limits
        # can round differently from this grid when the source uses AAC packets.
        audio_filter = f"aresample=16000,apad=whole_len={expected},atrim=end_sample={expected}"
        samples = run_media(
            common
            + [
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-af",
                audio_filter,
                "-f",
                "f32le",
                "pipe:1",
            ]
        )
        waveform = np.frombuffer(samples, dtype="<f4").copy()
        if len(waveform) != expected or not np.isfinite(waveform).all():
            raise OmniFactoryError(
                "E_VIDEO_AUDIO", "audio duration or samples do not match selected video"
            )
        for i in range(frames):
            clip = waveform[round(i / fps * 16000) : round(min((i + 1) / fps, end - start) * 16000)]
            if len(clip) < 1440:
                # Explicit zero-tail for native pooling; never drop the final short segment.
                clip = np.pad(clip, (0, 1440 - len(clip)))
            clips.append(clip)
    timeline = [
        {
            "start_us": round((start + i / fps) * 1e6),
            "end_us": round(min(start + (i + 1) / fps, end) * 1e6),
        }
        for i in range(frames)
    ]
    return images, clips, required, timeline
