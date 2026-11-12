"""Native Qwen3-Omni/MiniCPM-o chat transport for the pinned SGLang-Omni API."""

import base64
import binascii
import io
import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path

from omnifactory.artifacts import sha256, tensor_header, verify_export
from omnifactory.data.io import iter_samples
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view
from omnifactory.errors import OmniFactoryError

ADAPTERS = ("qwen3_omni", "minicpm_o_4_5")
DECODER_FILES = (
    "speech_tokenizer_v2_25hz.onnx",
    "campplus.onnx",
    "flow.pt",
    "flow.yaml",
    "hift.pt",
)
MIME_TYPES = {
    "image": {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    },
    "audio": {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"},
    "video": {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"},
}


def _crop_media(path, media, modality, *, include_audio=False, max_bytes):
    """Materialize the selected source interval without changing the source file."""
    if modality == "audio":
        import numpy as np
        import soundfile as sf

        info = sf.info(path)
        start = round(media.start_us * info.samplerate / 1e6)
        end = (
            round(media.end_us * info.samplerate / 1e6) if media.end_us is not None else info.frames
        )
        if not 0 <= start < end <= info.frames:
            raise OmniFactoryError("E_AUDIO_SLICE", "slice exceeds decoded audio duration")
        if (end - start) * info.channels * 4 + 4096 > max_bytes:
            raise OmniFactoryError("E_SERVE_MEDIA", "cropped audio exceeds inline byte budget")
        wave, rate = sf.read(path, start=start, stop=end, dtype="float32", always_2d=True)
        if len(wave) != end - start or not np.isfinite(wave).all():
            raise OmniFactoryError("E_SERVE_MEDIA", "invalid cropped audio")
        output = io.BytesIO()
        sf.write(output, wave, rate, format="WAV", subtype="FLOAT")
        return "audio/wav", output.getvalue()

    from omnifactory.adapters.video import run_media

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise OmniFactoryError("E_VIDEO_INSTALL", "install FFmpeg and ffprobe for video crops")
    info = json.loads(
        run_media(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        )
    )
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if video is None:
        raise OmniFactoryError("E_VIDEO", "asset has no video stream")
    duration = float(info.get("format", {}).get("duration", video.get("duration", 0)))
    start = media.start_us / 1e6
    end = media.end_us / 1e6 if media.end_us is not None else duration
    if not math.isfinite(duration) or not 0 <= start < end <= duration + 1e-4:
        raise OmniFactoryError("E_VIDEO_SLICE", "slice exceeds video duration")
    if include_audio and audio is None:
        raise OmniFactoryError("E_VIDEO_AUDIO", "audio_mode=include requires an audio stream")
    from fractions import Fraction

    # Probe complete source frame timestamps only within a bounded decode budget.
    # Large/long inputs must first be materialized as smaller source assets.
    width, height = int(video["width"]), int(video["height"])
    fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
    if (
        not 0 < width * height <= 4 * 1024 * 1024
        or not 0 < fps <= 120
        or duration > 60
        or math.ceil(duration * fps) * width * height * 3 > 512 * 1024 * 1024
    ):
        raise OmniFactoryError("E_DECODE_BUDGET", "video crop exceeds pixel/duration/decode budget")
    origin = float(info.get("format", {}).get("start_time", 0))
    absolute_start, absolute_end = origin + start, origin + end

    def frame_times(source):
        frames = json.loads(
            run_media(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_frames",
                    "-show_entries",
                    "frame=best_effort_timestamp_time",
                    "-of",
                    "json",
                    str(source),
                ]
            )
        )["frames"]
        return [float(frame["best_effort_timestamp_time"]) for frame in frames]

    original_times = frame_times(path)
    expected_times = [
        pts - absolute_start for pts in original_times if absolute_start <= pts < absolute_end
    ]
    if not expected_times:
        raise OmniFactoryError("E_VIDEO_FRAMES", "selected interval contains no video frame")
    if len(original_times) * width * height * 3 > 512 * 1024 * 1024:
        raise OmniFactoryError("E_DECODE_BUDGET", "source video exceeds decoded frame budget")
    if include_audio:
        rate, channels = int(audio["sample_rate"]), int(audio["channels"])
        audio_origin = float(audio.get("start_time", origin))
        first = round((absolute_start - audio_origin) * rate)
        stop = round((absolute_end - audio_origin) * rate)
        samples = stop - first
        audio_duration = float(audio.get("duration", duration))
        if not 0 <= first < stop <= round(audio_duration * rate):
            raise OmniFactoryError("E_VIDEO_AUDIO", "audio stream does not cover selected interval")
        if samples * channels * 4 + 4096 > max_bytes:
            raise OmniFactoryError("E_SERVE_MEDIA", "cropped soundtrack exceeds inline byte budget")
    with tempfile.TemporaryDirectory(prefix="omnifactory-video-") as temporary:
        output = Path(temporary) / "clip.mov"
        # Select on ORIGINAL timestamps. Input -ss plus duration-based trimming
        # can admit a frame beyond end after FFmpeg rounds a non-frame-aligned seek.
        # Keep the first selected frame's offset relative to the audio sample grid.
        args = [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-copyts",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map_metadata",
            "-1",
            "-vf",
            f"trim=start={absolute_start:.6f}:end={absolute_end:.6f},settb=1/1000000,setpts=PTS-{round(absolute_start * 1e6)}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-fps_mode",
            "passthrough",
            "-enc_time_base:v",
            "1:1000000",
            "-video_track_timescale",
            "1000000",
            "-movie_timescale",
            "1000000",
            "-avoid_negative_ts",
            "disabled",
        ]
        if include_audio:
            # PCM in MOV avoids AAC encoder padding. No global -t: both filters
            # independently terminate at their exact source frame/sample boundary.
            args += [
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_f32le",
                "-af",
                f"atrim=start_sample={first}:end_sample={stop},asetpts=PTS-STARTPTS",
            ]
        else:
            args += ["-an"]
        run_media(args + ["-fs", str(max_bytes + 1), str(output)])
        if not output.is_file() or not 0 < output.stat().st_size <= max_bytes:
            raise OmniFactoryError("E_SERVE_MEDIA", "cropped video exceeds inline byte budget")
        # -fs may stop successfully at a byte cap. Never accept a partial container.
        actual_times = frame_times(output)
        if len(actual_times) != len(expected_times) or any(
            abs(actual - expected) > 2e-6 for actual, expected in zip(actual_times, expected_times)
        ):
            raise OmniFactoryError(
                "E_VIDEO_FRAMES", "materialized video lost frames or source timing"
            )
        if include_audio:
            decoded = run_media(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(output),
                    "-map",
                    "0:a:0",
                    "-f",
                    "f32le",
                    "pipe:1",
                ]
            )
            if len(decoded) != samples * channels * 4:
                raise OmniFactoryError("E_VIDEO_AUDIO", "materialized soundtrack is incomplete")
        return "video/quicktime", output.read_bytes()


def validate_omni_export(export, adapter):
    """Reject thinker-only checkpoints even when their export format says merged.

    This checks native stage availability, not quality or successful GPU execution.
    Artifact sealing/verification remains the caller's responsibility.
    """
    if adapter not in ADAPTERS:
        raise OmniFactoryError("E_SERVE_BACKEND", f"unsupported SGLang-Omni adapter: {adapter}")
    root = Path(export)
    config = json.loads((root / "config.json").read_text())
    expected = "qwen3_omni_moe" if adapter == "qwen3_omni" else "minicpmo"
    if config.get("model_type") != expected:
        raise OmniFactoryError(
            "E_SERVE_EXPORT", "complete native --format sglang-omni export required"
        )
    index = root / "model.safetensors.index.json"
    if index.is_file():
        files = set(json.loads(index.read_text())["weight_map"].values())
    else:
        files = {"model.safetensors"}
    keys = set()
    for name in files:
        if Path(name).name != name or not (root / name).is_file():
            raise OmniFactoryError("E_SERVE_EXPORT", "missing or invalid native model shard")
        keys.update(tensor_header(root / name))
    prefixes = ("thinker.", "talker.", "code2wav.") if adapter == "qwen3_omni" else ("llm.", "tts.")
    if any(not any(key.startswith(prefix) for key in keys) for prefix in prefixes):
        raise OmniFactoryError("E_SERVE_EXPORT", f"native export must preserve stages {prefixes}")
    if adapter == "minicpm_o_4_5":
        if config.get("init_tts") is False:
            raise OmniFactoryError("E_SERVE_EXPORT", "MiniCPM native export has init_tts disabled")
        for name in DECODER_FILES:
            path = root / "assets/token2wav" / name
            if not path.is_file() or not path.stat().st_size:
                raise OmniFactoryError("E_SERVE_EXPORT", f"MiniCPM decoder asset missing: {name}")


def build_chat_request(sample, prediction_id, root, adapter, model, modality, max_audio_units):
    """Encode only prediction-visible history; preserve native video soundtrack policy."""
    if adapter not in ADAPTERS or modality not in ("text", "audio"):
        raise OmniFactoryError("E_SERVE_MODALITY", "unsupported SGLang-Omni adapter/output")
    if type(max_audio_units) is not int or max_audio_units <= 0:
        raise OmniFactoryError("E_CODEC_LENGTH", "max_audio_units must be positive")
    view = inference_view(
        sample, prediction_id, ("text", "audio") if modality == "audio" else ("text",)
    )
    resolver, total, source_total = LocalMediaResolver(Path(root)), 0, 0

    def media_url(asset_id, media=None, *, include_audio=False):
        nonlocal total, source_total
        asset = view.assets[asset_id]
        resolved = resolver.resolve(asset)
        source_total += resolved.size_bytes
        if source_total > 16 * 1024 * 1024:
            raise OmniFactoryError("E_SERVE_MEDIA", "inline media exceeds 16 MiB")
        path = Path(resolved.path)
        mime = MIME_TYPES[asset.modality].get(path.suffix.lower())
        if mime is None:
            raise OmniFactoryError("E_SERVE_MEDIA", f"unsupported inline {asset.modality} format")
        if media is not None and (media.start_us or media.end_us is not None):
            mime, data = _crop_media(
                path,
                media,
                asset.modality,
                include_audio=include_audio,
                max_bytes=16 * 1024 * 1024 - total,
            )
        else:
            data = path.read_bytes()
        total += len(data)
        if total > 16 * 1024 * 1024:
            raise OmniFactoryError("E_SERVE_MEDIA", "inline media exceeds 16 MiB")
        return f"data:{mime};base64," + base64.b64encode(data).decode()

    payload = dict(
        model=model, messages=[], modalities=["text"], max_tokens=128, temperature=0, stream=False
    )
    if modality == "audio":
        payload.update(
            modalities=["text", "audio"],
            audio={"format": "wav"},
            talker_max_new_tokens=max_audio_units,
        )
    video_audio_modes = set()
    for message in view.messages:
        content = []
        for block in message.content:
            if getattr(block, "placement", None) is not None:
                raise OmniFactoryError(
                    "E_SERVE_TIMELINE", "chat serving cannot preserve timeline placement"
                )
            if block.type == "text":
                content.append(dict(type="text", text=block.text))
            elif block.type == "image":
                content.append(dict(type="image_url", image_url={"url": media_url(block.asset_id)}))
            elif block.type == "audio":
                content.append(
                    dict(
                        type="audio_url",
                        audio_url={"url": media_url(block.media.asset_id, block.media)},
                    )
                )
            elif block.type == "video":
                use_audio = block.audio_mode == "include"
                video_audio_modes.add(use_audio)
                video = {
                    "url": media_url(block.media.asset_id, block.media, include_audio=use_audio)
                }
                if adapter == "minicpm_o_4_5":
                    video["use_audio"] = use_audio
                content.append(dict(type="video_url", video_url=video))
        payload["messages"].append(dict(role=message.role, content=content))
    if video_audio_modes:
        if adapter == "qwen3_omni" and len(video_audio_modes) != 1:
            raise OmniFactoryError(
                "E_SERVE_MEDIA", "Qwen3 requires one soundtrack policy per request"
            )
        payload["use_audio_in_video"] = True in video_audio_modes
    references = 0
    for condition in view.conditions:
        if (
            condition.kind == "reference_audio"
            and adapter == "minicpm_o_4_5"
            and not condition.transcript
        ):
            references += 1
            if references > 1:
                raise OmniFactoryError("E_SERVE_CONDITION", "one reference recording supported")
            # Speaker reference conditions the vocoder only, never the text decoder.
            if modality == "audio":
                payload["audio"]["ref_audio"] = media_url(condition.media.asset_id, condition.media)
        else:
            raise OmniFactoryError("E_SERVE_CONDITION", "unsupported native chat condition")
    return payload


def validate_chat_response(body, *, want_audio):
    """Return text and WAV bytes only after all choices report normal completion."""
    try:
        body = json.loads(body) if isinstance(body, (str, bytes)) else body
        choices = body["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("empty choices")
        texts, blobs, reasons = [], [], []
        for choice in choices:
            reason = choice.get("finish_reason")
            if reason != "stop":
                raise OmniFactoryError("E_SERVE_INCOMPLETE", f"expected chat stop, got {reason!r}")
            reasons.append(reason)
            message = choice["message"]
            content = message.get("content")
            choice_text, audio_items = [], []
            if isinstance(content, str):
                choice_text.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        choice_text.append(part["text"])
                    elif part.get("type") == "audio":
                        audio_items.append(part["audio"])
            if message.get("audio"):
                audio_items.append(message["audio"])
            if not any(text.strip() for text in choice_text):
                choice_text = [a["transcript"] for a in audio_items if a.get("transcript")]
            if (
                not choice_text
                or not all(isinstance(text, str) for text in choice_text)
                or not "".join(choice_text).strip()
            ):
                raise OmniFactoryError("E_SERVE_TEXT", "empty chat text/transcript")
            if bool(audio_items) != want_audio:
                raise OmniFactoryError(
                    "E_SERVE_AUDIO", "response does not match requested audio modality"
                )
            for item in audio_items:
                blob = base64.b64decode(item["data"], validate=True)
                if not blob:
                    raise OmniFactoryError("E_SERVE_AUDIO", "empty encoded audio")
                blobs.append(blob)
            texts.append("".join(choice_text))
        return {"text": "\n".join(texts), "audio": blobs, "finish_reasons": reasons}
    except OmniFactoryError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError, binascii.Error) as error:
        raise OmniFactoryError("E_SERVE_RESPONSE", "invalid native chat response") from error


def probe_omni(export, dataset, output, url, model, *, max_audio_units=128):
    """Exercise understanding and speech transport; retain explicit evidence limits.

    Upstream chat finish_reason reflects the thinker, not the talker. A returned
    WAV alone does not establish speech EOS or semantic audio quality.
    """
    import numpy as np
    import soundfile as sf

    from omnifactory.serving import http_request

    artifact = verify_export(export)
    adapter = artifact["adapter"]
    validate_omni_export(export, adapter)
    config = json.loads((Path(export) / "omnifactory_export.json").read_text())["inference_config"]
    native_config = json.loads((Path(export) / "config.json").read_text())
    codec_config = native_config.get("code2wav_config", {})
    samples_per_codec_frame = (
        math.prod(codec_config.get("upsample_rates", [8, 5, 4, 3]))
        * math.prod(codec_config.get("upsampling_ratios", [2, 2]))
        if adapter == "qwen3_omni"
        else 960
    )
    if config["task"] not in ("understanding_sft", "speech_sft"):
        raise OmniFactoryError(
            "E_SERVE_TASK", "native chat probe supports understanding/speech exports"
        )
    if (
        adapter == "qwen3_omni"
        and config["task"] == "speech_sft"
        and config.get("speech_speaker", "Ethan").casefold() != "ethan"
    ):
        # The pinned chat protocol drops speaker overrides before the Talker.
        raise OmniFactoryError(
            "E_SERVE_CONDITION", "pinned SGLang-Omni chat supports only the Ethan Qwen3 speaker"
        )
    modalities = ("text", "audio") if config["task"] == "speech_sft" else ("text",)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for sample in iter_samples(Path(dataset)):
        for prediction in sample.predictions:
            for modality in modalities:
                payload = build_chat_request(
                    sample,
                    prediction.id,
                    Path(dataset).parent,
                    adapter,
                    model,
                    modality,
                    max_audio_units,
                )
                if (
                    adapter == "minicpm_o_4_5"
                    and modality == "audio"
                    and "ref_audio" not in payload["audio"]
                    and not (Path(export) / "assets/HT_ref_audio.wav").is_file()
                ):
                    raise OmniFactoryError(
                        "E_SERVE_CONDITION",
                        "MiniCPM needs reference_audio or assets/HT_ref_audio.wav",
                    )
                name = f"request-{len(records):05d}"
                payload["request_id"] = "omnifactory-" + uuid.uuid4().hex
                request_path, response_path = (
                    output / f"{name}.request.json",
                    output / f"{name}.response",
                )
                request_path.write_text(json.dumps(payload))
                status, raw, headers = http_request(
                    url.rstrip("/") + "/v1/chat/completions", payload
                )
                response_path.write_bytes(raw)
                if status != 200:
                    raise OmniFactoryError("E_SERVE_HTTP", f"chat returned HTTP {status}")
                result = validate_chat_response(raw, want_audio=modality == "audio")
                record = dict(
                    sample_id=sample.id,
                    request_id=payload["request_id"],
                    prediction_id=prediction.id,
                    modality=modality,
                    endpoint="/v1/chat/completions",
                    status=status,
                    headers=headers,
                    request_sha256=sha256(request_path),
                    response_sha256=sha256(response_path),
                    text=result["text"],
                    finish_reasons=result["finish_reasons"],
                    finish_reason_scope="thinker",
                    audio=[],
                )
                for i, blob in enumerate(result["audio"]):
                    try:
                        info = sf.info(io.BytesIO(blob))
                        wave, rate = sf.read(io.BytesIO(blob), always_2d=True)
                    except (RuntimeError, ValueError) as error:
                        raise OmniFactoryError("E_SERVE_AUDIO", "invalid WAV response") from error
                    if (
                        info.format not in ("WAV", "WAVEX")
                        or not wave.size
                        or not np.isfinite(wave).all()
                        or np.abs(wave).max() == 0
                    ):
                        raise OmniFactoryError(
                            "E_SERVE_AUDIO", "invalid, empty or silent WAV response"
                        )
                    if rate != 24000:
                        raise OmniFactoryError("E_SERVE_AUDIO", "native Omni speech must be 24 kHz")
                    # EOS is omitted by both native decoders. Reaching the requested
                    # limit is suspect even if the Thinker finished normally. The
                    # context window can impose a smaller cap, so this is only a
                    # rejection guard and does not establish natural speech EOS.
                    if len(wave) >= max(1, max_audio_units - 1) * samples_per_codec_frame:
                        raise OmniFactoryError(
                            "E_SERVE_INCOMPLETE",
                            "speech reached its codec budget; raise max_audio_units",
                        )
                    path = output / f"{name}-{i}.wav"
                    path.write_bytes(blob)
                    record["audio"].append(
                        dict(
                            path=path.name,
                            sha256=sha256(path),
                            sample_rate=rate,
                            audio_frames=len(wave),
                            channels=wave.shape[1],
                            codec_frames=len(wave) / samples_per_codec_frame,
                            requested_codec_budget=max_audio_units,
                        )
                    )
                records.append(record)
    if not records:
        raise OmniFactoryError("E_SERVE_DATASET", "empty native chat probe dataset")
    return dict(
        status="passed",
        artifact_id=artifact["artifact_id"],
        adapter=adapter,
        requests=len(records),
        results=records,
        quality_verified=False,
        audio_completion_verified=False,
        backend="sglang-omni",
    )
