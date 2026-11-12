import base64
import io
import json
import math
import shutil
import struct
import subprocess
from types import SimpleNamespace

import pytest

from omnifactory.adapters.sglang_omni import (
    DECODER_FILES,
    build_chat_request,
    probe_omni,
    validate_chat_response,
    validate_omni_export,
)
from omnifactory.data.schema import CanonicalSample, LanguageCondition
from omnifactory.errors import OmniFactoryError


def native(root, adapter, *, speech=True):
    root.mkdir(exist_ok=True)
    qwen = adapter == "qwen3_omni"
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_omni_moe" if qwen else "minicpmo", "init_tts": speech})
    )
    keys = ["thinker.weight"] if qwen else ["llm.weight"]
    if speech:
        keys += ["talker.weight", "code2wav.weight"] if qwen else ["tts.weight"]
    header = json.dumps(
        {
            k: dict(dtype="F32", shape=[1], data_offsets=[4 * i, 4 * i + 4])
            for i, k in enumerate(keys)
        }
    ).encode()
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00" * (4 * len(keys))
    )
    if not qwen:
        (root / "assets/token2wav").mkdir(parents=True)
        for name in DECODER_FILES:
            (root / "assets/token2wav" / name).write_bytes(b"asset")


def prompt(raw_sample):
    raw_sample["messages"][0]["content"] = [
        {"id": "question", "type": "text", "text": "Say hello briefly."}
    ]
    return CanonicalSample.model_validate(raw_sample)


@pytest.mark.parametrize("adapter", ["qwen3_omni", "minicpm_o_4_5"])
def test_native_export_needs_speech_stages_even_for_merged(tmp_path, adapter):
    native(tmp_path, adapter, speech=False)
    with pytest.raises(OmniFactoryError, match="E_SERVE_EXPORT"):
        validate_omni_export(tmp_path, adapter)


def test_minicpm_requires_every_decoder_asset(tmp_path):
    native(tmp_path, "minicpm_o_4_5")
    validate_omni_export(tmp_path, "minicpm_o_4_5")
    (tmp_path / "assets/token2wav/flow.pt").unlink()
    with pytest.raises(OmniFactoryError, match="flow.pt"):
        validate_omni_export(tmp_path, "minicpm_o_4_5")


@pytest.mark.parametrize("adapter", ["qwen3_omni", "minicpm_o_4_5"])
def test_request_never_reads_targets_and_routes_native_speech_budget(raw_sample, tmp_path, adapter):
    sample = prompt(raw_sample)
    body = build_chat_request(sample, "p0", tmp_path, adapter, "served", "audio", 512)
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Say hello briefly."}]}
    ]
    assert body["modalities"] == ["text", "audio"]
    assert body["talker_max_new_tokens"] == 512
    assert "answer" not in json.dumps(body)
    text = build_chat_request(sample, "p0", tmp_path, adapter, "served", "text", 512)
    assert text["modalities"] == ["text"] and "audio" not in text


@pytest.mark.parametrize("adapter", ["qwen3_omni", "minicpm_o_4_5"])
@pytest.mark.parametrize("audio_mode", ["include", "exclude"])
def test_video_soundtrack_policy_and_inline_bytes(raw_sample, tmp_path, adapter, audio_mode):
    raw_sample["assets"]["clip"] = {"uri": "clip.mp4", "modality": "video"}
    raw_sample["messages"][0]["content"] = [
        {"id": "clip", "type": "video", "media": {"asset_id": "clip"}, "audio_mode": audio_mode}
    ]
    (tmp_path / "clip.mp4").write_bytes(b"video")
    sample = CanonicalSample.model_validate(raw_sample)
    body = build_chat_request(sample, "p0", tmp_path, adapter, "served", "text", 512)
    assert body["use_audio_in_video"] == (audio_mode == "include")
    video = body["messages"][0]["content"][0]["video_url"]
    assert video["url"] == "data:video/mp4;base64,dmlkZW8="
    if adapter == "minicpm_o_4_5":
        assert video["use_audio"] == (audio_mode == "include")


def test_request_rejects_timeline(raw_sample, tmp_path):
    raw_sample["messages"][0]["content"][0]["placement"] = {
        "timeline_id": "timeline",
        "offset_us": 0,
    }
    with pytest.raises(OmniFactoryError, match="E_SERVE_TIMELINE"):
        build_chat_request(
            CanonicalSample.model_validate(raw_sample),
            "p0",
            tmp_path,
            "qwen3_omni",
            "served",
            "text",
            512,
        )


@pytest.mark.parametrize("truncated_stage", ["video", "audio"])
def test_crop_rejects_successful_but_partial_ffmpeg_output(
    tmp_path, cropped_video_source, monkeypatch, truncated_stage
):
    from omnifactory.adapters import video
    from omnifactory.adapters.sglang_omni import _crop_media
    from omnifactory.data.schema import MediaSlice

    path, _, _ = cropped_video_source
    original = video.run_media

    def truncate(args):
        args = list(args)
        if args[0] == "ffmpeg" and "-vf" in args:
            if truncated_stage == "video":
                index = args.index("-vf") + 1
                args[index] = args[index].replace("end=0.677777", "end=0.377777")
            else:
                index = args.index("-af") + 1
                args[index] = args[index].replace(
                    "asetpts=PTS-STARTPTS", "atrim=end_sample=50,asetpts=PTS-STARTPTS"
                )
        return original(args)

    monkeypatch.setattr(video, "run_media", truncate)
    with pytest.raises(
        OmniFactoryError, match="E_VIDEO_FRAMES" if truncated_stage == "video" else "E_VIDEO_AUDIO"
    ):
        _crop_media(
            path,
            MediaSlice(asset_id="clip", start_us=155555, end_us=677777),
            "video",
            include_audio=True,
            max_bytes=16 * 1024 * 1024,
        )


def test_crop_rejects_large_decoded_dimensions_before_decode(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from omnifactory.adapters import video
    from omnifactory.adapters.sglang_omni import _crop_media
    from omnifactory.data.schema import MediaSlice

    calls = []

    def oversized(args):
        calls.append(args)
        return json.dumps(
            {
                "streams": [
                    {"codec_type": "video", "width": 8192, "height": 8192, "avg_frame_rate": "30/1"}
                ],
                "format": {"duration": "2"},
            }
        ).encode()

    monkeypatch.setattr(video, "run_media", oversized)
    with pytest.raises(OmniFactoryError, match="E_DECODE_BUDGET"):
        _crop_media(
            tmp_path / "clip.mp4",
            MediaSlice(asset_id="clip", start_us=155555, end_us=677777),
            "video",
            max_bytes=16 * 1024 * 1024,
        )
    assert len(calls) == 1 and calls[0][0] == "ffprobe"


def test_request_rejects_unsupported_conditions(raw_sample, tmp_path):
    sample = prompt(raw_sample)
    sample = sample.model_copy(
        update={
            "predictions": (
                sample.predictions[0].model_copy(
                    update={"conditions": (LanguageCondition(kind="language", value="en"),)}
                ),
            )
        }
    )
    with pytest.raises(OmniFactoryError, match="E_SERVE_CONDITION"):
        build_chat_request(sample, "p0", tmp_path, "qwen3_omni", "served", "audio", 512)


def response(*, reason="stop", audio=None, text="Hello."):
    message = {"content": text}
    if audio is not None:
        message["audio"] = {"data": base64.b64encode(audio).decode(), "transcript": "Hello."}
    return {"choices": [{"finish_reason": reason, "message": message}]}


@pytest.mark.parametrize("reason", ["length", None, "abort"])
def test_response_rejects_truncation_even_with_valid_audio(reason):
    with pytest.raises(OmniFactoryError, match="E_SERVE_INCOMPLETE"):
        validate_chat_response(response(reason=reason, audio=b"wave"), want_audio=True)


def test_response_requires_requested_output_and_preserves_transcript():
    with pytest.raises(OmniFactoryError, match="E_SERVE_AUDIO"):
        validate_chat_response(response(), want_audio=True)
    with pytest.raises(OmniFactoryError, match="E_SERVE_TEXT"):
        validate_chat_response(response(text=""), want_audio=False)
    parsed = validate_chat_response(response(audio=b"wave", text=None), want_audio=True)
    assert parsed == {"text": "Hello.", "audio": [b"wave"], "finish_reasons": ["stop"]}


@pytest.mark.parametrize("adapter", ["qwen3_omni", "minicpm_o_4_5"])
@pytest.mark.parametrize("max_audio_units", [512, 2])
def test_probe_checks_both_modes_and_saves_complete_wav(
    raw_sample, tmp_path, monkeypatch, adapter, max_audio_units
):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from omnifactory.adapters import sglang_omni

    export = tmp_path / "export"
    native(export, adapter)
    (export / "omnifactory_export.json").write_text(
        json.dumps({"inference_config": {"task": "speech_sft"}})
    )
    monkeypatch.setattr(
        sglang_omni, "verify_export", lambda _: {"adapter": adapter, "artifact_id": "artifact"}
    )
    dataset = tmp_path / "samples.jsonl"
    if adapter == "minicpm_o_4_5":
        raw_sample["assets"]["reference"] = {"uri": "reference.wav", "modality": "audio"}
        raw_sample["predictions"][0]["conditions"] = [
            {"kind": "reference_audio", "media": {"asset_id": "reference"}}
        ]
        (tmp_path / "reference.wav").write_bytes(b"reference")
    dataset.write_text(prompt(raw_sample).model_dump_json() + "\n")
    buffer = io.BytesIO()
    sf.write(buffer, np.sin(np.arange(2400) * 0.2), 24000, format="WAV")
    sent = []

    def http(url, body):
        sent.append(body)
        return (
            200,
            json.dumps(
                response(audio=buffer.getvalue() if "audio" in body["modalities"] else None)
            ).encode(),
            {},
        )

    monkeypatch.setattr("omnifactory.serving.http_request", http)
    if max_audio_units == 2:
        with pytest.raises(OmniFactoryError, match="E_SERVE_INCOMPLETE.*codec budget"):
            probe_omni(
                export,
                dataset,
                tmp_path / "result",
                "http://localhost:8000",
                "served",
                max_audio_units=max_audio_units,
            )
        return
    report = probe_omni(
        export,
        dataset,
        tmp_path / "result",
        "http://localhost:8000",
        "served",
        max_audio_units=max_audio_units,
    )
    assert [p["modalities"] for p in sent] == [["text"], ["text", "audio"]]
    if adapter == "minicpm_o_4_5":
        assert "ref_audio" not in json.dumps(sent[0])
        assert sent[1]["audio"]["ref_audio"] == "data:audio/wav;base64,cmVmZXJlbmNl"
    assert report["requests"] == 2 and report["quality_verified"] is False
    assert report["results"][1]["audio"][0]["audio_frames"] == 2400
    assert (tmp_path / "result/request-00001-0.wav").read_bytes() == buffer.getvalue()


@pytest.mark.parametrize("adapter", ["qwen3_omni", "minicpm_o_4_5"])
def test_sglang_export_assembles_complete_native_checkpoint(tmp_path, monkeypatch, adapter):
    pytest.importorskip("torch")
    from omnifactory.training import export

    model = SimpleNamespace(config=SimpleNamespace(use_cache=False))
    config = SimpleNamespace(
        adapter=adapter,
        task="understanding_sft",
        model=str(tmp_path),
        speech_assets=None,
        duplex_schedule=None,
        model_dump=lambda **_: {"adapter": adapter},
    )
    monkeypatch.setattr(
        export,
        "load_checkpoint",
        lambda *_, **__: (model, None, config, {"source_fingerprint": "source"}),
    )
    monkeypatch.setattr(export, "merge_lora", lambda m: m)
    calls = []

    def assemble(base, m, root, *, adapter):
        calls.append((base, m, adapter))
        native(root, adapter)
        return {"tensors": 3}

    monkeypatch.setattr(export, "assemble_native", assemble)
    result = export.export_builtin(tmp_path / "checkpoint", tmp_path / "out", format="sglang-omni")
    assert calls[0][2] == adapter and result["format"] == "sglang-omni"
    assert result["sglang_deployment_verified"] is False


def test_native_export_does_not_accept_unsupported_adapter(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from omnifactory.training import export

    monkeypatch.setattr(
        export,
        "load_checkpoint",
        lambda *_, **__: (None, None, SimpleNamespace(adapter="qwen2_5_omni"), {}),
    )
    with pytest.raises(OmniFactoryError, match="E_EXPORT_FORMAT"):
        export.export_builtin(tmp_path / "checkpoint", tmp_path / "out", format="sglang-omni")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("end_us", [750000, None])
def test_audio_crop_preserves_selected_samples_and_channels(raw_sample, tmp_path, end_us):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    wave = np.stack([np.linspace(-0.8, 0.8, 16000), np.sin(np.arange(16000) * 0.1)], axis=1).astype(
        "float32"
    )
    sf.write(tmp_path / "question.wav", wave, 16000, subtype="FLOAT")
    raw_sample["assets"]["question"]["uri"] = "question.wav"
    raw_sample["messages"][0]["content"][0]["media"].update(start_us=250000, end_us=end_us)
    payload = build_chat_request(
        CanonicalSample.model_validate(raw_sample),
        "p0",
        tmp_path,
        "minicpm_o_4_5",
        "served",
        "text",
        512,
    )
    uri = payload["messages"][0]["content"][0]["audio_url"]["url"]
    assert uri.startswith("data:audio/wav;base64,")
    cropped, rate = sf.read(
        io.BytesIO(base64.b64decode(uri.split(",", 1)[1])), dtype="float32", always_2d=True
    )
    assert rate == 16000
    np.testing.assert_array_equal(cropped, wave[4000 : 12000 if end_us else None])
    np.testing.assert_array_equal(
        sf.read(tmp_path / "question.wav", dtype="float32", always_2d=True)[0], wave
    )


@pytest.fixture(params=[24000, 44100])
def cropped_video_source(tmp_path, request):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required for real crop regression")
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    rate = request.param
    frames = np.stack([np.full((16, 16, 3), i * 5, dtype="uint8") for i in range(40)])
    (tmp_path / "frames.rgb").write_bytes(frames.tobytes())
    wave = np.stack(
        [np.linspace(-0.8, 0.8, 4 * rate), np.sin(np.arange(4 * rate) * 0.07)], axis=1
    ).astype("float32")
    sf.write(tmp_path / "source.wav", wave, rate, subtype="FLOAT")
    path = tmp_path / "source.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "16x16",
            "-framerate",
            "10",
            "-i",
            str(tmp_path / "frames.rgb"),
            "-i",
            str(tmp_path / "source.wav"),
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-g",
            "48",
            "-pix_fmt",
            "yuv444p",
            "-c:a",
            "pcm_f32le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path, wave, rate


@pytest.mark.parametrize("adapter", ["qwen3_omni", "minicpm_o_4_5"])
@pytest.mark.parametrize("audio_mode", ["include", "exclude"])
@pytest.mark.parametrize("start_us,end_us", [(700000, 2200000), (155555, 677777)])
def test_video_crop_preserves_nonzero_interval_and_soundtrack(
    raw_sample, tmp_path, cropped_video_source, adapter, audio_mode, start_us, end_us
):
    np = pytest.importorskip("numpy")
    path, wave, rate = cropped_video_source
    raw_sample["assets"]["clip"] = {"uri": path.name, "modality": "video"}
    raw_sample["messages"][0]["content"] = [
        {
            "id": "clip",
            "type": "video",
            "media": {"asset_id": "clip", "start_us": start_us, "end_us": end_us},
            "audio_mode": audio_mode,
        }
    ]
    payload = build_chat_request(
        CanonicalSample.model_validate(raw_sample), "p0", tmp_path, adapter, "served", "text", 512
    )
    uri = payload["messages"][0]["content"][0]["video_url"]["url"]
    assert uri.startswith("data:video/quicktime;base64,")
    materialized = tmp_path / "materialized.mov"
    materialized.write_bytes(base64.b64decode(uri.split(",", 1)[1]))

    def run(args):
        return subprocess.run(args, check=True, capture_output=True).stdout

    info = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(materialized),
            ]
        )
    )
    assert float(info["format"]["duration"]) <= (end_us - start_us) / 1e6 + 0.1
    first, stop = math.ceil(start_us / 1e5), math.ceil(end_us / 1e5)
    timestamps = json.loads(
        run(
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
                str(materialized),
            ]
        )
    )["frames"]
    assert [float(f["best_effort_timestamp_time"]) for f in timestamps] == pytest.approx(
        [i / 10 - start_us / 1e6 for i in range(first, stop)], abs=2e-6
    )
    for source, selected in [(path, slice(first, stop)), (materialized, slice(None))]:
        pixels = np.frombuffer(
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-fps_mode",
                    "passthrough",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
            ),
            dtype="uint8",
        ).reshape(-1, 16, 16, 3)[selected]
        if source == path:
            expected = pixels
        else:
            np.testing.assert_array_equal(pixels, expected)
    assert any(s["codec_type"] == "audio" for s in info["streams"]) == (audio_mode == "include")
    if audio_mode == "include":
        decoded = np.frombuffer(
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(materialized),
                    "-map",
                    "0:a:0",
                    "-f",
                    "f32le",
                    "pipe:1",
                ]
            ),
            dtype="<f4",
        ).reshape(-1, 2)
        np.testing.assert_array_equal(
            decoded, wave[round(start_us * rate / 1e6) : round(end_us * rate / 1e6)]
        )
    assert "answer" not in json.dumps(payload)


@pytest.mark.parametrize(
    "start_us,end_us,error",
    [(750000, 5000000, "E_VIDEO_SLICE"), (111111, 155555, "E_VIDEO_FRAMES")],
)
def test_video_crop_rejects_out_of_bounds(
    raw_sample, tmp_path, cropped_video_source, start_us, end_us, error
):
    path, _, _ = cropped_video_source
    raw_sample["assets"]["clip"] = {"uri": path.name, "modality": "video"}
    raw_sample["messages"][0]["content"] = [
        {
            "id": "clip",
            "type": "video",
            "media": {"asset_id": "clip", "start_us": start_us, "end_us": end_us},
            "audio_mode": "include",
        }
    ]
    with pytest.raises(OmniFactoryError, match=error):
        build_chat_request(
            CanonicalSample.model_validate(raw_sample),
            "p0",
            tmp_path,
            "qwen3_omni",
            "served",
            "text",
            512,
        )
