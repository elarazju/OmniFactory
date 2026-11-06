"""Real WebSocket duplex acceptance; incoming audio stays live during playback."""

import asyncio
import base64
import json
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

from omnifactory.adapters.personaplex import input_context
from omnifactory.adapters.speech import load_target_audio
from omnifactory.artifacts import sha256, verify_export
from omnifactory.data.io import iter_samples
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view
from omnifactory.errors import OmniFactoryError


async def session_probe(url, model, voice, persona, wave, output, *, timeout=120):
    parsed = urlsplit(url)
    endpoint = urlunsplit(
        (
            "wss" if parsed.scheme == "https" else "ws",
            parsed.netloc,
            "/v1/realtime",
            urlencode(dict(duplex=1, model=model, autostart=0)),
            "",
        )
    )
    frame_count = (len(wave) + 1919) // 1920
    sent, received, audio = [], [], []
    started, first_audio, last_append = time.monotonic(), None, None
    session_id = None
    condition = asyncio.Condition()
    async with connect(
        endpoint, max_size=32 * 1024 * 1024, open_timeout=30, close_timeout=10
    ) as ws:

        async def read():
            nonlocal first_audio, session_id
            async for raw in ws:
                event = json.loads(raw)
                received.append(event)
                if event.get("type") == "session.created":
                    session = event.get("session", {})
                    session_id = (
                        session.get("id") or session.get("session_id") or event.get("session_id")
                    )
                if event.get("type") in ("response.audio.delta", "response.output_audio.delta"):
                    if event.get("sample_rate_hz") != 24000:
                        raise OmniFactoryError("E_REALTIME_AUDIO", "expected 24 kHz PCM16 events")
                    chunk = base64.b64decode(event["delta"], validate=True)
                    if len(chunk) % (1920 * 2):
                        raise OmniFactoryError("E_REALTIME_AUDIO", "incomplete 80 ms output frame")
                    audio.append(chunk)
                    first_audio = first_audio or time.monotonic()
                async with condition:
                    condition.notify_all()

        reader = asyncio.create_task(read())

        async def wait_for(predicate):
            deadline = time.monotonic() + timeout
            while True:
                if any(e.get("type") == "error" for e in received):
                    raise OmniFactoryError(
                        "E_REALTIME_SERVER", "server returned error; see response events"
                    )
                if reader.done() and not reader.cancelled() and reader.exception():
                    reader.result()
                if predicate():
                    return
                if reader.done():
                    reader.result()
                    raise OmniFactoryError(
                        "E_REALTIME_CLOSED", "server closed before acceptance completed"
                    )
                if any(e.get("type") == "error" for e in received):
                    raise OmniFactoryError(
                        "E_REALTIME_SERVER", "server returned error; see response events"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OmniFactoryError(
                        "E_REALTIME_TIMEOUT", "session event/frame coverage timed out"
                    )
                async with condition:
                    try:
                        await asyncio.wait_for(condition.wait(), min(remaining, 0.2))
                    except asyncio.TimeoutError:
                        pass

        async def send(event):
            sent.append(event)
            await ws.send(json.dumps(event))

        try:
            await send(
                dict(
                    type="session.update",
                    session=dict(
                        model=model,
                        modalities=["audio", "text"],
                        input_audio_format="pcm_f32le",
                        output_audio_format="pcm16",
                        voice=voice,
                        instructions=persona,
                        turn_detection=None,
                        extra_body=dict(auto_response=True),
                    ),
                )
            )
            await wait_for(lambda: any(e.get("type") == "session.created" for e in received))
            created = next(e for e in received if e.get("type") == "session.created")
            capabilities = created.get("session", {}).get("capabilities", {})
            if (
                capabilities.get("implementation_level") != "model_native_duplex"
                or capabilities.get("chunk_period_ms") != 80
            ):
                raise OmniFactoryError(
                    "E_REALTIME_CAPABILITY", "server did not report native 80 ms duplex"
                )
            # The pinned vLLM-Omni stage processor acoustically de-delays cb1..7
            # using the NEXT raw Talker step (_agent_codes_to_codebook_major).
            # One explicit silent drain tick completes the original recording.
            # It is counted separately and cannot establish real-input overlap.
            drain_frames = 1
            for seq in range(frame_count + drain_frames):
                frame = wave[seq * 1920 : (seq + 1) * 1920]
                frame = np.ascontiguousarray(np.pad(frame, (0, 1920 - len(frame))), dtype="<f4")
                await send(
                    dict(
                        type="input_audio_buffer.append",
                        audio=base64.b64encode(frame).decode(),
                        format="pcm_f32le",
                        sample_rate_hz=24000,
                        duration_ms=80,
                        audio_end_ms=(seq + 1) * 80,
                        is_speech=bool(np.sqrt(np.mean(frame**2)) >= 1e-4),
                    )
                )
                if seq < frame_count:
                    last_append = time.monotonic()
                await asyncio.sleep(0.08)
            await wait_for(lambda: sum(map(len, audio)) >= frame_count * 1920 * 2)
            await send(dict(type="session.close"))
            await wait_for(lambda: any(e.get("type") == "session.closed" for e in received))
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            (output / "request-events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in sent)
            )
            (output / "response-events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in received)
            )
    raw = b"".join(audio)
    pcm = np.frombuffer(raw, dtype="<i2")
    if len(pcm) != frame_count * 1920 or not np.abs(pcm.astype(np.float32)).max():
        raise OmniFactoryError("E_REALTIME_AUDIO", "invalid frame count or silent session output")
    if first_audio is None or first_audio >= last_append:
        raise OmniFactoryError(
            "E_REALTIME_OVERLAP", "no output arrived while input was still streaming"
        )
    events = [
        e
        for e in received
        if e.get("type") in ("response.audio.delta", "response.output_audio.delta")
    ]
    proof = [e.get("metadata", {}).get("vllm_omni", {}) for e in events]
    if not any(
        p.get("runtime_impl") == "scheduler_data_plane"
        and p.get("uses_model_runner_scheduler") is True
        and p.get("runner_kv_backed") is True
        for p in proof
    ):
        raise OmniFactoryError("E_REALTIME_RUNTIME", "missing scheduler/KV-backed duplex evidence")
    response_ids = {e.get("response_id") for e in events}
    if len(response_ids) != 1 or None in response_ids:
        raise OmniFactoryError(
            "E_REALTIME_RESPONSE", "continuous session must have one response identity"
        )
    sf.write(output / "output.wav", pcm, 24000, subtype="PCM_16")
    return dict(
        session_id=session_id,
        input_frames=frame_count,
        drain_frames=drain_frames,
        wire_input_frames=frame_count + drain_frames,
        last_input_seconds=last_append - started,
        output_frames=len(pcm) // 1920,
        sample_rate=24000,
        audio_chunks=len(audio),
        first_audio_seconds=first_audio - started,
        output_during_input=True,
        request_sha256=sha256(output / "request-events.jsonl"),
        response_sha256=sha256(output / "response-events.jsonl"),
        audio_sha256=sha256(output / "output.wav"),
    )


def probe_realtime(export, dataset, output, url, model, *, max_audio_units=128):
    artifact = verify_export(export)
    metadata = json.loads((Path(export) / "omnifactory_export.json").read_text())
    config = metadata["inference_config"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    resolver = LocalMediaResolver(Path(dataset).parent)
    results = []
    for sample in iter_samples(Path(dataset)):
        for prediction in sample.predictions:
            view = inference_view(sample, prediction.id, ("text", "audio"))
            block, persona = input_context(view)
            wave, _ = load_target_audio(
                resolver,
                view.assets[block.media.asset_id],
                block.media,
                max_seconds=config["max_audio_seconds"],
                max_bytes=config["budget"]["max_decoded_media_bytes"],
                target_rate=24000,
            )
            if (len(wave) + 1919) // 1920 > max_audio_units:
                raise OmniFactoryError("E_CODEC_LENGTH", "realtime input exceeds frame bound")
            destination = output / f"session-{len(results):05d}"
            destination.mkdir()
            result = asyncio.run(
                session_probe(url, model, config["personaplex_voice"], persona, wave, destination)
            )
            results.append(dict(sample_id=sample.id, prediction_id=prediction.id, **result))
    if not results:
        raise OmniFactoryError("E_SERVE_DATASET", "empty realtime dataset")
    return dict(
        status="passed",
        artifact_id=artifact["artifact_id"],
        adapter="personaplex",
        modality="audio",
        transport="websocket",
        endpoint="/v1/realtime?duplex=1",
        requests=len(results),
        results=results,
        quality_verified=False,
    )
