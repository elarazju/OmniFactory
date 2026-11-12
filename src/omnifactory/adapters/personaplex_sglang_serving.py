"""PersonaPlex native SGLang-Omni realtime protocol, pinned at eeb418d210bd.

This probe requires output during live input, full input drain, and one completed
response. It does not equate protocol acceptance with speech quality verification.
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

from omnifactory.artifacts import sha256, verify_export
from omnifactory.errors import OmniFactoryError


async def session_probe(
    url, model, voice, persona, wave, output, *, timeout=180, require_overlap=True
):
    """Record one session; require_overlap=False is for explicit warmup only."""
    wave = np.asarray(wave)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise OmniFactoryError("E_REALTIME_AUDIO", "finite nonempty mono input required")
    parsed = urlsplit(url)
    endpoint = urlunsplit(
        (
            "wss" if parsed.scheme == "https" else "ws",
            parsed.netloc,
            parsed.path.rstrip("/") + "/v1/realtime",
            urlencode(dict(model=model)),
            "",
        )
    )
    output = Path(output)
    frame_count = (len(wave) + 1919) // 1920
    sent, received, audio = [], [], []
    started, first_audio, last_append = time.monotonic(), None, None
    session_id, granted = None, None
    condition = asyncio.Condition()
    async with connect(
        endpoint, max_size=32 * 1024 * 1024, open_timeout=30, close_timeout=10
    ) as ws:

        async def read():
            nonlocal first_audio
            async for raw in ws:
                event = json.loads(raw)
                received.append(event)
                if event.get("type") == "response.output_audio.delta":
                    chunk = base64.b64decode(event["delta"], validate=True)
                    if not chunk or len(chunk) % 2:
                        raise OmniFactoryError("E_REALTIME_AUDIO", "invalid PCM16 output chunk")
                    audio.append(chunk)
                    first_audio = first_audio or time.monotonic()
                async with condition:
                    condition.notify_all()

        reader = asyncio.create_task(read())

        async def wait_for(predicate):
            deadline = time.monotonic() + timeout
            while True:
                if any(e.get("type") == "error" for e in received):
                    raise OmniFactoryError("E_REALTIME_SERVER", "server error; see response events")
                if reader.done() and not reader.cancelled() and reader.exception():
                    reader.result()
                if predicate():
                    return
                if reader.done():
                    raise OmniFactoryError("E_REALTIME_CLOSED", "session closed before completion")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OmniFactoryError("E_REALTIME_TIMEOUT", "session completion timed out")
                async with condition:
                    try:
                        await asyncio.wait_for(condition.wait(), min(remaining, 0.2))
                    except asyncio.TimeoutError:
                        pass

        def events(kind):
            return [event for event in received if event.get("type") == kind]

        async def send(kind, **fields):
            event = dict(type=kind, event_id=f"probe-{len(sent)}", **fields)
            sent.append(event)
            await ws.send(json.dumps(event))

        try:
            await wait_for(lambda: events("session.created"))
            session_id = events("session.created")[0].get("session", {}).get("id")
            await send(
                "session.update",
                session=dict(
                    type="realtime",
                    model=model,
                    instructions=persona,
                    output_modalities=["audio"],
                    audio={
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": None,
                        },
                        "output": {"format": {"type": "audio/pcm", "rate": 24000}},
                    },
                    sglang=dict(voice=voice.removesuffix(".pt")),
                ),
            )
            await wait_for(lambda: events("session.updated"))
            granted = (
                events("session.updated")[0].get("session", {}).get("sglang", {}).get("granted", {})
            )
            if (
                granted.get("native_full_duplex") is not True
                or granted.get("interaction") != "native"
                or granted.get("native_unit_ms") != 80
                or granted.get("input_audio_format") != {"type": "audio/pcm", "rate": 24000}
                or granted.get("output_audio_format") != {"type": "audio/pcm", "rate": 24000}
                or granted.get("rejections")
            ):
                raise OmniFactoryError(
                    "E_REALTIME_CAPABILITY", "native 24 kHz/80 ms duplex required"
                )
            input_started = time.monotonic()
            for seq in range(frame_count):
                await asyncio.sleep(max(0, input_started + seq * 0.08 - time.monotonic()))
                frame = wave[seq * 1920 : (seq + 1) * 1920]
                frame = np.pad(frame, (0, 1920 - len(frame)))
                pcm = np.clip(np.rint(frame * 32768), -32768, 32767).astype("<i2")
                await send(
                    "input_audio_buffer.append",
                    audio=base64.b64encode(pcm).decode(),
                    sglang=dict(seq=seq, t_start_ms=float(seq * 80)),
                )
                last_append = time.monotonic()
            await asyncio.sleep(max(0, input_started + frame_count * 0.08 - time.monotonic()))
            await send("sglang.input_audio.end")
            await wait_for(lambda: events("sglang.input_audio.drained") and events("response.done"))
            await send("session.close")
            await wait_for(lambda: events("session.closed"))
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            for name, values in (("request", sent), ("response", received)):
                (output / f"{name}-events.jsonl").write_text(
                    "".join(json.dumps(event) + "\n" for event in values)
                )
    drained = events("sglang.input_audio.drained")
    if len(drained) != 1 or any(
        drained[0].get(key) != value
        for key, value in dict(
            accepted_end_ms=frame_count * 80,
            consumed_ms=frame_count * 80,
            discarded_ms=0,
            padding_ms=0,
        ).items()
    ):
        raise OmniFactoryError("E_REALTIME_DRAIN", "server did not consume all input frames")
    audio_events = events("response.output_audio.delta")
    response_ids = {event.get("response_id") for event in audio_events}
    done = events("response.done")
    if (
        len(response_ids) != 1
        or None in response_ids
        or len(done) != 1
        or done[0].get("response", {}).get("id") not in response_ids
        or done[0]["response"].get("status") != "completed"
        or done[0]["response"].get("status_details", {}).get("reason") != "input_audio_ended"
    ):
        raise OmniFactoryError(
            "E_REALTIME_RESPONSE", "one completed native duplex response required"
        )
    pcm = np.frombuffer(b"".join(audio), dtype="<i2")
    # Native first unit includes the prompt-tail frame: 160 ms, then 80 ms/unit.
    if len(pcm) != (frame_count + 1) * 1920 or not np.any(pcm):
        raise OmniFactoryError("E_REALTIME_AUDIO", "wrong frame count or silent session output")
    overlap = first_audio is not None and first_audio < last_append
    if require_overlap and not overlap:
        raise OmniFactoryError("E_REALTIME_OVERLAP", "no output arrived during live input")
    sf.write(output / "output.wav", pcm, 24000, subtype="PCM_16")
    return dict(
        session_id=session_id,
        input_frames=frame_count,
        wire_input_frames=frame_count,
        drain_frames=0,
        prompt_tail_frames=1,
        output_frames=len(pcm) // 1920,
        sample_rate=24000,
        audio_chunks=len(audio),
        output_during_input=overlap,
        first_audio_seconds=first_audio - started,
        last_input_seconds=last_append - started,
        granted_capabilities=granted,
        finish_reason="input_audio_ended",
        request_sha256=sha256(output / "request-events.jsonl"),
        response_sha256=sha256(output / "response-events.jsonl"),
        audio_sha256=sha256(output / "output.wav"),
    )


def probe_realtime(export, dataset, output, url, model, *, max_audio_units=128):
    from omnifactory.adapters.personaplex import input_context
    from omnifactory.adapters.speech import load_target_audio
    from omnifactory.data.io import iter_samples
    from omnifactory.data.media import LocalMediaResolver
    from omnifactory.data.schema import inference_view

    artifact = verify_export(export)
    metadata = json.loads((Path(export) / "omnifactory_export.json").read_text())
    config = metadata["inference_config"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    resolver = LocalMediaResolver(Path(dataset).parent)
    results = []
    warmup = None
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
            if warmup is None:
                warmup_output = output / "warmup"
                warmup_output.mkdir()
                warmup = dict(
                    purpose="initialize native model session caches; excluded from acceptance",
                    counts_as_acceptance=False,
                    source_sample_id=sample.id,
                    source_prediction_id=prediction.id,
                    **asyncio.run(
                        session_probe(
                            url,
                            model,
                            config["personaplex_voice"],
                            persona,
                            wave,
                            warmup_output,
                            require_overlap=False,
                        )
                    ),
                )
            destination = output / f"session-{len(results):05d}"
            destination.mkdir()
            result = asyncio.run(
                session_probe(
                    url,
                    model,
                    config["personaplex_voice"],
                    persona,
                    wave,
                    destination,
                )
            )
            results.append(dict(sample_id=sample.id, prediction_id=prediction.id, **result))
    if not results:
        raise OmniFactoryError("E_SERVE_DATASET", "empty realtime dataset")
    return dict(
        status="passed",
        artifact_id=artifact["artifact_id"],
        adapter="personaplex",
        backend="sglang-omni",
        modality="audio",
        transport="websocket",
        endpoint="/v1/realtime",
        requests=len(results),
        warmup=warmup,
        results=results,
        quality_verified=False,
    )
