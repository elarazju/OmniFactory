"""Wire-contract regressions using an explicitly synthetic WebSocket server."""

import asyncio
import base64
import io
import json
import tarfile

import pytest

pytest.importorskip("numpy")
pytest.importorskip("soundfile")
pytest.importorskip("websockets.asyncio.server")
import numpy as np
from websockets.asyncio.server import serve

from omnifactory.adapters.personaplex_sglang_assets import unpack_voices
from omnifactory.adapters.personaplex_sglang_serving import session_probe
from omnifactory.errors import OmniFactoryError


@pytest.mark.parametrize(
    "fault,error",
    [
        (None, None),
        ("no_overlap", "E_REALTIME_OVERLAP"),
        ("wrong_frames", "E_REALTIME_AUDIO"),
        ("discard", "E_REALTIME_DRAIN"),
        ("unfinished", "E_REALTIME_RESPONSE"),
        ("second_response", "E_REALTIME_RESPONSE"),
        ("not_native", "E_REALTIME_CAPABILITY"),
    ],
)
def test_sglang_native_wire_protocol(tmp_path, fault, error):
    observed = []
    output = tmp_path / "probe"
    output.mkdir()

    async def scenario():
        async def server(ws):
            assert ws.request.path == "/v1/realtime?model=personaplex"
            await ws.send(json.dumps(dict(type="session.created", session=dict(id="session-1"))))
            pending = []
            async for raw in ws:
                event = json.loads(raw)
                observed.append(event)
                assert event["event_id"] == f"probe-{len(observed) - 1}"
                kind = event["type"]
                if kind == "session.update":
                    config = event["session"]
                    assert config["sglang"]["voice"] == "NATF2"
                    assert config["instructions"] == "You are a helpful assistant."
                    assert config["output_modalities"] == ["audio"]
                    assert config["audio"]["input"] == {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": None,
                    }
                    await ws.send(
                        json.dumps(
                            dict(
                                type="session.updated",
                                session=dict(
                                    sglang=dict(
                                        granted=dict(
                                            native_full_duplex=fault != "not_native",
                                            interaction="native",
                                            native_unit_ms=80,
                                            rejections=[],
                                            input_audio_format=dict(type="audio/pcm", rate=24000),
                                            output_audio_format=dict(type="audio/pcm", rate=24000),
                                        )
                                    ),
                                ),
                            )
                        )
                    )
                elif kind == "input_audio_buffer.append":
                    seq = len([e for e in observed if e["type"] == kind]) - 1
                    assert event["sglang"] == dict(seq=seq, t_start_ms=float(seq * 80))
                    pcm = np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")
                    assert len(pcm) == 1920
                    assert pcm[0] == 8192  # Float input quantized to signed PCM16.
                    frames = 2 if seq == 0 and fault != "wrong_frames" else 1
                    delta = dict(
                        type="response.output_audio.delta",
                        response_id=(
                            "response-2"
                            if fault == "second_response" and seq == 1
                            else "response-1"
                        ),
                        delta=base64.b64encode(np.full(frames * 1920, 42, dtype="<i2")).decode(),
                    )
                    if fault == "no_overlap":
                        pending.append(delta)
                    else:
                        await ws.send(json.dumps(delta))
                elif kind == "sglang.input_audio.end":
                    for delta in pending:
                        await ws.send(json.dumps(delta))
                    await ws.send(
                        json.dumps(
                            dict(
                                type="response.done",
                                response=dict(
                                    id="response-1",
                                    status="incomplete" if fault == "unfinished" else "completed",
                                    status_details=dict(reason="input_audio_ended"),
                                ),
                            )
                        )
                    )
                    await ws.send(
                        json.dumps(
                            dict(
                                type="sglang.input_audio.drained",
                                accepted_end_ms=240,
                                consumed_ms=240,
                                padding_ms=0,
                                discarded_ms=80 if fault == "discard" else 0,
                            )
                        )
                    )
                elif kind == "session.close":
                    await ws.send(json.dumps(dict(type="session.closed", reason="client_closed")))
                    break

        async with serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            return await session_probe(
                f"http://127.0.0.1:{port}",
                "personaplex",
                "NATF2.pt",
                "You are a helpful assistant.",
                np.full(1920 * 3, 0.25),
                output,
                timeout=2,
            )

    if error:
        with pytest.raises(OmniFactoryError, match=error):
            asyncio.run(scenario())
    else:
        result = asyncio.run(scenario())
        assert result["output_during_input"] is True
        assert result["input_frames"] == result["wire_input_frames"] == 3
        assert result["drain_frames"] == 0
        assert result["output_frames"] == 4
        assert result["finish_reason"] == "input_audio_ended"
        assert (output / "output.wav").is_file()
    assert (output / "request-events.jsonl").is_file()
    assert (output / "response-events.jsonl").is_file()


def test_unpack_voices_uses_regular_basenames_and_is_idempotent(tmp_path):
    with tarfile.open(tmp_path / "voices.tgz", "w:gz") as archive:
        member = tarfile.TarInfo("../../voices/NATF2.pt")
        member.size = 5
        archive.addfile(member, io.BytesIO(b"voice"))
        link = tarfile.TarInfo("voices/EVIL.pt")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    unpack_voices(tmp_path)
    assert sorted(p.name for p in (tmp_path / "voices").iterdir()) == ["NATF2.pt"]
    assert (tmp_path / "voices/NATF2.pt").read_bytes() == b"voice"
    unpack_voices(tmp_path)


def test_unpack_voices_rejects_duplicate_presets(tmp_path):
    with tarfile.open(tmp_path / "voices.tgz", "w:gz") as archive:
        for path in ("a/NATF2.pt", "b/NATF2.pt"):
            member = tarfile.TarInfo(path)
            member.size = 5
            archive.addfile(member, io.BytesIO(b"voice"))
    with pytest.raises(OmniFactoryError, match="invalid PersonaPlex voice archive"):
        unpack_voices(tmp_path)
