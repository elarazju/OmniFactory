"""Wire-protocol tests; a local server is not pretrained deployment evidence."""

import asyncio
import base64
import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sphn")
pytest.importorskip("sentencepiece")
pytest.importorskip("websockets.asyncio.server")
import numpy as np
from websockets.asyncio.server import serve

from omnifactory.adapters.personaplex_serving import session_probe
from omnifactory.errors import OmniFactoryError


@pytest.mark.parametrize(
    "fault", [None, "server", "sample_rate", "runtime", "overlap", "missing_tail"]
)
def test_protocol_and_failure_evidence(tmp_path, fault):
    async def exercise():
        requests = []

        async def server(ws):
            delayed = []
            appended = 0
            async for raw in ws:
                request = json.loads(raw)
                requests.append(request)
                if request["type"] == "session.update":
                    await ws.send(
                        json.dumps(
                            dict(
                                type="session.created",
                                session=dict(
                                    session_id="test-session",
                                    capabilities=dict(
                                        implementation_level="model_native_duplex",
                                        chunk_period_ms=80,
                                    ),
                                ),
                            )
                        )
                    )
                    if fault == "server":
                        await ws.send(json.dumps(dict(type="error", message="test failure")))
                elif request["type"] == "input_audio_buffer.append":
                    pcm = np.frombuffer(base64.b64decode(request["audio"]), dtype="<f4")
                    assert len(pcm) == 1920 and request["sample_rate_hz"] == 24000
                    appended += 1
                    if appended == 4:
                        assert np.count_nonzero(pcm) == 0  # explicit silent drain tick
                    if appended == 1 or (appended == 4 and fault == "missing_tail"):
                        continue  # native one-frame acoustic de-delay
                    event = dict(
                        type="response.audio.delta",
                        response_id="continuous",
                        sample_rate_hz=16000 if fault == "sample_rate" else 24000,
                        delta=base64.b64encode(np.full(1920, 10, dtype="<i2")).decode(),
                        metadata=dict(
                            vllm_omni=dict(
                                runtime_impl="scheduler_data_plane",
                                uses_model_runner_scheduler=True,
                                runner_kv_backed=fault != "runtime",
                            )
                        ),
                    )
                    if fault == "overlap":
                        delayed.append(event)
                        if len(delayed) == 3:
                            await asyncio.sleep(0.1)
                            for item in delayed:
                                await ws.send(json.dumps(item))
                    else:
                        await ws.send(json.dumps(event))
                elif request["type"] == "session.close":
                    await ws.send(json.dumps(dict(type="session.closed")))
                    return

        async with serve(server, "127.0.0.1", 0) as server_instance:
            port = server_instance.sockets[0].getsockname()[1]
            call = session_probe(
                f"http://127.0.0.1:{port}",
                "model",
                "NATF2.pt",
                "persona",
                np.full(1920 * 3, 0.01, dtype=np.float32),
                tmp_path,
                timeout=0.5,
            )
            if fault:
                code = {
                    "server": "E_REALTIME_SERVER",
                    "sample_rate": "E_REALTIME_AUDIO",
                    "runtime": "E_REALTIME_RUNTIME",
                    "overlap": "E_REALTIME_OVERLAP",
                    "missing_tail": "E_REALTIME_TIMEOUT",
                }[fault]
                with pytest.raises(OmniFactoryError, match=code):
                    await call
            else:
                result = await call
                assert result["input_frames"] == result["output_frames"] == 3
                assert result["session_id"] == "test-session" and result["output_during_input"]
                assert result["drain_frames"] == 1 and result["wire_input_frames"] == 4
                assert result["first_audio_seconds"] < result["last_input_seconds"]
        assert (tmp_path / "request-events.jsonl").read_text()
        assert (tmp_path / "response-events.jsonl").read_text()
        assert requests[0]["session"]["input_audio_format"] == "pcm_f32le"

    asyncio.run(exercise())
