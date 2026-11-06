"""Launch a verified export and exercise native chat, speech or image HTTP APIs."""

import base64
import io
import json
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from omnifactory.adapters.runtime import adapter_spec, resolve
from omnifactory.artifacts import sha256, verify_export
from omnifactory.data.io import atomic_writer, iter_samples
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view
from omnifactory.deployment import serving_messages
from omnifactory.errors import OmniFactoryError


def http_request(url, body=None, timeout=600):
    request = Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read(), dict(response.headers)


def multipart_request(url, fields, files, *, output, timeout=600):
    """Send and retain exact multipart bytes for artifact-bound editing checks."""
    boundary = "omnifactory-" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        if not name.isidentifier():
            raise OmniFactoryError("E_SERVE_FIELD", "invalid multipart field name")
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{str(value).lower() if isinstance(value, bool) else value}\r\n"
            ).encode()
        )
    for name, (filename, mime, data) in files.items():
        if (
            not name.isidentifier()
            or Path(filename).name != filename
            or any(c in filename + mime for c in '\r\n"')
        ):
            raise OmniFactoryError("E_SERVE_FIELD", "invalid multipart file header")
        parts.extend(
            [
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
                ).encode(),
                data,
                b"\r\n",
            ]
        )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    Path(output).write_bytes(body)
    Path(str(output) + ".headers.json").write_text(json.dumps({"Content-Type": content_type}))
    request = Request(url, data=body, headers={"Content-Type": content_type})
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read(), dict(response.headers)


def probe(export, dataset, output, url, model, *, max_audio_units=128, size=256, image_steps=16):
    import numpy as np
    import soundfile as sf
    from PIL import Image

    artifact = verify_export(export)
    spec = adapter_spec(artifact["adapter"])
    if spec.realtime_probe:
        return resolve(spec.realtime_probe)(
            export, dataset, output, url, model, max_audio_units=max_audio_units
        )
    if spec.stage_only:
        raise OmniFactoryError("E_SERVE_STAGE", "stage-only export needs a native host pipeline")
    metadata = json.loads((Path(export) / "omnifactory_export.json").read_text())
    config = metadata["inference_config"]
    modality = {"image_sft": "image", "speech_sft": "audio", "understanding_sft": "text"}.get(
        config["task"]
    )
    if modality is None or modality not in spec.outputs:
        raise OmniFactoryError("E_SERVE_TASK", "unsupported exported task")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    resolver = LocalMediaResolver(Path(dataset).parent)
    for sample in iter_samples(Path(dataset)):
        for prediction in sample.predictions:
            view = inference_view(sample, prediction.id, (modality,))
            text = "".join(b.text for m in view.messages for b in m.content if b.type == "text")
            files = None
            if modality == "image":
                if not spec.image_edit_request_builder and any(
                    b.type != "text" for m in view.messages for b in m.content
                ):
                    raise OmniFactoryError(
                        "E_SERVE_MODALITY", "image generation probe requires text conditioning"
                    )
                endpoint = "/v1/images/generations"
                payload = dict(
                    model=model,
                    prompt=text,
                    n=1,
                    response_format="b64_json",
                    size=f"{size}x{size}",
                    num_inference_steps=image_steps,
                    seed=42,
                )
                if spec.image_edit_request_builder:
                    endpoint = "/v1/images/edits"
                    payload, files = resolve(spec.image_edit_request_builder)(
                        payload,
                        view=view,
                        resolver=resolver,
                        metadata=metadata,
                    )
            elif spec.outputs == ("audio",):
                if any(b.type != "text" for m in view.messages for b in m.content):
                    raise OmniFactoryError("E_SERVE_MODALITY", "TTS probe requires literal text")
                endpoint = "/v1/audio/speech"
                payload = dict(
                    model=model,
                    input=text,
                    voice="default",
                    response_format="wav",
                    stream=False,
                    max_new_tokens=max_audio_units,
                )
                if spec.speech_request_builder:
                    payload = resolve(spec.speech_request_builder)(
                        payload,
                        view=view,
                        resolver=resolver,
                        metadata={**metadata, "_export_root": str(Path(export).resolve())},
                    )
                elif len(view.conditions) > 1:
                    raise OmniFactoryError("E_SERVE_CONDITION", "one reference recording supported")
                if view.conditions and not spec.speech_request_builder:
                    condition = view.conditions[0]
                    if condition.kind != "reference_audio" or condition.transcript:
                        raise OmniFactoryError("E_SERVE_CONDITION", "unsupported TTS condition")
                    media = condition.media
                    path = resolver.resolve(view.assets[media.asset_id]).path
                    info = sf.info(path)
                    start = round(media.start_us * info.samplerate / 1e6)
                    stop = (
                        round(media.end_us * info.samplerate / 1e6)
                        if media.end_us is not None
                        else info.frames
                    )
                    wave, rate = sf.read(path, start=start, stop=stop, always_2d=True)
                    buffer = io.BytesIO()
                    sf.write(buffer, wave.mean(axis=1), rate, format="WAV", subtype="FLOAT")
                    payload["ref_audio"] = (
                        "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()
                    )
            else:
                endpoint = "/v1/chat/completions"
                payload = dict(
                    model=model,
                    messages=serving_messages(sample, prediction.id, Path(dataset).parent),
                    modalities=["text", "audio"] if modality == "audio" else ["text"],
                    max_tokens=128,
                    temperature=0,
                )
                if modality == "audio":
                    payload["audio"] = {
                        "format": "wav",
                        "voice": config.get("speech_speaker") or "default",
                    }
            name = f"request-{len(records):05d}"
            (output / f"{name}.request.json").write_text(json.dumps(payload))
            if files is None:
                status, raw, headers = http_request(url + endpoint, payload)
            else:
                status, raw, headers = multipart_request(
                    url + endpoint,
                    payload,
                    files,
                    output=output / f"{name}.request.body",
                )
            (output / f"{name}.response").write_bytes(raw)
            record = dict(
                sample_id=sample.id,
                prediction_id=prediction.id,
                status=status,
                headers=headers,
                response_sha256=sha256(output / f"{name}.response"),
                endpoint=endpoint,
            )
            if files is not None:
                record["request_body_sha256"] = sha256(output / f"{name}.request.body")
            if modality == "image":
                blob = base64.b64decode(json.loads(raw)["data"][0]["b64_json"], validate=True)
                image = Image.open(io.BytesIO(blob))
                image.load()
                if image.size != (size, size) or np.asarray(image).std() == 0:
                    raise OmniFactoryError("E_SERVE_IMAGE", "invalid/constant image")
                (output / f"{name}.png").write_bytes(blob)
                record.update(width=image.width, height=image.height)
            elif modality == "audio":
                blobs = [raw]
                if endpoint.endswith("completions"):
                    blobs = []
                    for choice in json.loads(raw)["choices"]:
                        message = choice["message"]
                        if message.get("audio"):
                            blobs.append(base64.b64decode(message["audio"]["data"], validate=True))
                        if isinstance(message.get("content"), list):
                            blobs.extend(
                                base64.b64decode(p["audio"]["data"], validate=True)
                                for p in message["content"]
                                if p.get("type") == "audio"
                            )
                    if not blobs:
                        raise OmniFactoryError("E_SERVE_AUDIO", "no audio returned")
                for i, blob in enumerate(blobs):
                    wave, rate = sf.read(io.BytesIO(blob))
                    if not wave.size or not np.isfinite(wave).all() or np.abs(wave).max() == 0:
                        raise OmniFactoryError("E_SERVE_AUDIO", "invalid/empty waveform")
                    (output / f"{name}-{i}.wav").write_bytes(blob)
                    record.update(sample_rate=rate, audio_frames=len(wave))
            else:
                text = json.loads(raw)["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise OmniFactoryError("E_SERVE_TEXT", "empty text")
            records.append(record)
    if not records:
        raise OmniFactoryError("E_SERVE_DATASET", "empty probe dataset")
    return dict(
        status="passed",
        artifact_id=artifact["artifact_id"],
        adapter=artifact["adapter"],
        requests=len(records),
        results=records,
        modality=modality,
        quality_verified=False,
    )


def validate_deployment(
    export, dataset, output, executable, deploy_config, *, startup_timeout=300, max_audio_units=128
):
    """The process command binds HTTP evidence to the exact verified artifact."""
    export, output = Path(export).resolve(), Path(output).resolve()
    artifact = verify_export(export)
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    model = "omnifactory-acceptance"
    command = [
        str(executable),
        "serve",
        str(export),
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model,
        "--trust-remote-code",
        "--deploy-config",
        str(Path(deploy_config).resolve()),
        "--disable-log-stats",
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2))
    (output / "deploy.yaml").write_bytes(Path(deploy_config).read_bytes())
    result = dict(status="failed", artifact_id=artifact["artifact_id"])
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + startup_timeout
            url = f"http://127.0.0.1:{port}"
            while True:
                if process.poll() is not None:
                    raise OmniFactoryError("E_SERVE_START", "server exited; see server.log")
                try:
                    _, data, _ = http_request(url + "/v1/models", timeout=2)
                    if model in {m["id"] for m in json.loads(data)["data"]}:
                        break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise OmniFactoryError("E_SERVE_TIMEOUT", "server did not become ready")
                time.sleep(1)
            result = probe(
                export, dataset, output / "requests", url, model, max_audio_units=max_audio_units
            )
            result["launch_command"] = command
        except Exception as exc:
            result["error"] = str(exc)
            raise
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result["server_log_sha256"] = sha256(output / "server.log")
            with atomic_writer(output / "acceptance.json") as stream:
                stream.write(json.dumps(result, indent=2))
    return result
