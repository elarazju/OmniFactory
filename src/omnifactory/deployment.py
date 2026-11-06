"""Explicit HTTP smoke checks against an already running vLLM-Omni server.

This checks text responses to text/image/audio prompts. It does not certify
native speech generation or identify the server implementation cryptographically.
"""

import base64
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from omnifactory.data.io import atomic_writer, iter_samples
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view
from omnifactory.errors import OmniFactoryError


def serving_messages(sample, prediction_id, root):
    view = inference_view(sample, prediction_id, requested_outputs=("text",))
    if view.conditions:
        raise OmniFactoryError(
            "E_SERVE_CONDITION", "generation conditions are not supported by this checker"
        )
    resolver = LocalMediaResolver(root)
    messages = []
    total = 0
    for message in view.messages:
        blocks = []
        for block in message.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
                continue
            if block.type == "video" or getattr(block, "placement", None) is not None:
                raise OmniFactoryError(
                    "E_SERVE_MODALITY", "video/timeline serving checks are not implemented"
                )
            if block.type == "audio" and (block.media.start_us or block.media.end_us is not None):
                raise OmniFactoryError(
                    "E_SERVE_SLICE", "materialize audio slices before sending them"
                )
            asset_id = block.asset_id if block.type == "image" else block.media.asset_id
            resolved = resolver.resolve(view.assets[asset_id])
            total += resolved.size_bytes
            if total > 16 * 1024 * 1024:
                raise OmniFactoryError("E_SERVE_BYTES", "smoke request media exceeds 16 MiB")
            path = Path(resolved.path)
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            if block.type == "image":
                mime = mimetypes.guess_type(path.name)[0]
                if mime not in ("image/png", "image/jpeg", "image/webp"):
                    raise OmniFactoryError("E_SERVE_IMAGE", "expected PNG/JPEG/WebP")
                blocks.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
                )
            else:
                format = path.suffix.lower().lstrip(".")
                if format not in ("wav", "mp3"):
                    raise OmniFactoryError("E_SERVE_AUDIO", "expected WAV or MP3")
                blocks.append(
                    {"type": "input_audio", "input_audio": {"data": data, "format": format}}
                )
        messages.append({"role": message.role, "content": blocks})
    return messages


def check_server(
    base_url: str,
    model: str,
    *,
    dataset: Path | None = None,
    root: Path | None = None,
    output: Path | None = None,
):
    url = urlparse(base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise OmniFactoryError(
            "E_SERVE_URL", "expected an HTTP(S) API base URL without credentials/query"
        )
    if output and dataset and output.resolve() == dataset.resolve():
        raise OmniFactoryError("E_OUTPUT", "report must not overwrite the dataset")
    base = base_url.rstrip("/")

    def request(path, body=None):
        req = Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise OmniFactoryError("E_SERVE_HTTP", str(exc)) from exc

    available = request("/models")
    if model not in {m["id"] for m in available.get("data", [])}:
        raise OmniFactoryError("E_SERVE_MODEL", "requested model is not listed by the server")
    if dataset:
        prompts = (
            (s.id, p.id, serving_messages(s, p.id, root or dataset.parent))
            for s in iter_samples(dataset)
            for p in s.predictions
        )
    else:
        prompts = iter([("text-smoke", "p0", [{"role": "user", "content": "Reply with OK."}])])
    results = []
    for sample_id, prediction_id, messages in prompts:
        response = request(
            "/chat/completions",
            {
                "model": model,
                "messages": messages,
                "modalities": ["text"],
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OmniFactoryError("E_SERVE_RESPONSE", "missing assistant text") from exc
        if not isinstance(text, str) or not text.strip():
            raise OmniFactoryError("E_SERVE_RESPONSE", "empty assistant text")
        results.append({"sample_id": sample_id, "prediction_id": prediction_id, "text": text})
    if not results:
        raise OmniFactoryError("E_EMPTY_DATASET", "no serving prompts")
    result = {
        "model": model,
        "base_url": base,
        "requests": len(results),
        "responses": results,
        "text_response_smoke_passed": True,
        "speech_generation_verified": False,
        "scope": "HTTP response validity only; not semantic quality or backend identity",
    }
    if output:
        with atomic_writer(output) as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result
