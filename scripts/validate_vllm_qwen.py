"""Exercise a running vLLM-Omni server with text and speech output requests."""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


def content_for(sample, dataset):
    content = []
    for message in sample["messages"]:
        if message["role"] != "user":
            continue
        for block in message["content"]:
            kind = block["type"]
            if kind == "text":
                content.append({"type": "text", "text": block["text"]})
            elif kind in ("image", "audio", "video"):
                asset_id = block["asset_id"] if kind == "image" else block["media"]["asset_id"]
                asset = sample["assets"][asset_id]
                path = Path(asset["uri"])
                if not path.is_absolute():
                    path = dataset.parent / path
                mime = {"image": "image/png", "audio": "audio/wav", "video": "video/mp4"}[kind]
                url = "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()
                content.append({"type": kind + "_url", kind + "_url": {"url": url}})
    content.append({"type": "text", "text": "Answer in five words or fewer."})
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", choices=("qwen2_5_omni", "qwen3_omni"), default="qwen2_5_omni")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--video-only", action="store_true", help="Use only video rows from the demo"
    )
    args = parser.parse_args()
    codec_eos = 2150 if args.adapter == "qwen3_omni" else 8294
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in args.dataset.read_text().splitlines()]
    if args.video_only:
        rows = [
            row
            for row in rows
            if any(
                block["type"] == "video"
                for msg in row["messages"]
                if msg["role"] == "user"
                for block in msg["content"]
            )
        ]
        if not rows:
            parser.error("--video-only requires at least one video row")
    else:
        assert len(rows) == 4
    reports = []
    for audio in (False, True):
        for row in rows:
            request = {
                "model": args.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                        "capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    },
                    {"role": "user", "content": content_for(row, args.dataset)},
                ],
                "modalities": ["text", "audio"] if audio else ["text"],
                "max_tokens": 64,
                "temperature": 0.0,
                "stream": False,
                "sampling_params_list": [
                    {"temperature": 0.0, "max_tokens": 64, "seed": 42},
                    {
                        "temperature": 0.9,
                        "top_p": 1.0 if args.adapter == "qwen3_omni" else 0.8,
                        "top_k": 50 if args.adapter == "qwen3_omni" else 40,
                        "max_tokens": 750,
                        "seed": 42,
                        "repetition_penalty": 1.05,
                        "stop_token_ids": [codec_eos],
                    },
                    {"temperature": 0.0, "max_tokens": 750, "seed": 42},
                ],
            }
            if any(x["type"] == "video_url" for x in request["messages"][-1]["content"]):
                request["mm_processor_kwargs"] = {"use_audio_in_video": True}
            name = row["id"] + ("-audio" if audio else "-text")
            (args.output / (name + ".request.json")).write_text(json.dumps(request))
            response = requests.post(args.url + "/v1/chat/completions", json=request, timeout=600)
            (args.output / (name + ".json")).write_text(response.text)
            response.raise_for_status()
            result = response.json()
            choices = result["choices"]
            text = []
            waves = []
            for choice in choices:
                message = choice["message"]
                value = message.get("content")
                if isinstance(value, str) and value:
                    text.append(value)
                elif isinstance(value, list):
                    for part in value:
                        if part.get("type") == "text":
                            text.append(part["text"])
                        elif part.get("type") == "audio":
                            waves.append(part["audio"]["data"])
                if message.get("audio"):
                    waves.append(message["audio"]["data"])
                    if message["audio"].get("transcript"):
                        text.append(message["audio"]["transcript"])
            assert any(value.strip() for value in text), result
            assert all(x["finish_reason"] == "stop" for x in choices), result
            frames = 0
            for i, data in enumerate(waves):
                blob = base64.b64decode(data, validate=True)
                waveform, rate = sf.read(io.BytesIO(blob))
                assert rate == 24000 and waveform.size and np.isfinite(waveform).all()
                assert np.any(waveform != 0), "silent waveform"
                (args.output / f"{name}-{i}.wav").write_bytes(blob)
                frames += len(waveform)
            assert frames > 0 if audio else not waves, result
            report = {
                "id": name,
                "http_status": response.status_code,
                "text": text,
                "audio_frames": frames,
                "finish_reasons": [x["finish_reason"] for x in choices],
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
    (args.output / "acceptance.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "requests_passed": len(reports),
                "input_modalities": ["video_with_audio"]
                if args.video_only
                else ["text", "image", "audio", "video_with_audio"],
                "output_modalities": ["text", "text+audio"],
                "results": reports,
                "quality_verified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
