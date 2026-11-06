"""HTTP acceptance of trained MiniCPM/MiMo exports with text and waveform outputs."""

import argparse
import base64
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from validate_vllm_qwen import content_for


def request_content(row, dataset, family):
    content = content_for(row, dataset)
    if family != "minicpm":
        return content
    # Upstream MiniCPM's video_url loader supplies visual frames. Include the
    # audio track explicitly when requested by the canonical sample.
    for message in row["messages"]:
        if message["role"] != "user":
            continue
        for block in message["content"]:
            if block["type"] != "video" or block.get("audio_mode") != "include":
                continue
            path = Path(row["assets"][block["media"]["asset_id"]]["uri"])
            if not path.is_absolute():
                path = dataset.parent / path
            wav = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            ).stdout
            content.insert(
                -1,
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/wav;base64," + base64.b64encode(wav).decode()},
                },
            )
    return content


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("minicpm", "mimo"), required=True)
    p.add_argument("--url", default="http://127.0.0.1:8092")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    results = []
    for want_audio in (False, True):
        for row in rows:
            content = request_content(row, args.dataset, args.family)
            params = [dict(temperature=0.0, top_p=1.0, top_k=-1, seed=42, max_tokens=128)]
            if args.family == "minicpm":
                params.extend(
                    [
                        dict(
                            temperature=0.8,
                            top_p=0.85,
                            top_k=25,
                            seed=42,
                            max_tokens=512,
                            repetition_penalty=1.05,
                        ),
                        dict(temperature=0.0, seed=42, max_tokens=4096),
                    ]
                )
            else:
                # MiMo's global steps interleave text and audio patches. Use
                # the upstream client's budget and repetition settings, not a
                # short text-only token budget for this fused stage.
                params[0].update(
                    max_tokens=3200,
                    repetition_penalty=1.1,
                    stop_token_ids=[151671, 151645],
                )
                params.append(dict(temperature=0.0, seed=42, max_tokens=65536))
            request = dict(
                model=args.model,
                messages=[dict(role="user", content=content)],
                modalities=["text", "audio"] if want_audio else ["text"],
                stream=False,
                max_tokens=params[0]["max_tokens"],
                temperature=0.0,
                sampling_params_list=params,
            )
            if args.family == "minicpm":
                request["chat_template_kwargs"] = {
                    "enable_thinking": False,
                    "use_tts_template": want_audio,
                }
            name = row["id"] + ("-audio" if want_audio else "-text")
            response = requests.post(args.url + "/v1/chat/completions", json=request, timeout=600)
            (args.output / f"{name}.json").write_text(response.text)
            response.raise_for_status()
            result = response.json()
            texts, blobs = [], []
            for choice in result["choices"]:
                message = choice["message"]
                if isinstance(message.get("content"), str) and message["content"]:
                    texts.append(message["content"])
                elif isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item.get("type") == "text":
                            texts.append(item["text"])
                        elif item.get("type") == "audio":
                            blobs.append(item["audio"]["data"])
                if message.get("audio"):
                    blobs.append(message["audio"]["data"])
                    if message["audio"].get("transcript"):
                        texts.append(message["audio"]["transcript"])
            frames = 0
            for i, blob in enumerate(blobs):
                raw = base64.b64decode(blob)
                waveform, rate = sf.read(io.BytesIO(raw))
                assert rate == 24000 and waveform.size and np.isfinite(waveform).all()
                (args.output / f"{name}-{i}.wav").write_bytes(raw)
                frames += len(waveform)
            assert texts, result
            assert frames > 0 if want_audio else not blobs, result
            assert all(c["finish_reason"] == "stop" for c in result["choices"]), result
            record = dict(
                id=name,
                http_status=response.status_code,
                text=texts,
                audio_frames=frames,
                finish_reasons=[c["finish_reason"] for c in result["choices"]],
            )
            results.append(record)
            print(json.dumps(record), flush=True)
    (args.output / "acceptance.json").write_text(
        json.dumps(
            dict(
                family=args.family,
                model=args.model,
                requests_passed=len(results),
                results=results,
                quality_verified=False,
                live_http_verified=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
