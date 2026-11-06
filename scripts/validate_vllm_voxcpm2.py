"""Live HTTP acceptance of a trained native VoxCPM2 export; no quality claim."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from omnifactory.data.io import iter_samples
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8093")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    resolver = LocalMediaResolver(args.dataset.parent)
    reports = []
    for sample in iter_samples(args.dataset):
        view = inference_view(sample, "p0", ("audio",))
        assert len(view.messages) == 1 and all(b.type == "text" for b in view.messages[0].content)
        text = "".join(b.text for b in view.messages[0].content)
        request = dict(
            model=args.model,
            input=text,
            voice="default",
            response_format="wav",
            stream=False,
            max_new_tokens=128,
        )
        assert len(view.conditions) <= 1
        if view.conditions:
            condition = view.conditions[0]
            assert condition.kind == "reference_audio" and not condition.transcript
            media = condition.media
            path = resolver.resolve(sample.assets[media.asset_id]).path
            info = sf.info(path)
            start = round(media.start_us * info.samplerate / 1e6)
            end = (
                round(media.end_us * info.samplerate / 1e6)
                if media.end_us is not None
                else info.frames
            )
            wave, rate = sf.read(path, start=start, stop=end, dtype="float32", always_2d=True)
            blob = io.BytesIO()
            sf.write(blob, wave.mean(axis=1), rate, format="WAV", subtype="FLOAT")
            request["ref_audio"] = (
                "data:audio/wav;base64," + base64.b64encode(blob.getvalue()).decode()
            )
        response = requests.post(args.url + "/v1/audio/speech", json=request, timeout=600)
        (args.output / (sample.id + ".response")).write_bytes(response.content)
        response.raise_for_status()
        wave, rate = sf.read(io.BytesIO(response.content))
        assert rate == 48000 and wave.size and np.isfinite(wave).all()
        assert np.max(np.abs(wave)) > 0
        (args.output / (sample.id + ".wav")).write_bytes(response.content)
        record = dict(
            id=sample.id,
            input=text,
            reference_audio=bool(view.conditions),
            http_status=response.status_code,
            audio_frames=len(wave),
            sample_rate=rate,
            response_sha256=hashlib.sha256(response.content).hexdigest(),
            headers=dict(response.headers),
        )
        reports.append(record)
        print(json.dumps(record), flush=True)
    assert reports and {r["reference_audio"] for r in reports} == {False, True}
    (args.output / "acceptance.json").write_text(
        json.dumps(
            dict(
                model=args.model,
                requests_passed=len(reports),
                results=reports,
                live_http_verified=True,
                quality_verified=False,
                termination_observable=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
