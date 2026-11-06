"""Live BAGEL trained-export image API acceptance, independent of training runtime."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from omnifactory.data.io import iter_samples
from omnifactory.data.schema import inference_view


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8094")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample in iter_samples(args.dataset):
        view = inference_view(sample, "p0", ("image",))
        assert len(view.messages) == 1 and all(b.type == "text" for b in view.messages[0].content)
        prompt = "".join(b.text for b in view.messages[0].content)
        payload = dict(
            model=args.model,
            prompt=prompt,
            response_format="b64_json",
            n=1,
            size=f"{args.size}x{args.size}",
            num_inference_steps=args.steps,
            seed=42,
        )
        response = requests.post(args.url + "/v1/images/generations", json=payload, timeout=600)
        (args.output / (sample.id + ".json")).write_text(response.text)
        response.raise_for_status()
        result = response.json()
        assert len(result["data"]) == 1
        blob = base64.b64decode(result["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(blob))
        image.load()
        assert image.size == (args.size, args.size)
        pixels = np.asarray(image.convert("RGB"))
        assert pixels.std() > 0
        (args.output / (sample.id + ".png")).write_bytes(blob)
        row = dict(
            id=sample.id,
            http_status=response.status_code,
            width=image.width,
            height=image.height,
            sha256=hashlib.sha256(blob).hexdigest(),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert rows
    (args.output / "acceptance.json").write_text(
        json.dumps(
            dict(
                model=args.model,
                requests_passed=len(rows),
                results=rows,
                live_http_verified=True,
                quality_verified=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
