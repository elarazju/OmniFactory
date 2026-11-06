"""Locally generated image/prompt pairs for execution checks, never a quality dataset."""

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory


def create_image_demo(output: Path, size=256, *, editing=False):
    if size < 16 or size > 1024:
        raise ValueError("fixture image size must be between 16 and 1024")
    with new_directory(output) as root:
        rows = []
        for i, color in enumerate(("red", "blue", "green", "orange", "purple", "yellow")):
            image = Image.new("RGB", (size, size), "white")
            draw = ImageDraw.Draw(image)
            bounds = (size // 4, size // 4, 3 * size // 4, 3 * size // 4)
            shape = "circle" if i % 2 == 0 else "square"
            (draw.ellipse if shape == "circle" else draw.rectangle)(bounds, fill=color)
            path = root / f"image-{i}.png"
            image.save(path)
            row = CanonicalSample.model_validate(
                dict(
                    schema_version="0.1",
                    id=f"image-{i}",
                    provenance=dict(
                        dataset="omnifactory-image-demo",
                        source_id=str(i),
                        group_id=str(i),
                        split="train" if i < 4 else "validation",
                    ),
                    assets={
                        "target": dict(
                            uri=path.name,
                            modality="image",
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        )
                    },
                    messages=[
                        dict(
                            id="user",
                            role="user",
                            content=[
                                dict(
                                    id="prompt",
                                    type="text",
                                    text=f"A {color} {shape} on a white background.",
                                )
                            ],
                        ),
                        dict(
                            id="assistant",
                            role="assistant",
                            content=[dict(id="image", type="image", asset_id="target")],
                        ),
                    ],
                    predictions=[
                        dict(id="p0", target_message_id="assistant", target_block_ids=["image"])
                    ],
                )
            )
            if editing:
                source = Image.new("RGB", (size, size), "white")
                source_draw = ImageDraw.Draw(source)
                (source_draw.ellipse if shape == "circle" else source_draw.rectangle)(
                    bounds, fill="gray"
                )
                source_path = root / f"source-{i}.png"
                source.save(source_path)
                raw = row.model_dump(mode="json")
                raw["provenance"]["dataset"] = "omnifactory-image-edit-demo"
                raw["assets"]["source"] = dict(
                    uri=source_path.name,
                    modality="image",
                    sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                )
                raw["messages"][0]["content"] = [
                    dict(id="source", type="image", asset_id="source"),
                    dict(
                        id="prompt",
                        type="text",
                        text=f"Change the gray {shape} to {color}. Keep the white background.",
                    ),
                ]
                row = CanonicalSample.model_validate(raw)
            rows.append(row)
        for split in ("train", "validation"):
            (root / f"{split}.jsonl").write_text(
                "".join(r.model_dump_json() + "\n" for r in rows if r.provenance.split == split)
            )
    return dict(
        dataset=str(output.resolve()),
        synthetic_images=True,
        quality_benchmark=False,
        task="image_editing" if editing else "text_to_image",
    )
