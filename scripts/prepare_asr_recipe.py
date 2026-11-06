"""Make an explicit transcript-only prompt variant of existing canonical ASR data.

Preserves transcripts, IDs, groups and recording hashes; resolves media paths to
original files. This changes the task instruction, not the evaluation reference.
"""

import argparse
from pathlib import Path

from omnifactory.artifacts import sha256
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory, write_json

SYSTEM_PROMPT = "You are a speech recognition system. Transcribe the supplied audio verbatim. Output only the transcript, with no introduction, explanation, or offer of assistance."


def prepare(source, output, *, system_prompt=SYSTEM_PROMPT):
    if not system_prompt.strip():
        raise ValueError("system prompt cannot be empty")
    counts = {}
    with new_directory(output) as root:
        for split in ("train", "validation"):
            records = []
            for sample in iter_samples(source / f"{split}.jsonl"):
                row = sample.model_dump(mode="json")
                for asset in row["assets"].values():
                    path = Path(asset["uri"])
                    if "://" in asset["uri"]:
                        raise ValueError("local media required")
                    asset["uri"] = str((path if path.is_absolute() else source / path).resolve())
                row["messages"] = [
                    dict(
                        id="asr-system",
                        role="system",
                        content=[dict(id="asr-instruction", type="text", text=system_prompt)],
                    ),
                    *[m for m in row["messages"] if m["role"] != "system"],
                ]
                records.append(CanonicalSample.model_validate(row))
            if not records:
                raise ValueError("empty split")
            counts[split] = len(records)
            (root / f"{split}.jsonl").write_text(
                "".join(r.model_dump_json() + "\n" for r in records)
            )
        write_json(
            root / "dataset_info.json",
            dict(
                source=str(source.resolve()),
                system_prompt=system_prompt,
                counts=counts,
                source_manifests={split: sha256(source / f"{split}.jsonl") for split in counts},
                scope="explicit ASR instruction variant; targets and media unchanged",
            ),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    prepare(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
