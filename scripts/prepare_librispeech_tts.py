"""Re-express an existing speaker-disjoint ASR pilot as real text/waveform TTS pairs.

Retains the original partition/license; references come from other utterances of
the same speaker and split. This small execution dataset is not a TTS benchmark.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from omnifactory.artifacts import sha256
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.training.artifacts import new_directory, write_json


def prepare(
    source, output, count=4, *, train_count=None, validation_count=None, text_case="original"
):
    if text_case not in ("original", "sentence"):
        raise ValueError("text_case must be original or sentence")
    with new_directory(output) as root:
        groups_seen = {}
        text_changes = []
        for split in ("train", "validation"):
            limit = {"train": train_count, "validation": validation_count}[split] or count
            samples = list(iter_samples(source / f"{split}.jsonl"))
            groups = defaultdict(list)
            for sample in samples:
                groups[sample.provenance.group_id].append(sample)
            groups_seen[split] = set(groups)
            rows = []
            # Retain explicit source IDs and group identities; no repartitioning.
            for sample in samples:
                if len(rows) == limit:
                    break
                row = sample.model_dump(mode="json")
                text = row["messages"][1]["content"][0]["text"]
                original_text = text
                if text_case == "sentence":
                    if text != text.upper():
                        raise ValueError("sentence conversion expects uppercase LibriSpeech text")
                    text = text.capitalize()
                text_changes.append(
                    dict(id=sample.id, split=split, original=original_text, transformed=text)
                )
                asset = dict(row["assets"]["recording"])
                asset["uri"] = str((source / asset["uri"]).resolve())
                row["assets"] = {"recording": asset}
                row["messages"] = [
                    dict(
                        id="user",
                        role="user",
                        content=[dict(id="utterance", type="text", text=text)],
                    ),
                    dict(
                        id="answer",
                        role="assistant",
                        content=[
                            dict(id="transcript", type="text", text=text),
                            dict(id="target", type="audio", media=dict(asset_id="recording")),
                        ],
                    ),
                ]
                conditions = []
                if len(rows) % 2:
                    reference = next(
                        (r for r in groups[sample.provenance.group_id] if r.id != sample.id), None
                    )
                    if reference is not None:
                        ref = reference.assets["recording"].model_dump(mode="json")
                        ref["uri"] = str((source / ref["uri"]).resolve())
                        row["assets"]["reference"] = ref
                        conditions.append(
                            dict(kind="reference_audio", media=dict(asset_id="reference"))
                        )
                row["predictions"] = [
                    dict(
                        id="p0",
                        target_message_id="answer",
                        target_block_ids=["transcript", "target"],
                        conditions=conditions,
                    )
                ]
                row["speech_pairs"] = [dict(text_block_id="transcript", audio_block_id="target")]
                rows.append(CanonicalSample.model_validate(row))
            if len(rows) != limit:
                raise ValueError("insufficient utterances/references in supplied ASR split")
            (root / f"{split}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))
        assert not groups_seen["train"] & groups_seen["validation"]
        write_json(
            root / "dataset_info.json",
            dict(
                source=str(source.resolve()),
                source_subset="LibriSpeech dev-clean pilot",
                license="CC BY 4.0",
                counts={"train": train_count or count, "validation": validation_count or count},
                speaker_disjoint=True,
                quality_benchmark=False,
                text_case=text_case,
                source_manifests={
                    split: sha256(source / f"{split}.jsonl") for split in ("train", "validation")
                },
            ),
        )
        write_json(root / "text-normalization.json", text_changes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--train-count", type=int)
    p.add_argument("--validation-count", type=int)
    p.add_argument(
        "--text-case",
        choices=("original", "sentence"),
        default="original",
        help="Explicit casing ablation for uppercase transcripts; sentence mode loses acronym/name capitalization",
    )
    args = p.parse_args()
    if any(n is not None and n < 2 for n in (args.count, args.train_count, args.validation_count)):
        p.error("count must be at least two to cover unconditioned and reference modes")
    prepare(
        args.source,
        args.output,
        args.count,
        train_count=args.train_count,
        validation_count=args.validation_count,
        text_case=args.text_case,
    )
