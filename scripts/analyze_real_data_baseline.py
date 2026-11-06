"""Describe paired baseline errors without treating small WER changes as proof.

Cluster bootstrap resamples speakers, preserving paired outputs and every
utterance of each selected speaker. Intervals are descriptive development-set
uncertainty, not a substitute for independent data or generation-seed repeats.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from omnifactory.artifacts import sha256
from omnifactory.training.artifacts import write_json
from omnifactory.training.asr_metrics import word_errors


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(baseline, dataset, *, repeats=5000, seed=42):
    samples = read_rows(dataset)
    by_id = {s["id"]: s for s in samples}
    if len(by_id) != len(samples) or repeats < 100:
        raise ValueError("requires unique sample IDs and >=100 bootstrap repeats")
    suffix = "transcripts" if (baseline / "base-transcripts.jsonl").exists() else "predictions"
    files = [baseline / f"{arm}-{suffix}.jsonl" for arm in ("base", "adapter", "export")]
    arms = {arm: read_rows(p) for arm, p in zip(("base", "adapter", "export"), files, strict=True)}
    expected = [(r["sample_id"], r["reference"]) for r in arms["base"]]
    if len({r[0] for r in expected}) != len(expected) or {r[0] for r in expected} != set(by_id):
        raise ValueError("requires one output per supplied sample and complete coverage")
    for records in arms.values():
        if [(r["sample_id"], r["reference"]) for r in records] != expected:
            raise ValueError("paired identity/reference/order mismatch")
    cases = []
    for i, (sample_id, reference) in enumerate(expected):
        sample = by_id[sample_id]
        counts = {
            arm: word_errors(reference, records[i]["prediction"]) for arm, records in arms.items()
        }
        cases.append(
            dict(
                sample_id=sample_id,
                speaker=sample["provenance"]["group_id"],
                reference_audio=any(
                    c["kind"] == "reference_audio" for c in sample["predictions"][0]["conditions"]
                ),
                reference_words=counts["base"]["reference_words"],
                errors={arm: c["errors"] for arm, c in counts.items()},
            )
        )
    groups = defaultdict(list)
    for row in cases:
        groups[row["speaker"]].append(row)
    pairs = {}
    for left, right in [("base", "adapter"), ("adapter", "export")]:
        rng = random.Random(seed)
        speakers = sorted(groups)
        deltas = []
        for _ in range(repeats):
            sampled = [r for group in rng.choices(speakers, k=len(speakers)) for r in groups[group]]
            deltas.append(
                sum(r["errors"][right] - r["errors"][left] for r in sampled)
                / sum(r["reference_words"] for r in sampled)
            )
        deltas.sort()
        changes = [r["errors"][right] - r["errors"][left] for r in cases]
        pairs[left + "->" + right] = dict(
            delta_wer=sum(changes) / sum(r["reference_words"] for r in cases),
            speaker_bootstrap_95_percentile=[
                deltas[int(0.025 * (repeats - 1))],
                deltas[int(0.975 * (repeats - 1))],
            ],
            improved=sum(x < 0 for x in changes),
            unchanged=sum(x == 0 for x in changes),
            worsened=sum(x > 0 for x in changes),
        )
    strata = {}
    for conditioned in (False, True):
        selected = [r for r in cases if r["reference_audio"] == conditioned]
        if selected:
            total = sum(r["reference_words"] for r in selected)
            strata["reference" if conditioned else "no_reference"] = dict(
                cases=len(selected),
                reference_words=total,
                wer={arm: sum(r["errors"][arm] for r in selected) / total for arm in arms},
            )
    return dict(
        cases=cases,
        pairs=pairs,
        strata=strata,
        speakers=len(groups),
        bootstrap_repeats=repeats,
        seed=seed,
        evidence={str(p.resolve()): sha256(p) for p in [dataset, *files]},
        scope="paired speaker-cluster bootstrap on existing development outputs; not perceptual quality or independent replication",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a new output; preserve prior reports")
    write_json(args.output, analyze(args.baseline, args.dataset))


if __name__ == "__main__":
    main()
