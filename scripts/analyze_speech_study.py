"""Freeze a development choice and evaluate paired multi-seed confirmation.

Resample speakers, averaging training seeds within each utterance first. Do not
count repeated seeds as extra independent test examples. Audio metrics remain
energy diagnostics; WER remains an ASR proxy.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from omnifactory.artifacts import sha256
from omnifactory.training.artifacts import write_json
from omnifactory.training.asr_metrics import corpus_wer, word_errors


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checked_scores(root):
    report = read(root / "report.json")
    if report["status"] != "completed":
        raise ValueError("scoring must be complete")
    for name, digest in report["evidence"].items():
        if sha256(root / name) != digest:
            raise ValueError("score evidence changed")
    return report, {k: rows(root / f"{k}-transcripts.jsonl") for k in report["metrics"]}


def tails(records):
    result = {}
    for threshold in ("-50", "-60"):
        values = [r["trailing_low_energy_seconds"][threshold] for r in records]
        result[threshold] = dict(
            mean_seconds=sum(values) / len(values),
            max_seconds=max(values),
            over_two_seconds=sum(v > 2 for v in values),
        )
    return result


def paired(left, right, samples, *, repeats=5000, bootstrap_seed=42):
    if not left or len(left) != len(right) or repeats < 100:
        raise ValueError("paired seed lists and >=100 repeats required")
    ids = [s["id"] for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation sample")
    expected = [(r["sample_id"], r["reference"]) for r in left[0]]
    if [x[0] for x in expected] != ids:
        raise ValueError("missing/misordered evaluation samples")
    for records in [*left, *right]:
        if [(r["sample_id"], r["reference"]) for r in records] != expected:
            raise ValueError("unpaired references")
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        group = sample["provenance"]["group_id"]
        if not group:
            raise ValueError("speaker identities required")
        groups[group].append(i)
    if len(groups) < 2:
        raise ValueError("speaker bootstrap requires at least two speakers")
    n = len(left)
    counts = [
        [word_errors(a["reference"], a["prediction"]) for a in records]
        for records in [*left, *right]
    ]
    words = [c["reference_words"] for c in counts[0]]
    delta = [
        sum(counts[n + j][i]["errors"] - counts[j][i]["errors"] for j in range(n)) / n
        for i in range(len(ids))
    ]

    def rate(indices):
        return sum(delta[i] for i in indices) / sum(words[i] for i in indices)

    rng = random.Random(bootstrap_seed)
    speakers = sorted(groups)
    draws = sorted(
        rate([i for s in rng.choices(speakers, k=len(speakers)) for i in groups[s]])
        for _ in range(repeats)
    )
    loo = [rate([i for s in speakers if s != omit for i in groups[s]]) for omit in speakers]
    return dict(
        cases=len(ids),
        speakers=len(speakers),
        training_seed_pairs=n,
        delta_wer=rate(list(range(len(ids)))),
        speaker_bootstrap_95_percentile=[
            draws[int(0.025 * (repeats - 1))],
            draws[int(0.975 * (repeats - 1))],
        ],
        seed_deltas=[
            corpus_wer(b)["wer"] - corpus_wer(a)["wer"] for a, b in zip(left, right, strict=True)
        ],
        leave_one_speaker_out_delta_range=[min(loo), max(loo)],
        largest_case_contributions=[
            dict(sample_id=ids[i], mean_delta_errors=delta[i])
            for i in sorted(range(len(ids)), key=lambda i: abs(delta[i]), reverse=True)[:10]
        ],
    )


def select(protocol_path, score_root, output):
    protocol = read(protocol_path)
    report, arms = checked_scores(score_root)
    identity = read(score_root / "identity.json")
    if identity["dataset_sha256"] != protocol["development"]["dataset_sha256"]:
        raise ValueError("development dataset changed")
    energy = read(score_root / "audio-diagnostics.json")
    names = list(protocol["candidates"])
    if set(arms) != set(names):
        raise ValueError("missing development candidates")
    ranking = []
    for i, name in enumerate(names):
        spec = protocol["candidates"][name]
        if sha256(Path(spec["config"])) != spec["config_sha256"]:
            raise ValueError("candidate configuration changed")
        t = tails(energy[name])["-50"]
        term = report["termination"][name]
        rank = [
            term["total"] - term["finished"],
            report["metrics"][name]["errors"],
            t["over_two_seconds"],
            t["mean_seconds"],
            i,
        ]
        ranking.append(
            dict(name=name, rank=rank, metrics=report["metrics"][name], tails=tails(energy[name]))
        )
    ranking.sort(key=lambda r: r["rank"])
    chosen = next(r["name"] for r in ranking if r["name"] != "control")
    if output.exists():
        raise ValueError("selection is immutable; choose a fresh study")
    write_json(
        output,
        dict(
            status="frozen",
            selected=chosen,
            best_overall=ranking[0]["name"],
            ranking=ranking,
            protocol_sha256=sha256(protocol_path),
            dev_report_sha256=sha256(score_root / "report.json"),
            script_sha256=sha256(Path(__file__)),
            scope="one predeclared challenger; no joint LR search; freeze before independent test generation",
        ),
    )


def confirm(protocol_path, selection_path, score_root, dataset, output):
    protocol, selection = read(protocol_path), read(selection_path)
    if selection["status"] != "frozen" or selection["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("selection/protocol mismatch")
    report, arms = checked_scores(score_root)
    samples = rows(dataset)
    identity = read(score_root / "identity.json")
    if identity["dataset_sha256"] != sha256(dataset):
        raise ValueError("confirmation dataset changed")
    if len(samples) != protocol["confirmation"]["cases"]:
        raise ValueError("confirmation coverage differs from protocol")
    if len({s["provenance"]["group_id"] for s in samples}) != protocol["confirmation"]["speakers"]:
        raise ValueError("confirmation speaker coverage differs from protocol")
    if any(s["provenance"]["split"] != "test" for s in samples):
        raise ValueError("confirmation requires test provenance")
    seeds = protocol["confirmation"]["training_seeds"]
    controls = [f"control_s{s}" for s in seeds]
    selected = [f"selected_s{s}" for s in seeds]
    required = {*controls, *selected, "base", "selected_export", "groundtruth", "native_reference"}
    if set(arms) != required:
        raise ValueError("incomplete confirmation arms")
    primary = paired([arms[k] for k in controls], [arms[k] for k in selected], samples)
    vs_base = paired([arms["base"]] * len(seeds), [arms[k] for k in selected], samples)
    native_reference = paired([arms["native_reference"]], [arms[selected[0]]], samples)
    merge = paired([arms[selected[0]]], [arms["selected_export"]], samples)
    energy = read(score_root / "audio-diagnostics.json")
    summaries = {k: tails(v) for k, v in energy.items()}

    def mean_tail(labels):
        return sum(summaries[k]["-50"]["mean_seconds"] for k in labels) / len(labels)

    def long_tail(labels):
        return sum(summaries[k]["-50"]["over_two_seconds"] for k in labels) / len(labels)

    gates = dict(
        paired_speaker_interval_below_zero=primary["speaker_bootstrap_95_percentile"][1] < 0,
        every_training_seed_improves=all(d < 0 for d in primary["seed_deltas"]),
        improvement_survives_omitting_any_speaker=primary["leave_one_speaker_out_delta_range"][1]
        < 0,
        improvement_over_base=vs_base["speaker_bootstrap_95_percentile"][1] < 0,
        all_selected_outputs_finish=all(
            report["termination"][k]["finished"] == len(samples)
            for k in [*selected, "selected_export"]
        ),
        mean_tail_not_worse=mean_tail(selected) <= mean_tail(controls),
        long_tail_count_not_worse=long_tail(selected) <= long_tail(controls),
        selected_beats_prior_native_reference=report["metrics"][selected[0]]["wer"]
        < report["metrics"]["native_reference"]["wer"],
        export_wer_retains_improvement=report["metrics"]["selected_export"]["wer"]
        < min(
            report["metrics"][controls[0]]["wer"],
            report["metrics"]["base"]["wer"],
            report["metrics"]["native_reference"]["wer"],
        ),
    )
    strata = {}
    for conditioned in (False, True):
        indices = [
            i
            for i, s in enumerate(samples)
            if any(c["kind"] == "reference_audio" for c in s["predictions"][0]["conditions"])
            == conditioned
        ]
        subset = [samples[i] for i in indices]
        strata["reference" if conditioned else "no_reference"] = paired(
            [[arms[k][i] for i in indices] for k in controls],
            [[arms[k][i] for i in indices] for k in selected],
            subset,
        )
    if output.exists():
        raise ValueError("confirmation report is immutable")
    write_json(
        output,
        dict(
            status="completed",
            stable_improvement_supported=all(gates.values()),
            gates=gates,
            selected=selection["selected"],
            primary=primary,
            vs_base=vs_base,
            vs_native_reference_seed42=native_reference,
            merge=merge,
            metrics=report["metrics"],
            energy=summaries,
            strata=strata,
            evidence={
                str(p.resolve()): sha256(p)
                for p in (
                    protocol_path,
                    selection_path,
                    score_root / "report.json",
                    dataset,
                    Path(__file__),
                )
            },
            scope="one frozen challenger; three training seeds with fixed generation RNG; speaker-cluster intervals; ASR/energy proxies, not perceptual ratings",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("select", "confirm"))
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--dataset", type=Path)
    a = p.parse_args()
    if a.mode == "select":
        select(a.protocol, a.scores, a.output)
    else:
        if not a.selection or not a.dataset:
            p.error("confirmation requires --selection and --dataset")
        confirm(a.protocol, a.selection, a.scores, a.dataset, a.output)


if __name__ == "__main__":
    main()
