"""Prepare auditable two-channel AMI excerpts for PersonaPlex execution validation.

These are two headset channels of four-person meetings, not a clean two-person
conversation benchmark. All participant identities and omitted-channel activity
are reported. No synthetic speech, concatenation, time warping or clock repair.
"""

import argparse
import hashlib
import json
import tempfile
import zipfile
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree

from omnifactory.artifacts import sha256
from omnifactory.datasets.duplex_import import FRAME_US, import_duplex
from omnifactory.training.artifacts import write_json

ARCHIVE_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
SOURCE = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


def words_from_xml(data):
    words = []
    for element in ElementTree.fromstring(data):
        if element.tag != "w":
            continue
        if element.attrib.get("punc") == "true":
            if words:
                words[-1]["text"] += element.text or ""
            continue
        if not element.text or not element.text.strip():
            raise ValueError("empty AMI lexical token")
        words.append(
            dict(
                text=element.text,
                start_us=int(Decimal(element.attrib["starttime"]) * 1000000),
                end_us=int(Decimal(element.attrib["endtime"]) * 1000000),
            )
        )
    return words


def safe_cut(requested, words, *, maximum, radius=5000000):
    candidates = range(
        max(0, (requested - radius) // FRAME_US), min(maximum, requested + radius) // FRAME_US + 1
    )
    for frame in sorted(candidates, key=lambda frame: (abs(frame * FRAME_US - requested), frame)):
        time = frame * FRAME_US
        if not any(w["start_us"] < time < w["end_us"] for w in words):
            return time
    raise ValueError(f"no common word boundary within {radius} us of {requested}")


def prepare(annotations, signals, output, tokenizer, *, signal_suffix="", word_slack_us=80000):
    annotations, signals, output, tokenizer = map(Path, (annotations, signals, output, tokenizer))
    if sha256(annotations) != ARCHIVE_SHA256:
        raise ValueError("annotations must match pinned AMI manual 1.6.2 archive")
    rows, provenance = [], []
    with zipfile.ZipFile(annotations) as archive:
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        for meeting, split in (("ES2002a", "train"), ("ES2003a", "validation")):
            metadata = next(m for m in meetings if m.attrib.get("observation") == meeting)
            speakers = {s.attrib["nxt_agent"]: s.attrib for s in metadata if s.tag == "speaker"}
            words = {
                s: words_from_xml(archive.read(f"words/{meeting}.{s}.words.xml")) for s in speakers
            }
            all_words = [w for items in words.values() for w in items]
            desired = (
                (48000000, 80000000, 112000000, 144000000, 180000000)
                if split == "train"
                else (0, 60000000, 180000000)
            )
            cuts = [safe_cut(t, all_words, maximum=180000000) for t in desired]
            if any(a >= b for a, b in zip(cuts, cuts[1:])):
                raise ValueError("safe boundaries crossed")
            windows = (
                list(zip(cuts, cuts[1:]))
                if split == "train"
                else [(cuts[0], cuts[1]), (cuts[0], cuts[-1])]
            )
            for index, (start, end) in enumerate(windows):
                selected = [w for w in words["B"] if start <= w["start_us"] and w["end_us"] <= end]
                if not selected:
                    raise ValueError("selected excerpt has no agent transcript")
                row = dict(
                    id=f"{meeting}-{index}",
                    dataset="ami-two-headset-excerpts-v1",
                    group_id=meeting[:-1],
                    split=split,
                    license="CC BY 4.0",
                    source_url=SOURCE,
                    speaker_ids=[v["global_name"] for v in speakers.values()],
                    user_audio=str(
                        (
                            signals
                            / f"{meeting}.Headset-{speakers['A']['channel']}{signal_suffix}.wav"
                        ).resolve()
                    ),
                    agent_audio=str(
                        (
                            signals
                            / f"{meeting}.Headset-{speakers['B']['channel']}{signal_suffix}.wav"
                        ).resolve()
                    ),
                    start_us=start,
                    end_us=end,
                    transcript=" ".join(w["text"] for w in selected),
                    alignment="word_uniform",
                    words=selected,
                    persona="You are participating in a project meeting.",
                )
                rows.append(row)
                provenance.append(
                    dict(
                        id=row["id"],
                        meeting=meeting,
                        user_agent="A",
                        target_agent="B",
                        channels={s: v["channel"] for s, v in speakers.items()},
                        requested_boundaries_us=list(desired),
                        actual_start_us=start,
                        actual_end_us=end,
                        user_word_count=sum(start <= w["start_us"] < end for w in words["A"]),
                        target_word_count=len(selected),
                        omitted_channel_word_counts={
                            s: sum(start <= w["start_us"] < end for w in ws)
                            for s, ws in words.items()
                            if s not in ("A", "B")
                        },
                        annotation_sha256={
                            s: hashlib.sha256(
                                archive.read(f"words/{meeting}.{s}.words.xml")
                            ).hexdigest()
                            for s in speakers
                        },
                    )
                )
    # Source audio paths are absolute, so a temporary source manifest is portable.
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "recordings.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = import_duplex(
            source, output, tokenizer, word_slack_us=word_slack_us, max_seconds=180
        )
    write_json(
        output / "ami-provenance.json",
        dict(
            archive_url=SOURCE,
            archive_sha256=ARCHIVE_SHA256,
            scope="two real synchronized channels of four-person meetings; omitted participants and crosstalk remain limitations; execution validation only",
            word_timing="AMI manual annotations; subword positions estimated explicitly with word_uniform",
            overlapping_validation_excerpts=True,
            independent_quality_benchmark=False,
            records=provenance,
        ),
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "signals", "output", "tokenizer"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--signal-suffix", default="")
    p.add_argument("--word-slack-us", type=int, default=80000)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
