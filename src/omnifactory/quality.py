"""Paired generated-speech evaluation, with explicit scorer identity and no MOS proxy."""

import json
from pathlib import Path

from omnifactory.artifacts import sha256
from omnifactory.data.io import atomic_writer
from omnifactory.errors import OmniFactoryError
from omnifactory.training.asr_metrics import corpus_wer


def compare_transcripts(baseline, tuned, output, *, artifact_id, scorer):
    """Rows bind id, reference, ASR prediction and actual generated waveform hash."""

    def load(path):
        rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        keyed = {}
        for row in rows:
            key = row["sample_id"]
            if key in keyed or not row.get("audio_sha256") or not row.get("reference"):
                raise OmniFactoryError(
                    "E_QUALITY_PAIR", "duplicate/missing identity, waveform hash or reference"
                )
            audio = Path(row["audio"])
            if not audio.is_absolute():
                audio = Path(path).parent / audio
            if sha256(audio) != row["audio_sha256"]:
                raise OmniFactoryError("E_QUALITY_HASH", f"waveform changed for {key}")
            keyed[key] = row
        if not keyed:
            raise OmniFactoryError("E_QUALITY_PAIR", "empty quality set")
        return keyed

    a, b = load(baseline), load(tuned)
    if a.keys() != b.keys() or any(a[k]["reference"] != b[k]["reference"] for k in a):
        raise OmniFactoryError(
            "E_QUALITY_PAIR", "baseline and tuned cases/references must match exactly"
        )
    before, after = corpus_wer(a.values()), corpus_wer(b.values())
    report = dict(
        status="passed",
        artifact_id=artifact_id,
        scorer=scorer,
        baseline=before,
        tuned=after,
        wer_delta=after["wer"] - before["wer"],
        improvement_measured=after["wer"] < before["wer"],
        normalization="English NFKC lowercase; punctuation removed; no number expansion",
        scope="ASR-based generated-speech intelligibility proxy; not human MOS, speaker similarity or general quality",
        inputs={str(Path(p).resolve()): sha256(p) for p in (baseline, tuned)},
    )
    with atomic_writer(Path(output)) as stream:
        stream.write(json.dumps(report, indent=2))
    return report
