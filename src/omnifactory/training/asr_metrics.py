"""Explicit English pilot normalization and corpus word edit rate.

Lowercase Unicode NFKC text; preserve ASCII apostrophes, replace other punctuation
with spaces. Numbers are not expanded. This is not the official LibriSpeech scorer.
"""

import re
import unicodedata


def words(text):
    normalized = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.sub(r"[^a-z0-9'\s]", " ", normalized).split()


def word_errors(reference, prediction):
    ref, hyp = words(reference), words(prediction)
    if not ref:
        raise ValueError("WER requires nonempty normalized references")
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        row = [i]
        for j, candidate in enumerate(hyp, 1):
            row.append(min(previous[j] + 1, row[-1] + 1, previous[j - 1] + (word != candidate)))
        previous = row
    return {"errors": previous[-1], "reference_words": len(ref)}


def corpus_wer(rows):
    counts = [word_errors(row["reference"], row["prediction"]) for row in rows]
    errors = sum(r["errors"] for r in counts)
    references = sum(r["reference_words"] for r in counts)
    if not references:
        raise ValueError("empty evaluation set")
    return {
        "wer": errors / references,
        "errors": errors,
        "reference_words": references,
        "examples": len(counts),
    }
