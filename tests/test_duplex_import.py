"""Timing integrity and synchronized physical samples are importer invariants."""

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("soundfile")
import numpy as np
import soundfile as sf

pytest.importorskip("sentencepiece")
import sentencepiece as spm

from omnifactory.data.io import iter_samples
from omnifactory.datasets.duplex_import import Word, import_duplex, project_words
from omnifactory.errors import OmniFactoryError


@pytest.fixture
def tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("hello there how are you today we should start now okay\n" * 40))
    prefix = str(tmp_path / "tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=30,
        hard_vocab_limit=False,
        minloglevel=2,
        model_type="bpe",
        bos_id=-1,
        eos_id=-1,
    )
    return Path(prefix + ".model")


def inputs(tmp_path, tokenizer, *, stereo=False):
    rate = 16000
    t = np.arange(rate * 4) / rate
    a = (0.1 * np.sin(2 * np.pi * 237 * t)).astype("float32")
    b = (0.2 * np.sin(2 * np.pi * 390 * t)).astype("float32")
    row = dict(
        id="case",
        dataset="test",
        group_id="meeting",
        split="train",
        license="test-only",
        source_url="fixture",
        speaker_ids=["a", "b"],
        start_us=1000000,
        end_us=3000000,
        transcript="hello there",
    )
    sp = spm.SentencePieceProcessor(str(tokenizer))
    ids = sp.encode(row["transcript"])
    row.update(token_ids=ids, token_times_us=[1000000 + i * 80000 for i in range(len(ids))])
    if stereo:
        sf.write(tmp_path / "stereo.wav", np.stack([a, b], -1), rate, subtype="FLOAT")
        row.update(stereo_audio="stereo.wav", user_channel=0, agent_channel=1)
    else:
        sf.write(tmp_path / "user.wav", a, rate, subtype="FLOAT")
        sf.write(tmp_path / "agent.wav", b, rate, subtype="FLOAT")
        row.update(user_audio="user.wav", agent_audio="agent.wav")
    return row, a, b


def run(tmp_path, tokenizer, rows):
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return import_duplex(source, tmp_path / "output", tokenizer)


@pytest.mark.parametrize("stereo", [False, True])
def test_import_retains_exact_common_crop_and_token_grid(tmp_path, tokenizer, stereo):
    row, a, b = inputs(tmp_path, tokenizer, stereo=stereo)
    run(tmp_path, tokenizer, [row])
    sample = next(iter_samples(tmp_path / "output/train.jsonl"))
    for name, expected in (("user", a), ("agent", b)):
        actual, rate = sf.read(tmp_path / "output" / sample.assets[name].uri, dtype="float32")
        np.testing.assert_array_equal(actual, expected[16000:48000])
        assert rate == 16000
    alignment = json.loads((tmp_path / "output/text-alignment.json").read_text())[0]
    assert alignment["frames"] == list(range(len(row["token_ids"])))
    assert alignment["audio_sha256"] == sample.assets["agent"].sha256
    report = json.loads((tmp_path / "output/import-report.json").read_text())
    assert report["records"][0]["sources"]["agent"]["start_sample"] == 16000


@pytest.mark.parametrize(
    "problem",
    ["same_channel", "duration", "frame_collision", "wrong_tokens", "boundary", "same_audio"],
)
def test_invalid_sources_and_timing_never_publish(tmp_path, tokenizer, problem):
    row, _, _ = inputs(tmp_path, tokenizer, stereo=problem == "same_channel")
    if problem == "same_channel":
        row["agent_channel"] = 0
    elif problem == "duration":
        sf.write(tmp_path / "user.wav", np.zeros(100), 16000)
    elif problem == "frame_collision":
        row["token_times_us"][1] = row["token_times_us"][0] + 1
    elif problem == "wrong_tokens":
        row["token_ids"][0] = 0
    elif problem == "boundary":
        row["start_us"] = 1000001
    else:
        row["agent_audio"] = row["user_audio"]
    with pytest.raises((ValueError, OmniFactoryError)):
        run(tmp_path, tokenizer, [row])
    assert not (tmp_path / "output").exists()


def test_speaker_and_recording_leaks_rejected(tmp_path, tokenizer):
    row, _, _ = inputs(tmp_path, tokenizer)
    second = dict(row, id="other", split="validation", group_id="other-meeting")
    with pytest.raises(OmniFactoryError, match="E_SPLIT_LEAK"):
        run(tmp_path, tokenizer, [row, second])
    second["speaker_ids"] = ["c", "d"]
    with pytest.raises(OmniFactoryError, match="recording bytes"):
        run(tmp_path, tokenizer, [row, second])


def test_word_projection_uses_complete_tokenization_and_rejects_density(tmp_path, tokenizer):
    sp = spm.SentencePieceProcessor(str(tokenizer))
    words = [
        Word(text="hello", start_us=1000000, end_us=2000000),
        Word(text="there", start_us=2000000, end_us=3000000),
    ]
    ids, frames, info = project_words("hello there", words, sp, start_us=1000000, end_us=3000000)
    assert ids == tuple(sp.encode("hello there"))
    assert all(a < b for a, b in zip(frames, frames[1:]))
    assert info["token_timing_is_estimated"]
    with pytest.raises(OmniFactoryError, match="density"):
        project_words(
            "hello there",
            [Word(text="hello", start_us=0, end_us=1), Word(text="there", start_us=1, end_us=2)],
            sp,
            start_us=0,
            end_us=80000,
        )
    with pytest.raises(OmniFactoryError, match="transcript"):
        project_words("hello, there", words, sp, start_us=1000000, end_us=3000000)
    with pytest.raises(OmniFactoryError, match="crop"):
        project_words("hello there", words, sp, start_us=1000001, end_us=3000000)
