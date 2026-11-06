import json

import pytest
from typer.testing import CliRunner

from omnifactory.cli import app
from omnifactory.data.io import iter_samples
from omnifactory.datasets.importers import import_alpaca, import_tts
from omnifactory.errors import OmniFactoryError


def test_dataset_catalog_available_without_training():
    result = CliRunner().invoke(app, ["datasets", "list"])
    assert result.exit_code == 0
    assert {r["name"] for r in json.loads(result.stdout)} == {
        "librispeech-dev-clean-pilot",
        "image-demo",
        "flow-demo",
        "mimo-audio-demo",
        "omni-demo",
        "omni-av-demo",
        "qwen-native-speech",
        "contract-examples",
    }


def test_alpaca_import_tracks_same_prompt_group_across_splits(tmp_path):
    source = tmp_path / "alpaca.jsonl"
    source.write_text(
        json.dumps({"instruction": "Answer the question", "input": "What is 1+1?", "output": "2"})
        + "\n"
    )
    a, b = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    import_alpaca(source, a, name="local", split="train")
    import_alpaca(source, b, name="local", split="validation")
    train = next(iter_samples(a))
    validation = next(iter_samples(b))
    assert train.provenance.group_id == validation.provenance.group_id
    assert train.messages[0].content[0].text == "Answer the question\nWhat is 1+1?"


def test_import_rejects_silent_media_loss_and_preserves_output(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "out.jsonl"
    source.write_text(
        json.dumps({"instruction": "Describe", "output": "A dog", "images": ["dog.png"]}) + "\n"
    )
    output.write_text("original")
    with pytest.raises(OmniFactoryError, match="E_IMPORT_FIELDS"):
        import_alpaca(source, output, name="local")
    assert output.read_text() == "original"


def test_tts_import_preserves_group_and_blocks_reference_leak(tmp_path):
    from pydantic import ValidationError

    target = tmp_path / "target.wav"
    target.write_bytes(b"placeholder audio bytes: decoding happens in prepare")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"different reference audio bytes")
    source, output = tmp_path / "pairs.jsonl", tmp_path / "canonical.jsonl"
    raw = dict(
        id="utterance-1",
        group_id="speaker-1",
        text="Hello",
        audio="target.wav",
        reference_audio="reference.wav",
    )
    source.write_text(json.dumps(raw))
    import_tts(source, output, name="tts")
    sample = next(iter_samples(output))
    assert sample.provenance.group_id == "speaker-1"
    assert sample.messages[0].content[0].text == sample.messages[1].content[0].text == "Hello"
    assert sample.assets["audio"].uri == str(target)
    assert sample.speech_pairs[0].audio_block_id == "wave"
    reference.write_bytes(target.read_bytes())
    original = output.read_bytes()
    with pytest.raises(ValidationError, match="E_TARGET_LEAK"):
        import_tts(source, output, name="tts")
    assert output.read_bytes() == original
