import json

import pytest
from pydantic import ValidationError

from omnifactory import CanonicalSample, OmniDataProcessor
from omnifactory.data.io import atomic_writer, iter_samples, validate_dataset
from omnifactory.data.media import LocalMediaResolver
from omnifactory.errors import OmniFactoryError


def test_all_examples_roundtrip(examples_path):
    samples = list(iter_samples(examples_path))
    assert len(samples) == 4
    for sample in samples:
        assert CanonicalSample.model_validate_json(sample.model_dump_json()) == sample


def test_inference_omits_gold_text_and_media(raw_sample):
    processor = OmniDataProcessor()
    sample = processor.normalize(raw_sample)
    view = processor.to_inference(sample, "p0", ("text", "audio"))
    assert set(view.assets) == {"question"}
    assert [m.id for m in view.messages] == ["m0"]
    assert "answer.wav" not in view.model_dump_json()
    assert "请先确认" not in view.model_dump_json()


def test_multiturn_predictions_preserve_only_previous_answers(examples_path):
    sample = list(iter_samples(examples_path))[2]
    p = OmniDataProcessor()
    assert [m.id for m in p.to_inference(sample, "p0").messages] == ["m0"]
    assert [m.id for m in p.to_inference(sample, "p1").messages] == ["m0", "m1", "m2"]
    assert "电源开关" not in p.to_inference(sample, "p1").model_dump_json()


def test_reference_preserved_but_target_removed(reference_sample):
    p = OmniDataProcessor()
    view = p.to_inference(p.normalize(reference_sample), "p0")
    assert set(view.assets) == {"voice"}
    assert view.conditions[0].kind == "reference_audio"
    assert "response.wav" not in view.model_dump_json()


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("wrong_asset", "E_ASSET"),
        ("wrong_target", "E_TARGET"),
        ("duplicate_block", "E_DUPLICATE_ID"),
        ("bad_time", "E_TIME_RANGE"),
        ("cycle", "E_ASSET_LINEAGE"),
    ],
)
def test_invalid_references_rejected(raw_sample, mutation, code):
    if mutation == "wrong_asset":
        raw_sample["messages"][0]["content"][0]["media"]["asset_id"] = "missing"
    elif mutation == "wrong_target":
        raw_sample["predictions"][0]["target_block_ids"] = ["question-audio"]
    elif mutation == "duplicate_block":
        raw_sample["messages"][1]["content"][0]["id"] = "question-audio"
    elif mutation == "bad_time":
        raw_sample["messages"][0]["content"][0]["media"]["end_us"] = 999999999
    else:
        raw_sample["assets"]["question"]["derived_from"] = "answer"
        raw_sample["assets"]["answer"]["derived_from"] = "question"
    with pytest.raises(ValidationError, match=code):
        CanonicalSample.model_validate(raw_sample)


def test_video_requires_explicit_audio_mode(examples_path):
    raw = json.loads(examples_path.read_text().splitlines()[1])
    del raw["messages"][0]["content"][1]["audio_mode"]
    with pytest.raises(ValidationError, match="audio_mode"):
        CanonicalSample.model_validate(raw)


def test_ref_target_hash_alias_rejected(reference_sample):
    for asset in reference_sample["assets"].values():
        asset["sha256"] = "a" * 64
    with pytest.raises(ValidationError, match="E_TARGET_LEAK"):
        CanonicalSample.model_validate(reference_sample)


def test_disjoint_reference_target_slices_allowed(reference_sample):
    reference_sample["assets"]["response"]["uri"] = reference_sample["assets"]["voice"]["uri"]
    reference_sample["predictions"][0]["conditions"][0]["media"]["end_us"] = 1000
    reference_sample["messages"][1]["content"][1]["media"]["start_us"] = 1000
    CanonicalSample.model_validate(reference_sample)


def test_unknown_fields_rejected(raw_sample):
    raw_sample["silently_ignored"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CanonicalSample.model_validate(raw_sample)


def test_understanding_plan_reports_ignored_audio_target(raw_sample):
    p = OmniDataProcessor()
    plan = p.plan(p.normalize(raw_sample), model="qwen2_5_omni_3b")
    assert not plan.training_ready
    assert plan.predictions[0].target_block_ids == ("answer-text",)
    assert plan.predictions[0].ignored_target_block_ids == ("answer-audio",)
    assert plan.diagnostics[0].code == "W_UNUSED_TARGET"


def test_minicpm_has_different_stage_ids(raw_sample):
    p = OmniDataProcessor()
    plan = p.plan(p.normalize(raw_sample), model="minicpm_o_4_5")
    assert plan.stages == ("llm", "tts", "code2wav")


def test_speech_plan_requires_explicit_pair(raw_sample):
    raw_sample["speech_pairs"] = []
    p = OmniDataProcessor()
    with pytest.raises(OmniFactoryError, match="E_SPEECH_PAIR"):
        p.plan(p.normalize(raw_sample), model="qwen3_omni", task="speech_sft")


def test_speech_plan_is_not_training_support(raw_sample):
    p = OmniDataProcessor()
    plan = p.plan(p.normalize(raw_sample), model="qwen3_omni", task="speech_sft")
    assert plan.status == "semantic_plan_only"
    assert any("codec" in step for step in plan.pending)


def test_unknown_task_rejected(raw_sample):
    p = OmniDataProcessor()
    with pytest.raises(OmniFactoryError, match="E_TASK"):
        p.plan(p.normalize(raw_sample), model="qwen2_5_omni_3b", task="unsupported_task")


def test_dataset_checks_split_leaks_and_duplicates(raw_sample, tmp_path):
    first = json.dumps(raw_sample)
    raw_sample["id"] = "second"
    raw_sample["provenance"]["split"] = "test"
    path = tmp_path / "bad.jsonl"
    path.write_text(first + "\n" + first + "\n" + json.dumps(raw_sample) + "\n{bad\n")
    report = validate_dataset(path)
    assert (report.records, report.valid, report.invalid) == (4, 1, 3)
    assert any("E_DUPLICATE_SAMPLE" in e for e in report.errors)
    assert any("E_SPLIT_LEAK" in e for e in report.errors)


def test_empty_dataset_report(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    assert "E_EMPTY_DATASET" in validate_dataset(path).errors[0]


def test_manifest_hashes_actual_bytes_and_detects_change(raw_sample, tmp_path):
    for key, asset in raw_sample["assets"].items():
        asset["uri"] = key
        (tmp_path / key).write_bytes(key.encode())
    resolver = LocalMediaResolver(tmp_path)
    manifest = resolver.manifest(CanonicalSample.model_validate(raw_sample))
    assert manifest.verification == "file_bytes_only"
    assert manifest.assets["question"].size_bytes == len(b"question")
    assert manifest == resolver.manifest(manifest.sample)
    (tmp_path / "question").write_bytes(b"different")
    with pytest.raises(OmniFactoryError, match="E_MEDIA_HASH"):
        resolver.manifest(manifest.sample)


def test_manifest_detects_hidden_reference_alias(reference_sample, tmp_path):
    for key, asset in reference_sample["assets"].items():
        asset["uri"] = key
        (tmp_path / key).write_bytes(b"identical bytes at different paths")
    with pytest.raises(ValidationError, match="E_TARGET_LEAK"):
        LocalMediaResolver(tmp_path).manifest(CanonicalSample.model_validate(reference_sample))


def test_remote_media_not_downloaded(raw_sample, tmp_path):
    raw_sample["assets"]["question"]["uri"] = "https://example.com/question.wav"
    with pytest.raises(OmniFactoryError, match="E_MEDIA_URI"):
        LocalMediaResolver(tmp_path).manifest(CanonicalSample.model_validate(raw_sample))


def test_atomic_writer_preserves_previous_file_on_late_failure(tmp_path):
    target = tmp_path / "manifest.jsonl"
    target.write_text("original")
    with pytest.raises(ValueError):
        with atomic_writer(target) as stream:
            stream.write("partial result")
            raise ValueError("later sample failed")
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]
