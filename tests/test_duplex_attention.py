"""Temporal visibility actually changes native MiniCPM attention and gradients."""

import copy
import json
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
import torch

from omnifactory.adapters.duplex import (
    TokenEvent,
    TokenSchedule,
    bind_duplex,
    native_attention,
    token_fingerprint,
)
from omnifactory.adapters.minicpm import MiniCPMAdapter, load_minicpm
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.demo import create_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.prepare import prepare
from omnifactory.training.tiny_minicpm import create_tiny_minicpm


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("duplex")
    create_tiny_minicpm(root / "model")
    create_demo(root / "data")
    config = TrainConfig(
        model=str(root / "model"),
        adapter="minicpm_o_4_5",
        device=os.environ.get("OMNIFACTORY_TEST_DEVICE", "cpu"),
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        max_steps=1,
        components={"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.001}},
        budget={
            "max_examples": 1,
            "max_llm_tokens": 1024,
            "max_vision_patches": 128,
            "max_audio_frames": 1000,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 1000000,
            "max_stream_events": 100,
        },
    )
    sample = list(iter_samples(config.dataset))[2]
    raw = sample.model_dump(mode="json")
    block = next(b for b in raw["messages"][0]["content"] if b["type"] == "audio")
    raw["messages"][0]["content"].append({**block, "id": "future-mic"})
    sample = CanonicalSample.model_validate(raw)
    adapter = MiniCPMAdapter.from_config(config)
    encoded = adapter.encode(sample, "p0", contract_id="duplex")
    bounds = encoded.payload["audio_bounds"][0].tolist()
    n = encoded.payload["input_ids"].shape[1]
    segments = []
    cursor = 0
    for i, (start, end) in enumerate(bounds):
        if start > cursor:
            segments.append(
                TokenEvent(
                    id=f"context-{i}",
                    start_token=cursor,
                    end_token=start,
                    start_us=0,
                    end_us=1,
                    available_us=1,
                    direction="condition",
                    modality="text",
                )
            )
        segments.append(
            TokenEvent(
                id=f"mic-{i}",
                start_token=start,
                end_token=end,
                start_us=i * 2000000,
                end_us=i * 2000000 + 1000000,
                available_us=i * 2000000 + 1000000,
                direction="input",
                modality="audio",
            )
        )
        cursor = end
    if cursor < encoded.prompt_length - 1:
        segments.append(
            TokenEvent(
                id="header",
                start_token=cursor,
                end_token=encoded.prompt_length - 1,
                start_us=0,
                end_us=1,
                available_us=1,
                direction="condition",
                modality="text",
            )
        )
    segments.append(
        TokenEvent(
            id="reply",
            start_token=encoded.prompt_length - 1,
            end_token=n,
            start_us=1500000,
            end_us=2500000,
            available_us=2500000,
            direction="output",
            modality="text",
        )
    )
    schedule = TokenSchedule(
        sample_id=sample.id,
        prediction_id="p0",
        input_ids_sha256=token_fingerprint(encoded.payload["input_ids"]),
        events=tuple(segments),
    )
    return root, config, adapter, sample, encoded, schedule


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_native_future_audio_cannot_change_earlier_reply(runtime, attention):
    _, config, adapter, _, encoded, schedule = runtime
    model = load_minicpm(config.model_copy(update={"attention": attention})).eval()
    timed = bind_duplex(copy.deepcopy(encoded), schedule)
    row = adapter.to_device(timed.payload, model, config)
    base = adapter.forward(model, row).logits[:, encoded.prompt_length - 1 :]
    future = copy.deepcopy(row)
    future["audio_features"][1] = torch.randn_like(future["audio_features"][1]) * 10
    actual = adapter.forward(model, future).logits[:, encoded.prompt_length - 1 :]
    torch.testing.assert_close(base, actual, atol=1e-6, rtol=1e-6)
    past = copy.deepcopy(row)
    past["audio_features"][0] = torch.randn_like(past["audio_features"][0]) * 10
    assert not torch.allclose(
        base,
        adapter.forward(model, past).logits[:, encoded.prompt_length - 1 :],
        atol=1e-6,
        rtol=1e-6,
    )
    model.zero_grad()
    row["audio_features"] = row["audio_features"].detach().requires_grad_()
    adapter.forward(model, row).loss.backward()
    assert row["audio_features"].grad[1].abs().sum() == 0
    assert row["audio_features"].grad[0].abs().sum() > 0
    mask, _ = native_attention(row, dtype=torch.float32)
    start, end = encoded.payload["audio_bounds"][0][1]
    assert (mask[0, 0, encoded.prompt_length - 1 :, start:end] < 0).all()


def test_timing_fingerprint_and_budget_are_enforced(runtime):
    _, _, _, _, encoded, schedule = runtime
    with pytest.raises(OmniFactoryError, match="E_DUPLEX_FINGERPRINT"):
        bind_duplex(
            copy.deepcopy(encoded), schedule.model_copy(update={"input_ids_sha256": "0" * 64})
        )
    with pytest.raises(OmniFactoryError, match="E_ALIGNMENT_BUDGET"):
        bind_duplex(copy.deepcopy(encoded), schedule, max_pairs=1)


def test_timed_prepare_train_reload(runtime):
    root, config, adapter, sample, _, schedule = runtime
    dataset = root / "timed.jsonl"
    dataset.write_text(sample.model_dump_json() + "\n")
    timing = root / "timing.json"
    timing.write_text(json.dumps([schedule.model_dump(mode="json")]))
    cfg = TrainConfig.model_validate(
        {
            **config.model_dump(),
            "dataset": dataset,
            "dataset_root": config.dataset.parent,
            "duplex_schedule": timing,
        }
    )
    manifest = prepare(cfg, root / "prepared")
    assert manifest["format"] == "omnifactory-prepared-v4"
    assert all(entry["supervision"] for entry in manifest["entries"])
    result = train(cfg)
    model, loaded, _, _ = load_checkpoint(__import__("pathlib").Path(result["checkpoint"]))
    row = loaded.encode(sample, "p0", contract_id="reload").payload
    assert torch.isfinite(loaded.forward(model, loaded.to_device(row, model, config)).loss)


def test_clip_cannot_be_declared_available_before_its_measured_duration(runtime):
    _, _, _, _, encoded, schedule = runtime
    events = tuple(
        e.model_copy(update={"end_us": e.start_us + 1}) if e.id == "mic-0" else e
        for e in schedule.events
    )
    with pytest.raises(OmniFactoryError, match="E_DUPLEX_AUDIO"):
        bind_duplex(copy.deepcopy(encoded), schedule.model_copy(update={"events": events}))
