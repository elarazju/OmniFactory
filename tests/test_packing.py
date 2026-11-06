# ruff: noqa: E402
"""Native multimodal mRoPE, boundary isolation and parameter-gradient equivalence."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from omnifactory.adapters.packing import packed_batch
from omnifactory.adapters.qwen2_5 import Qwen25Adapter, load_thinker
from omnifactory.data.io import iter_samples
from omnifactory.datasets.demo import create_demo
from omnifactory.planning.batching import ModalBatchPlanner, SampleCost
from omnifactory.training.config import TrainConfig
from omnifactory.training.tiny import create_tiny


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    create_tiny(tmp_path / "model")
    create_demo(tmp_path / "data")
    config = TrainConfig(
        model=str(tmp_path / "model"),
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "run",
        max_length=1024,
        components={"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.001}},
        budget=dict(
            max_examples=4,
            max_llm_tokens=4096,
            max_vision_patches=1024,
            max_audio_frames=1000,
            max_codec_frames=1,
            max_decoded_media_bytes=1000000,
        ),
    )
    adapter = Qwen25Adapter.from_config(config)
    rows = [
        adapter.encode(s, "p0", contract_id="test").payload
        for s in list(iter_samples(config.dataset))[:3]
    ]
    return config, adapter, rows


@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_packing_matches_independent_multimodal_gradients(fixture, checkpointing):
    config, adapter, rows = fixture
    model = load_thinker(config).train()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    count = sum(int((r["labels"][:, 1:] != -100).sum()) for r in rows)
    expected_loss = 0
    for row in rows:
        n = int((row["labels"][:, 1:] != -100).sum())
        loss = model(**row, use_cache=False).loss * n / count
        expected_loss += float(loss.detach())
        loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    output = adapter.forward(model, packed_batch(model, rows))
    output.loss.backward()
    assert float(output.loss.detach()) == pytest.approx(expected_loss, abs=2e-6)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=3e-4)
    assert sum(output.losses.counts.values()) == count


def test_no_cross_sample_visibility_and_boundary_targets(fixture):
    config, _, rows = fixture
    model = load_thinker(config).eval()
    with torch.no_grad():
        packed = packed_batch(model, rows)
        original = model(**packed, use_cache=False).logits
        altered = deepcopy(rows)
        # Change only ordinary text in the first example; later media/positions must be unaffected.
        altered[0]["input_ids"][0, 10:20] = 100
        changed = model(**packed_batch(model, altered), use_cache=False).logits
    boundary = rows[0]["input_ids"].shape[1]
    assert not torch.equal(original[:, :boundary], changed[:, :boundary])
    torch.testing.assert_close(original[:, boundary:], changed[:, boundary:], rtol=0, atol=0)
    for start in packed["packing_cu_seqlens"][:-1]:
        assert packed["labels"][0, start] == -100
    assert packed["attention_mask"] == {"full_attention": None}


def test_packed_cost_and_configuration(fixture):
    config, _, _ = fixture
    rows = [
        SampleCost(
            sample_id=str(i),
            prediction_id="p0",
            contract_id="x",
            modalities=("text",),
            llm_tokens=n,
        )
        for i, n in enumerate([10, 100])
    ]
    planner = ModalBatchPlanner(config.budget, packing=True)
    cost = planner.cost(rows)
    assert cost.allocated_llm_tokens == 110
    assert cost.allocated_attention_pairs == 10100
    assert list(planner.plan(rows))[0].execution == "varlen"
    with pytest.raises(ValueError, match="packing requires"):
        TrainConfig.model_validate({**config.model_dump(), "packing": "flash_attention_2"})


def test_native_export_does_not_serialize_packing_backend(fixture, tmp_path):
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    config, _, rows = fixture
    model = load_thinker(config).eval()
    with torch.no_grad():
        model(**packed_batch(model, rows), use_cache=False)
        expected = model(**rows[0], use_cache=False).logits
    model.save_pretrained(tmp_path / "export")
    reloaded = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        tmp_path / "export", attn_implementation="sdpa"
    ).eval()
    assert reloaded.model.config._attn_implementation == "sdpa"
    with torch.no_grad():
        torch.testing.assert_close(
            reloaded(**rows[0], use_cache=False).logits, expected, rtol=0, atol=0
        )
