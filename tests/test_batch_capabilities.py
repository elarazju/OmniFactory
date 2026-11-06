"""Base-only validation of task capabilities and every planning entry point."""

from dataclasses import replace

import pytest
from test_batching import budget, row

from omnifactory.adapters.batching import TaskBatchCapability
from omnifactory.adapters.runtime import adapter_spec, register_adapter
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import ModalBatchPlanner


def test_registry_requires_complete_capabilities():
    source = adapter_spec("qwen2_5_omni")
    with pytest.raises(OmniFactoryError, match="E_BATCH_CAPABILITY"):
        register_adapter(replace(source, name="invalid_missing_caps", batch_capabilities={}))
    with pytest.raises(OmniFactoryError, match="E_BATCH_CAPABILITY"):
        register_adapter(
            replace(
                source,
                name="invalid_caps",
                batch_capabilities=dict.fromkeys(source.tasks, "padded"),
            )
        )
    with pytest.raises(OmniFactoryError, match="E_BATCH_CAPABILITY"):
        TaskBatchCapability(heterogeneous=True).validate()


def test_bounded_heterogeneous_planning_and_fallback_resume():
    capability = adapter_spec("qwen2_5_omni").batch_capability("understanding_sft")
    samples = [
        row(str(i), 10 + i).model_copy(
            update={"modalities": ("text", m) if m != "text" else ("text",)}
        )
        for i, m in enumerate(["text", "image", "audio", "video"] * 5)
    ]
    planner = ModalBatchPlanner(budget(), window_size=4, seed=19, capability=capability)
    plans = list(planner.plan(samples))
    assert plans == list(planner.plan(samples))
    assert plans[3:] == list(planner.plan(iter(samples)))[3:]
    by_id = {s.sample_id: s for s in samples}
    native = [p for p in plans if p.execution == "padded"]
    assert all(len(p.example_ids) == 3 for p in native)
    for p in native:
        assert {m for sid, _ in p.example_ids for m in by_id[sid].modalities} == {
            "text",
            "image",
            "audio",
        }
        assert p.cost.useful_llm_tokens <= p.cost.allocated_llm_tokens
    assert all(
        p.fallback_reason == "unsupported_native_modalities:video"
        for p in plans
        if p.execution == "sequential"
    )
    incompatible = samples[0].model_copy(
        update={"sample_id": "other", "contract_id": "other-recipe"}
    )
    assert len(list(planner.plan([samples[0], incompatible]))) == 2
    with pytest.raises(OmniFactoryError, match="E_SAMPLE_OVER_BUDGET"):
        list(planner.plan([samples[0].model_copy(update={"llm_tokens": 2001})]))


def test_minicpm_declares_batched_decoder_and_singleton_media():
    cap = adapter_spec("minicpm_o_4_5").batch_capability("understanding_sft")
    assert cap.heterogeneous and cap.mode == "padded" and cap.media_execution == "per_example"
    assert cap.decision([row("a", 10)], timed=True) == (
        "sequential",
        "timed_attention_requires_sample_local_execution",
    )
    assert adapter_spec("qwen3_omni").batch_capability("understanding_sft").mode == "sequential"


def test_benchmark_source_contract_is_identical():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from omnifactory.training.benchmark import source_batches

    cap = adapter_spec("qwen2_5_omni").batch_capability("understanding_sft")
    samples = [
        row("text", 20),
        row("image", 30).model_copy(update={"modalities": ("text", "image")}),
        row("video", 40).model_copy(update={"modalities": ("video",)}),
        row("other", 20).model_copy(update={"contract_id": "other-recipe"}),
    ]
    planner = ModalBatchPlanner(budget(), capability=cap)
    assert list(source_batches(samples, planner)) == [
        (("text", "p0"), ("image", "p0")),
        (("video", "p0"),),
        (("other", "p0"),),
    ]


def test_capability_version_binds_execution_but_not_native_encoding(monkeypatch, tmp_path):
    from omnifactory.adapters import runtime
    from omnifactory.training.config import TrainConfig

    cfg = TrainConfig(
        model="fixture-model",
        dataset=tmp_path / "train.jsonl",
        output_dir=tmp_path / "run",
        components={"language_model": {"mode": "full", "lr": 0.001}},
        budget=budget(),
    )
    before, encoding = cfg.execution_contract(), cfg.encoding_contract()
    source = adapter_spec(cfg.adapter)
    capabilities = {
        **source.batch_capabilities,
        cfg.task: replace(source.batch_capability(cfg.task), version="next-native-layout"),
    }
    monkeypatch.setitem(
        runtime._REGISTRY, cfg.adapter, replace(source, batch_capabilities=capabilities)
    )
    assert cfg.execution_contract() != before
    assert cfg.encoding_contract() == encoding
    opted = cfg.model_copy(update={"heterogeneous_batches": True})
    assert opted.execution_contract() != cfg.execution_contract()
    assert opted.encoding_contract() == encoding
