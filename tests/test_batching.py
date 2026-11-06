import random

import pytest
from pydantic import ValidationError

from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import (
    MemoryProfile,
    ModalBatchPlanner,
    ResourceBudget,
    SampleCost,
)


def budget(**overrides):
    return ResourceBudget.model_validate(
        dict(
            max_examples=4,
            max_llm_tokens=2000,
            max_vision_patches=1000,
            max_audio_frames=4000,
            max_codec_frames=1000,
            max_decoded_media_bytes=1000000,
        )
        | overrides
    )


def row(name, length, **kwargs):
    return SampleCost(
        sample_id=name,
        prediction_id="p0",
        contract_id="test-v1",
        modalities=("text",),
        llm_tokens=length,
        **kwargs,
    )


def test_padded_allocation_not_sum_of_useful_tokens():
    planner = ModalBatchPlanner(budget())
    cost = planner.cost([row("a", 100), row("b", 1000)])
    assert cost.useful_llm_tokens == 1100
    assert cost.allocated_llm_tokens == 2000


def test_preserves_every_example_and_all_budget_dimensions():
    rng = random.Random(42)
    samples = [
        row(
            str(i),
            rng.randint(1, 1200),
            audio_frames=rng.randint(0, 1500),
            vision_patches=rng.randint(0, 300),
            decoded_media_bytes=1000,
        )
        for i in range(100)
    ]
    planner = ModalBatchPlanner(budget(), window_size=13, seed=9)
    plans = list(planner.plan(samples))
    ids = [identity for plan in plans for identity in plan.example_ids]
    assert len(ids) == len(set(ids)) == len(samples)
    assert set(ids) == {(s.sample_id, s.prediction_id) for s in samples}
    for plan in plans:
        assert not planner.exceeded(plan.cost)
        assert not plan.gpu_fit_verified


def test_same_seed_same_plan():
    samples = [row(str(i), 400) for i in range(20)]
    a = list(ModalBatchPlanner(budget(), seed=10).plan(samples))
    b = list(ModalBatchPlanner(budget(), seed=10).plan(samples))
    c = list(ModalBatchPlanner(budget(), seed=11).plan(samples))
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    "field,value",
    [
        ("llm_tokens", 2001),
        ("audio_frames", 4001),
        ("vision_patches", 1001),
        ("codec_frames", 1001),
        ("decoded_media_bytes", 1000001),
    ],
)
def test_oversize_single_sample_not_truncated(field, value):
    data = row("long-video", 100).model_dump()
    data[field] = value
    with pytest.raises(OmniFactoryError, match="E_SAMPLE_OVER_BUDGET.*long-video"):
        list(ModalBatchPlanner(budget()).plan([SampleCost.model_validate(data)]))


def test_different_contracts_never_mix():
    a = row("a", 100)
    b = row("b", 100).model_copy(update={"contract_id": "another-model"})
    plans = list(ModalBatchPlanner(budget()).plan([a, b]))
    assert len(plans) == 2


def test_different_modalities_bucket_separately():
    a = row("a", 100)
    b = row("b", 100).model_copy(update={"modalities": ("text", "video")})
    assert len(list(ModalBatchPlanner(budget()).plan([a, b]))) == 2


def test_gpu_budget_requires_profile():
    with pytest.raises(OmniFactoryError, match="E_MEMORY_PROFILE"):
        ModalBatchPlanner(budget(max_gpu_bytes=10000))


def test_explicit_profile_applies_margin_and_preserves_provenance():
    profile = MemoryProfile(
        profile_id="demo",
        contract_id="test-v1",
        hardware="synthetic",
        provenance="synthetic",
        fixed_bytes=100,
        bytes_per_llm_token=2,
        bytes_per_vision_patch=0,
        bytes_per_audio_frame=0,
        bytes_per_codec_frame=0,
        safety_fraction=0.1,
    )
    planner = ModalBatchPlanner(budget(max_gpu_bytes=400), memory_profile=profile)
    plans = list(planner.plan([row("a", 100), row("b", 100)]))
    assert len(plans) == 2
    assert plans[0].cost.estimated_gpu_bytes == 330
    assert plans[0].memory_provenance == "synthetic"
    assert not plans[0].gpu_fit_verified


def test_profile_contract_mismatch_rejected():
    profile = MemoryProfile(
        profile_id="demo",
        contract_id="different-model",
        hardware="synthetic",
        provenance="synthetic",
        fixed_bytes=100,
        bytes_per_llm_token=2,
        bytes_per_vision_patch=0,
        bytes_per_audio_frame=0,
        bytes_per_codec_frame=0,
    )
    with pytest.raises(OmniFactoryError, match="E_MEMORY_PROFILE"):
        list(ModalBatchPlanner(budget(), memory_profile=profile).plan([row("a", 100)]))


def test_bounded_window_does_not_exhaust_source():
    consumed = []

    def source():
        for i in range(10000):
            consumed.append(i)
            yield row(str(i), 100)

    iterator = ModalBatchPlanner(budget(), window_size=8).plan(source())
    next(iterator)
    assert len(consumed) == 8


def test_negative_and_float_costs_rejected():
    for value in (-1, 1.5, True):
        with pytest.raises(ValidationError):
            row("bad", value)


@pytest.mark.parametrize("packing,timed", [(False, False), (True, False), (True, True)])
def test_pure_telemetry_preserves_capability_packing_and_timed(packing, timed):
    from omnifactory.adapters.batching import TaskBatchCapability
    from omnifactory.planning.batching import estimate_batch_cost

    cap = TaskBatchCapability(mode="padded", packing=True, timed=False)
    costs = [row("a", 8), row("b", 17)]
    pure = estimate_batch_cost(costs, packing=packing, capability=cap, timed=timed)
    planned = ModalBatchPlanner(budget(), packing=packing, capability=cap, timed=timed).cost(costs)
    assert pure == planned
    assert pure.allocated_llm_tokens == (25 if packing and not timed else 34)
    assert pure.allocated_attention_pairs == (8**2 + 17**2 if packing and not timed else 2 * 17**2)
    assert pure.estimated_gpu_bytes is None
