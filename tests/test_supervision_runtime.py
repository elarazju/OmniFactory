"""Real autograd and safetensors checks for mixed continuous/discrete aligned supervision."""

import copy
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from omnifactory.adapters.base import EncodedExample
from omnifactory.adapters.supervision import (
    causal_visibility,
    tensor_ref,
)
from omnifactory.data.processor import OmniDataProcessor
from omnifactory.data.supervision import (
    AlignedStream,
    CausalAlignment,
    ContinuousRepresentation,
    DiscreteRepresentation,
    LossObjective,
    Normalizer,
    ReductionMeasure,
    SupervisionPlan,
    SupervisionTarget,
    Timeline,
)
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import ModalBatchPlanner, ResourceBudget, SampleCost
from omnifactory.training.engine import backward_window, load_encoded, load_tensors
from omnifactory.training.losses import LossBundle, LossTerm, supervised_losses
from omnifactory.training.prepare import save_encoded


def mixed_plan(tensors):
    def ref(key, axes):
        return tensor_ref(key, tensors[key], axes)

    text = SupervisionTarget(
        id="text",
        modality="text",
        representation=DiscreteRepresentation(vocabulary_sizes=(3,)),
        values=ref("symbols", ("frame",)),
        mask=ref("text_valid", ("frame",)),
        measure=ReductionMeasure(unit="symbols"),
    )
    audio = SupervisionTarget(
        id="audio",
        modality="audio",
        representation=ContinuousRepresentation(space="audio_vae", encoder_id="fixture-v1"),
        values=ref("latents", ("frame", "feature")),
        mask=ref("audio_valid", ("frame",)),
        measure=ReductionMeasure(
            unit="seconds", feature_reduction="mean", weights=ref("duration", ("frame",))
        ),
    )
    return SupervisionPlan(
        targets=(text, audio),
        normalizers=(
            Normalizer(id="text_symbols", unit="symbols"),
            Normalizer(id="audio_seconds", unit="seconds"),
        ),
        objectives=(
            LossObjective(
                id="text_ce",
                target_id="text",
                kind="cross_entropy",
                normalizer="text_symbols",
                weight=0.7,
            ),
            LossObjective(
                id="latent_mse",
                target_id="audio",
                kind="mse",
                normalizer="audio_seconds",
                weight=2.5,
            ),
        ),
    )


def aligned_example():
    tensors = {
        "symbols": torch.tensor([0, 2, -100]),
        "text_valid": torch.tensor([True, True, False]),
        "latents": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]]),
        "audio_valid": torch.tensor([True, True, False]),
        "duration": torch.tensor([0.1, 0.2, 999.0], dtype=torch.float64),
        "mic": torch.randn(4, 2),
        "mic_valid": torch.tensor([True, True, True, False]),
        "mic_intervals": torch.tensor([[0, 10000], [10000, 20000], [20000, 30000], [30000, 40000]]),
        "mic_available": torch.tensor([10000, 20000, 30000, 40000]),
        "reply_intervals": torch.tensor([[20000, 120000], [120000, 320000], [0, 0]]),
        "reply_available": torch.tensor([120000, 320000, 0]),
    }
    plan = mixed_plan(tensors)

    def ref(key, axes):
        return tensor_ref(key, tensors[key], axes)

    mic = AlignedStream(
        id="mic",
        timeline_id="conversation",
        modality="audio",
        direction="input",
        representation=ContinuousRepresentation(space="audio_features"),
        values=ref("mic", ("frame", "feature")),
        event_axis="frame",
        intervals_us=ref("mic_intervals", ("frame", "boundary")),
        available_at_us=ref("mic_available", ("frame",)),
        valid=ref("mic_valid", ("frame",)),
    )
    target = plan.targets[1]
    reply = AlignedStream(
        id="reply",
        timeline_id="conversation",
        modality="audio",
        direction="output",
        target_id="audio",
        representation=target.representation,
        values=target.values,
        event_axis="frame",
        intervals_us=ref("reply_intervals", ("frame", "boundary")),
        available_at_us=ref("reply_available", ("frame",)),
        valid=target.mask,
    )
    raw = plan.model_dump()
    raw.update(
        timelines=[Timeline(id="conversation")],
        streams=[mic, reply],
        alignments=[
            CausalAlignment(query_stream="reply", context_stream="mic"),
            CausalAlignment(query_stream="reply", context_stream="reply"),
        ],
    )
    plan = SupervisionPlan.model_validate(raw)
    return EncodedExample(
        tensors=tensors,
        cost=SampleCost(
            sample_id="aligned",
            prediction_id="p0",
            contract_id="test",
            modalities=("text", "audio"),
            latent_elements=14,
            stream_events=7,
        ),
        supervision=plan,
    ).validate()


def test_mixed_aligned_artifact_roundtrip_and_inference_guard(tmp_path):
    encoded = aligned_example()
    assert "input_ids" not in encoded.payload and "labels" not in encoded.payload
    assert encoded.normalization_counts == {
        "text_symbols": 2.0,
        "audio_seconds": pytest.approx(0.3),
    }
    (tmp_path / "tensors").mkdir()
    entry = save_encoded(tmp_path, encoded, 0)
    restored = load_encoded(tmp_path, entry)
    assert restored.supervision == encoded.supervision
    for key in encoded.tensors:
        torch.testing.assert_close(encoded.tensors[key], restored.tensors[key])
    with pytest.raises(OmniFactoryError, match="E_MEASURE_UNIT"):
        _ = restored.target_tokens
    with pytest.raises(OmniFactoryError, match="E_TARGET_LEAK"):
        restored.validate(inference=True)
    entry["normalization_counts"]["audio_seconds"] = 20
    with pytest.raises(OmniFactoryError, match="E_PREPARED_MEASURE"):
        load_encoded(tmp_path, entry)


def test_stream_visibility_blocks_future_padding_and_self_target():
    encoded = aligned_example()
    p, t = encoded.supervision, encoded.tensors
    visible = causal_visibility(p, t, "reply", "mic")
    assert visible.tolist() == [[True, True, False, False], [True, True, True, False], [False] * 4]
    self_mask = causal_visibility(p, t, "reply", "reply")
    assert self_mask.tolist() == [[False] * 3, [True, False, False], [False] * 3]
    with pytest.raises(OmniFactoryError, match="E_ALIGNMENT_BUDGET"):
        causal_visibility(p, t, "reply", "mic", max_pairs=2)
    t["mic_available"][0] = 0
    with pytest.raises(OmniFactoryError, match="E_STREAM_TIME"):
        causal_visibility(p, t, "reply", "mic")


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("nan_latent", "E_CONTINUOUS_VALUES"),
        ("negative_duration", "E_MEASURE"),
        ("bad_symbol", "E_DISCRETE_RANGE"),
        ("missing", "E_TENSOR_REFERENCE"),
        ("wrong_dtype", "E_TENSOR_REFERENCE"),
        ("bad_interval", "E_STREAM_TIME"),
    ],
)
def test_runtime_contract_rejects_corruption(mutation, error):
    encoded = aligned_example()
    t = encoded.tensors
    if mutation == "nan_latent":
        t["latents"][0, 0] = float("nan")
    elif mutation == "negative_duration":
        t["duration"][0] = -1
    elif mutation == "bad_symbol":
        t["symbols"][0] = 3
    elif mutation == "missing":
        del t["audio_valid"]
    elif mutation == "wrong_dtype":
        t["audio_valid"] = t["audio_valid"].long()
    elif mutation == "bad_interval":
        t["mic_intervals"][1, 1] = 0
    with pytest.raises(OmniFactoryError, match=error):
        encoded.validate()


def test_latent_and_stream_costs_require_explicit_budgets():
    example = aligned_example()
    raw = dict(
        max_llm_tokens=1,
        max_vision_patches=1,
        max_audio_frames=1,
        max_codec_frames=1,
        max_decoded_media_bytes=1000,
    )
    with pytest.raises(OmniFactoryError, match="E_SAMPLE_OVER_BUDGET"):
        list(ModalBatchPlanner(ResourceBudget(**raw)).plan([example.cost]))
    raw.update(max_latent_elements=14, max_stream_events=7)
    plans = list(ModalBatchPlanner(ResourceBudget(**raw)).plan([example.cost]))
    assert plans[0].cost.allocated_llm_tokens == 0
    example.cost = example.cost.model_copy(update={"latent_elements": 0})
    with pytest.raises(OmniFactoryError, match="E_SUPERVISION_COST"):
        example.validate()


class ReferenceAdapter:
    """Actual linear multi-task model through the production accumulation function."""

    def collate(self, rows):
        return {key: torch.cat([r[key] for r in rows]) for key in rows[0]}

    @staticmethod
    def to_device(batch, model, config):
        return batch

    def forward(self, model, batch):
        output = model(batch["features"])
        bundle = supervised_losses(
            mixed_plan(batch), batch, {"text_ce": output[:, :3], "latent_mse": output[:, 3:]}
        )
        return SimpleNamespace(losses=bundle, loss=bundle.loss)


def test_production_accumulation_uses_each_objectives_own_measure(tmp_path):
    torch.manual_seed(9)
    (tmp_path / "tensors").mkdir()
    entries = []
    for i, length in enumerate((2, 5)):
        tensors = {
            "features": torch.randn(length, 3),
            "symbols": torch.randint(0, 3, (length,)),
            "text_valid": torch.arange(length) < i + 1,
            "latents": torch.randn(length, 2),
            "audio_valid": torch.ones(length, dtype=torch.bool),
            "duration": torch.arange(1, length + 1, dtype=torch.float64) * (0.1 if i else 0.7),
        }
        encoded = EncodedExample(
            tensors=tensors,
            supervision=mixed_plan(tensors),
            cost=SampleCost(
                sample_id=str(i),
                prediction_id="p",
                contract_id="mixed",
                modalities=("text", "audio"),
                latent_elements=length * 2,
            ),
        )
        entries.append(save_encoded(tmp_path, encoded, i))
    full = torch.nn.Linear(3, 5)
    micro = copy.deepcopy(full)
    adapter = ReferenceAdapter()
    results = []
    for model, chunks in ((full, [entries]), (micro, [[e] for e in entries])):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        bundle = backward_window(model, adapter, None, chunks, lambda e: load_tensors(tmp_path, e))
        optimizer.step()
        results.append(bundle)
    for a, b in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(results[0].loss, results[1].loss)
    assert results[0].units == {"text_symbols": "symbols", "audio_seconds": "seconds"}


def test_reference_reducers_exclude_masked_values_and_keep_gradients():
    example = aligned_example()
    t, plan = example.tensors, example.supervision
    logits = torch.randn(3, 3, requires_grad=True)
    latent = torch.randn(3, 2, requires_grad=True)
    bundle = supervised_losses(plan, t, {"text_ce": logits, "latent_mse": latent})
    expected = 0.7 * torch.nn.functional.cross_entropy(logits[:2], t["symbols"][:2])
    mse = (latent[:2] - t["latents"][:2]).square().mean(-1)
    expected = expected + 2.5 * (mse * t["duration"][:2]).sum() / t["duration"][:2].sum()
    torch.testing.assert_close(bundle.loss, expected)
    bundle.loss.backward()
    assert logits.grad[2].abs().sum() == 0 and latent.grad[2].abs().sum() == 0


def test_flow_objectives_require_explicit_model_implementation():
    example = aligned_example()
    raw = example.supervision.model_dump()
    raw["objectives"][1]["kind"] = "flow_matching"
    plan = SupervisionPlan.model_validate(raw)
    with pytest.raises(OmniFactoryError, match="E_LOSS_IMPLEMENTATION"):
        supervised_losses(
            plan, example.tensors, {"text_ce": torch.randn(3, 3), "latent_mse": torch.randn(3, 2)}
        )


def test_processor_encode_validates_the_execution_contract():
    example = aligned_example()

    class Adapter:
        def encode(self, *args, **kwargs):
            return example

    result = OmniDataProcessor().encode(None, "p", adapter=Adapter(), contract_id="test")
    assert result is example
    with pytest.raises(OmniFactoryError, match="E_TARGET_LEAK"):
        OmniDataProcessor().encode(None, "p", adapter=Adapter(), contract_id="test", inference=True)


def test_loss_bundle_rejects_nonfinite_sums_and_invalid_denominators():
    with pytest.raises(OmniFactoryError, match="E_LOSS_NUMERATOR"):
        LossBundle(
            (LossTerm("x", torch.tensor(float("nan")), "frames"),),
            {"frames": 1},
            {"frames": "frames"},
        )
    with pytest.raises(OmniFactoryError, match="E_LOSS_COUNTS"):
        LossBundle(
            (LossTerm("x", torch.tensor(1.0), "frames"),), {"frames": 0}, {"frames": "frames"}
        )


def test_shared_target_measure_is_counted_once_and_disabled_terms_do_not_count():
    example = aligned_example()
    raw = example.supervision.model_dump()
    raw["objectives"] = (*raw["objectives"], {**raw["objectives"][1], "id": "auxiliary_mse"})
    example.supervision = SupervisionPlan.model_validate(raw)
    assert example.normalization_counts == pytest.approx({"text_symbols": 2, "audio_seconds": 0.3})
    raw["objectives"][0]["weight"] = 0
    example.supervision = SupervisionPlan.model_validate(raw)
    assert example.normalization_counts == pytest.approx({"text_symbols": 0, "audio_seconds": 0.3})


def test_masked_prediction_nan_has_no_loss_or_gradient_contribution():
    example = aligned_example()
    logits = torch.randn(3, 3)
    latent = torch.randn(3, 2)
    logits[2] = float("nan")
    latent[2] = float("nan")
    logits.requires_grad_()
    latent.requires_grad_()
    bundle = supervised_losses(
        example.supervision, example.tensors, {"text_ce": logits, "latent_mse": latent}
    )
    bundle.loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(latent.grad).all()
    assert logits.grad[2].abs().sum() == 0 and latent.grad[2].abs().sum() == 0


def test_stream_target_cannot_supervise_invalid_event():
    example = aligned_example()
    raw = example.supervision.model_dump()
    example.tensors["reply_valid"] = torch.tensor([False, True, False])
    raw["streams"][1]["valid"] = tensor_ref(
        "reply_valid", example.tensors["reply_valid"], ("frame",)
    ).model_dump()
    example.supervision = SupervisionPlan.model_validate(raw)
    with pytest.raises(OmniFactoryError, match="E_STREAM_TARGET"):
        example.validate()


def test_window_measure_rejects_real_undercount_but_tolerates_float_roundoff():
    bundle = LossBundle(
        (LossTerm("mse", torch.tensor(1.0), "duration"),),
        {"duration": 3.6},
        {"duration": "seconds"},
    )
    assert torch.isfinite(bundle.scaled_loss({"duration": 3.5999999999999996}))
    with pytest.raises(OmniFactoryError, match="E_LOSS_COUNTS"):
        bundle.scaled_loss({"duration": 3.5})


def test_prepared_artifact_requires_explicit_supervision(tmp_path):
    with pytest.raises(OmniFactoryError, match="E_PREPARED_VERSION"):
        load_encoded(tmp_path, {"target_tokens": 10})


def test_multicodebook_ce_excludes_padded_vocabulary_classes():
    tensors = {"codes": torch.tensor([[1, 2], [0, 1]]), "valid": torch.ones(2, 2, dtype=torch.bool)}
    target = SupervisionTarget(
        id="codec",
        modality="audio",
        representation=DiscreteRepresentation(vocabulary_sizes=(2, 3), codebook_axis="codebook"),
        values=tensor_ref("codes", tensors["codes"], ("frame", "codebook")),
        mask=tensor_ref("valid", tensors["valid"], ("frame", "codebook")),
        measure=ReductionMeasure(unit="symbols"),
    )
    plan = SupervisionPlan(
        targets=(target,),
        normalizers=(Normalizer(id="symbols", unit="symbols"),),
        objectives=(
            LossObjective(
                id="codec_ce", target_id="codec", kind="cross_entropy", normalizer="symbols"
            ),
        ),
    )
    logits = torch.randn(2, 2, 3)
    logits[:, 0, 2] = 1000  # Not a class in the first codebook.
    logits.requires_grad_()
    bundle = supervised_losses(plan, tensors, {"codec_ce": logits})
    expected = (
        sum(
            torch.nn.functional.cross_entropy(
                logits[:, i, :size], tensors["codes"][:, i], reduction="sum"
            )
            for i, size in enumerate((2, 3))
        )
        / 4
    )
    torch.testing.assert_close(bundle.loss, expected)
    bundle.loss.backward()
    assert logits.grad[:, 0, 2].abs().sum() == 0
    with pytest.raises(OmniFactoryError, match="E_LOSS_SHAPE"):
        supervised_losses(plan, tensors, {"codec_ce": torch.randn(2, 2, 4)})
