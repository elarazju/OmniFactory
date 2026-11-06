"""The public contract imports without torch; types reject ambiguous supervision."""

import pytest
from pydantic import ValidationError

from omnifactory.data.supervision import (
    ContinuousRepresentation,
    DiscreteRepresentation,
    LossObjective,
    Normalizer,
    ReductionMeasure,
    SupervisionPlan,
    SupervisionTarget,
    TensorRef,
)


def latent_target(unit="frames"):
    return SupervisionTarget(
        id="audio",
        modality="audio",
        representation=ContinuousRepresentation(space="audio_vae"),
        values=TensorRef(key="latents", axes=("frame", "feature"), shape=(3, 8), dtype="float32"),
        mask=TensorRef(key="valid", axes=("frame",), shape=(3,), dtype="bool"),
        measure=ReductionMeasure(unit=unit, feature_reduction="mean"),
    )


def test_continuous_contract_roundtrip_without_token_fields():
    plan = SupervisionPlan(
        targets=(latent_target(),),
        normalizers=(Normalizer(id="audio_frames", unit="frames"),),
        objectives=(
            LossObjective(
                id="velocity", target_id="audio", kind="flow_matching", normalizer="audio_frames"
            ),
        ),
    )
    assert SupervisionPlan.model_validate_json(plan.model_dump_json()) == plan
    assert plan.targets[0].representation.kind == "continuous"


@pytest.mark.parametrize("unit", ["symbols", "seconds", "elements"])
def test_invalid_latent_measure_is_rejected(unit):
    with pytest.raises(ValidationError, match="E_MEASURE"):
        latent_target(unit)


@pytest.mark.parametrize("kind,unit", [("cross_entropy", "frames"), ("mse", "symbols")])
def test_loss_kind_and_unit_are_not_interchangeable(kind, unit):
    with pytest.raises(ValidationError, match="E_LOSS"):
        SupervisionPlan(
            targets=(latent_target(),),
            normalizers=(Normalizer(id="normalizer", unit=unit),),
            objectives=(
                LossObjective(id="loss", target_id="audio", kind=kind, normalizer="normalizer"),
            ),
        )


def test_codebooks_require_an_explicit_axis():
    with pytest.raises(ValidationError, match="E_CODEBOOK_AXIS"):
        DiscreteRepresentation(vocabulary_sizes=(1024, 128))


def test_unknown_targets_and_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="E_LOSS_REFERENCE"):
        SupervisionPlan(
            objectives=(
                LossObjective(id="bad", target_id="missing", kind="mse", normalizer="missing"),
            )
        )
    with pytest.raises(ValidationError, match="E_SUPERVISION_ID"):
        SupervisionPlan(targets=(latent_target(), latent_target()))


def test_tensor_axes_are_unique_and_ranked():
    with pytest.raises(ValidationError, match="E_SUPERVISION_AXES"):
        TensorRef(key="x", axes=("time", "time"), shape=(2, 3), dtype="float32")
