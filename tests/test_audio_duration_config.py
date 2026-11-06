"""Duration limits belong to adapter capabilities, not all audio tasks globally."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnifactory.adapters.runtime import adapter_specs
from omnifactory.training.config import TrainConfig, read_config


def test_long_duplex_recipe_and_existing_family_limits():
    examples = Path(__file__).resolve().parents[1] / "examples"
    duplex = read_config(examples / "train-personaplex-real.yaml")
    assert duplex.max_audio_seconds == 180
    assert duplex.max_codec_tokens == 2250
    assert duplex.max_length > duplex.max_codec_tokens
    # Other built-ins retain their previous cap until independently validated.
    for spec in adapter_specs():
        assert spec.max_audio_seconds == (None if spec.name == "personaplex" else 30)
    qwen = read_config(examples / "train-qwen2.5-asr-baseline.yaml")
    for seconds in (30, 30.001):
        values = {**qwen.model_dump(), "max_audio_seconds": seconds}
        if seconds == 30:
            assert TrainConfig.model_validate(values).max_audio_seconds == seconds
        else:
            with pytest.raises(ValidationError, match="qwen2_5_omni max_audio_seconds"):
                TrainConfig.model_validate(values)


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_duplex_duration_must_remain_finite_and_positive(seconds):
    recipe = Path(__file__).resolve().parents[1] / "examples/train-personaplex-real.yaml"
    values = {**read_config(recipe).model_dump(), "max_audio_seconds": seconds}
    with pytest.raises(ValidationError, match="max_audio_seconds"):
        TrainConfig.model_validate(values)


def test_agent_depth_requires_frozen_user_parameters_and_no_user_objective():
    recipe = Path(__file__).resolve().parents[1] / "examples/train-personaplex-real.yaml"
    base = read_config(recipe)
    values = {**base.model_dump(), "personaplex_agent_only_depth": True}
    optimized = TrainConfig.model_validate(values)
    assert optimized.encoding_contract() == base.encoding_contract()
    assert optimized.execution_contract() != base.execution_contract()
    with pytest.raises(ValidationError, match="zero user loss"):
        TrainConfig.model_validate({**values, "personaplex_user_loss_weight": 0.1})
    for component, policy in (
        ("code_predictor", dict(mode="full", lr=1e-4)),
        ("audio_projector", dict(mode="full", lr=1e-4)),
        ("speech_generator", dict(mode="lora", rank=8, alpha=16, lr=1e-4, target_modules=["8"])),
    ):
        with pytest.raises(ValidationError, match="frozen user depth"):
            TrainConfig.model_validate({**values, "components": {component: policy}})
