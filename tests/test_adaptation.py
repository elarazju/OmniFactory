import pytest
from pydantic import ValidationError

from omnifactory.errors import OmniFactoryError
from omnifactory.models.registry import builtin_registry
from omnifactory.planning.adaptation import AdaptationRequest, ParameterSpec, compile_adaptation


def inventory():
    return [
        ParameterSpec(name="llm.model.q_proj.weight", shape=(64, 64), kind="linear_weight"),
        ParameterSpec(name="llm.model.v_proj.weight", shape=(32, 64), kind="linear_weight"),
        ParameterSpec(
            name="audio_projection_layer.linear.weight", shape=(64, 32), kind="linear_weight"
        ),
        ParameterSpec(name="apm.weight", shape=(32, 32)),
    ]


def compile_plan(components, parameters=None):
    spec = builtin_registry().get("minicpm_o_4_5")
    request = AdaptationRequest(model=spec.key, components=components)
    return compile_adaptation(spec, request, inventory() if parameters is None else parameters)


def test_frozen_llm_still_on_projector_gradient_path():
    plan = compile_plan({"audio_projector": {"mode": "full", "lr": 2e-5}})
    assert set(plan.gradient_components) == {"audio_projector", "language_model"}
    assert "audio_encoder" in plan.cacheable_components
    assert "language_model" not in plan.cacheable_components
    assert plan.trainable_numel == 64 * 32


def test_language_lora_does_not_require_encoder_backward():
    plan = compile_plan({"language_model": {"mode": "lora", "lr": 1e-4, "rank": 8, "alpha": 16}})
    assert plan.gradient_components == ("language_model",)
    assert "audio_projector" in plan.cacheable_components
    assert plan.trainable_numel == 8 * (64 + 64) + 8 * (32 + 64)


def test_frozen_interleaved_moe_components_remain_on_gradient_path():
    spec = builtin_registry().get("qwen3_omni")
    plan = compile_adaptation(
        spec,
        AdaptationRequest(
            model=spec.key,
            components={"language_model": {"mode": "lora", "lr": 1e-4, "rank": 2, "alpha": 4}},
        ),
        [
            ParameterSpec(
                name="thinker.model.layers.0.self_attn.q_proj.weight",
                shape=(16, 16),
                kind="linear_weight",
            )
        ],
    )
    assert set(plan.gradient_components) == {
        "language_model",
        "language_experts",
        "language_router",
    }
    assert "audio_encoder" in plan.cacheable_components
    assert {g.component for g in plan.groups} == {"language_model"}


def test_invalid_gradient_coupling_rejected():
    raw = builtin_registry().get("qwen3_omni").model_dump()
    raw["gradient_couplings"] = [["language_model", "unknown"]]
    from omnifactory.models.registry import ModelSpec

    with pytest.raises(ValidationError, match="gradient coupling"):
        ModelSpec.model_validate(raw)


def test_different_lrs_preserved():
    plan = compile_plan(
        {
            "language_model": {"mode": "lora", "lr": 1e-4, "rank": 8, "alpha": 16},
            "audio_projector": {"mode": "full", "lr": 2e-5},
        }
    )
    assert {g.component: g.lr for g in plan.groups} == {
        "language_model": 1e-4,
        "audio_projector": 2e-5,
    }


def test_no_matching_lora_modules_is_error():
    with pytest.raises(OmniFactoryError, match="E_NO_MATCH"):
        compile_plan(
            {
                "language_model": {
                    "mode": "lora",
                    "lr": 1e-4,
                    "rank": 8,
                    "alpha": 16,
                    "target_modules": ["nonexistent"],
                }
            }
        )


def test_cannot_full_train_quantized_weight():
    params = inventory()
    params[2] = params[2].model_copy(update={"quantized": True})
    with pytest.raises(OmniFactoryError, match="E_QUANTIZED_FULL"):
        compile_plan({"audio_projector": {"mode": "full", "lr": 1e-4}}, params)


def test_tied_trainable_aliases_rejected():
    params = inventory()
    params[0] = params[0].model_copy(update={"storage_id": "tied"})
    params[1] = params[1].model_copy(update={"storage_id": "tied"})
    with pytest.raises(OmniFactoryError, match="E_SHARED_TRAINABLE"):
        compile_plan({"language_model": {"mode": "full", "lr": 1e-4}}, params)


def test_unknown_component_not_silently_ignored():
    with pytest.raises(OmniFactoryError, match="E_COMPONENT"):
        compile_plan({"audio_projetor": {"mode": "full", "lr": 1e-4}})


def test_component_not_on_loss_path_rejected():
    with pytest.raises(OmniFactoryError, match="E_UNUSED_COMPONENT"):
        compile_plan({"speech_generator": {"mode": "full", "lr": 1e-4}})


def test_child_projector_ownership_excludes_parent_encoder():
    spec = builtin_registry().get("qwen2_5_omni_3b")
    request = AdaptationRequest(
        model=spec.key, components={"vision_projector": {"mode": "full", "lr": 1e-4}}
    )
    params = [
        ParameterSpec(name="thinker.visual.merger.mlp.0.weight", shape=(16, 16)),
        ParameterSpec(name="thinker.visual.blocks.0.weight", shape=(16, 16)),
    ]
    plan = compile_adaptation(spec, request, params)
    assert plan.groups[0].parameters == ("thinker.visual.merger.mlp.0.weight",)
    assert "vision_encoder" in plan.cacheable_components


def test_invalid_policy_fields():
    with pytest.raises(ValidationError, match="frozen"):
        AdaptationRequest(
            model="minicpm_o_4_5", components={"language_model": {"mode": "frozen", "lr": 0.1}}
        )


def test_stage_ids_are_model_owned_and_registry_rejects_duplicates():
    registry = builtin_registry()
    assert registry.get("minicpm_o_4_5").stages[0].id == "llm"
    assert registry.get("qwen2_5_omni_3b").stages[0].id == "thinker"
    with pytest.raises(OmniFactoryError, match="E_DUPLICATE_MODEL"):
        registry.register(registry.get("qwen3_omni"))


def test_cyclic_component_graph_rejected():
    spec = builtin_registry().get("minicpm_o_4_5").model_dump(mode="json")
    spec["components"][0]["inputs"] = ["language_model"]
    with pytest.raises(ValidationError, match="cyclic"):
        type(builtin_registry().get("minicpm_o_4_5")).model_validate(spec)
