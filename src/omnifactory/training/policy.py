"""Execute the component compiler against real parameters using PEFT LoRA layers."""

import torch
from peft import LoraConfig, inject_adapter_in_model, prepare_model_for_kbit_training
from peft.tuners.tuners_utils import BaseTunerLayer

from omnifactory.adapters.runtime import adapter_spec, policy_namespace
from omnifactory.errors import OmniFactoryError
from omnifactory.models.registry import builtin_registry
from omnifactory.planning.adaptation import AdaptationRequest, ParameterSpec, compile_adaptation


def parameter_inventory(model, prefix="thinker."):
    modules = dict(model.named_modules())
    inventory = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parent, _, attribute = name.rpartition(".")
        module = modules[parent]
        linear = isinstance(module, torch.nn.Linear) and attribute == "weight"
        quantized = type(parameter).__name__ == "Params4bit"
        shape = (module.out_features, module.in_features) if linear else tuple(parameter.shape)
        inventory.append(
            ParameterSpec(
                name=prefix + name,
                shape=shape,
                kind="linear_weight" if linear else "parameter",
                quantized=quantized,
                storage_id=str(parameter.untyped_storage().data_ptr()),
            )
        )
    return inventory


def apply_policy(model, config):
    model_key, prefix = policy_namespace(config.adapter, config.task)
    inventory = parameter_inventory(model, prefix)
    spec = adapter_spec(config.adapter)
    plan = compile_adaptation(
        spec.model_spec or builtin_registry().get(model_key),
        AdaptationRequest(
            model=model_key,
            components=config.components,
            loss_components=spec.loss_components[config.task],
        ),
        inventory,
    )
    if config.quantization != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.requires_grad_(False)
    targets, ranks, alphas = [], {}, {}
    for group in plan.groups:
        if group.mode == "lora":
            for parameter in group.parameters:
                name = parameter.removeprefix(prefix).removesuffix(".weight")
                targets.append(name)
                ranks[name], alphas[name] = group.rank, group.alpha
    if targets:
        inject_adapter_in_model(
            LoraConfig(
                target_modules=targets,
                r=next(iter(ranks.values())),
                lora_alpha=next(iter(alphas.values())),
                rank_pattern=ranks,
                alpha_pattern=alphas,
                lora_dropout=0.0,
                bias="none",
            ),
            model,
        )
    optimizer_groups = []
    for group in plan.groups:
        parameters = []
        for name in group.parameters:
            name = name.removeprefix(prefix)
            if group.mode == "full":
                parameters.append(model.get_parameter(name))
            else:
                module = model.get_submodule(name.removesuffix(".weight"))
                parameters.extend(module.lora_A["default"].parameters())
                parameters.extend(module.lora_B["default"].parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        optimizer_groups.append(
            {"params": parameters, "lr": group.lr, "component": group.component}
        )
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != plan.trainable_numel:
        raise OmniFactoryError(
            "E_TRAINABLE_COUNT", f"compiled {plan.trainable_numel}, injected {actual}"
        )
    if config.gradient_checkpointing:
        checkpoint_model = model.llm if config.adapter == "minicpm_o_4_5" else model
        if config.adapter in ("qwen2_5_omni", "qwen3_omni") and config.task == "speech_sft":
            checkpoint_model = model.thinker
        checkpoint_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if (
        config.gradient_checkpointing
        and config.task == "speech_sft"
        and config.adapter not in ("mimo_audio", "voxcpm2", "cosyvoice3", "personaplex")
    ):
        speech_model = model.tts.model if config.adapter == "minicpm_o_4_5" else model.talker
        speech_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if config.feature_cache is not None:
        from omnifactory.training.feature_cache import install_feature_cache

        install_feature_cache(model, config)
    if config.adapter == "minicpm_o_4_5":
        from omnifactory.adapters.minicpm_precision import configure_minicpm_precision

        configure_minicpm_precision(model, config)
    return model, optimizer_groups, plan


def merge_lora(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, BaseTunerLayer):
            module.merge(safe_merge=True)
            parent_name, _, attribute = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attribute, module.get_base_layer())
    # The weights now belong to a plain HF model, not a PEFT adapter directory.
    if hasattr(model, "peft_config"):
        del model.peft_config
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = False
    return model
