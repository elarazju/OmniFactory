"""Explicit BF16 storage/media + FP32 decoder computation.

Base model tensors retain BF16 storage, including native media encoders. Decoder
linear operations materialize transient FP32 weights; residuals, attention and
LoRA arithmetic/gradients are FP32. Native BF16 parity is NOT established by this
more expensive policy. Parameter keys and native export storage remain intact.
"""

from functools import wraps
from types import MethodType

import torch
from torch.nn import functional as F

from omnifactory.errors import OmniFactoryError


def _fp32_linear(self, value):
    # Disabling autocast is necessary even when the caller's inputs are FP32.
    with torch.autocast(device_type=value.device.type, enabled=False):
        return F.linear(
            value.float(), self.weight.float(), None if self.bias is None else self.bias.float()
        )


def _decoder_inputs(module, args, kwargs):
    embeddings = kwargs.get("inputs_embeds")
    if embeddings is None:
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids, args = args[0], (None, *args[1:])
            kwargs.pop("input_ids", None)
        elif ids is not None:
            kwargs["input_ids"] = None
        if ids is not None:
            embeddings = module.get_input_embeddings()(ids)
    if embeddings is not None:
        kwargs["inputs_embeds"] = embeddings.float()
    return args, kwargs


def configure_minicpm_precision(model, config):
    """Install after load and after LoRA injection; repeated installation is safe.

    Methods/hooks are runtime policy, not serialized model tensors or vendored
    source patches. Export metadata retains the policy for OmniFactory reload.
    """
    if config.minicpm_precision == "native":
        return model
    if torch.backends.cuda.matmul.allow_tf32:
        raise OmniFactoryError(
            "E_MINICPM_PRECISION", "fp32_decoder_compute requires TF32 matmul disabled"
        )
    for name, parameter in model.named_parameters():
        if ".lora_" in name and parameter.requires_grad:
            parameter.data = parameter.data.float()
    for module in model.llm.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(_fp32_linear, module)
    if not hasattr(model.llm, "_omnifactory_precision_hook"):
        model.llm._omnifactory_precision_hook = model.llm.register_forward_pre_hook(
            _decoder_inputs, with_kwargs=True
        )
        original_forward = model.llm.forward

        @wraps(original_forward)
        def forward(*args, **kwargs):
            if torch.backends.cuda.matmul.allow_tf32:
                raise OmniFactoryError(
                    "E_MINICPM_PRECISION", "fp32_decoder_compute requires TF32 matmul disabled"
                )
            # Preserve the HF signature for generation argument validation.
            # Non-reentrant checkpointing captures this disabled autocast state.
            with torch.autocast(device_type=model.llm.device.type, enabled=False):
                return original_forward(*args, **kwargs)

        model.llm.forward = forward
    return model
