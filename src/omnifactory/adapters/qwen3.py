"""Native Qwen3-Omni MoE understanding execution.

Only conversation storage/label boundaries are shared with Qwen2.5. Native
Qwen3 processors and modules own audio reduction, interleaved mRoPE and visual
deepstack. Router auxiliary training, packing and feature caching are separate
capabilities and are deliberately not inherited as supported configurations.
"""

import torch
from transformers import (
    PretrainedConfig,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from omnifactory.adapters.qwen2_5 import Qwen25Adapter
from omnifactory.errors import OmniFactoryError


class Qwen3Adapter(Qwen25Adapter):
    processor_class = Qwen3OmniMoeProcessor

    codec_vocabularies = (3072,) + (2048,) * 15
    codec_axes = ("frame", "codebook")

    def load_code_labels(self):
        from omnifactory.adapters.qwen3_speech import Qwen3CodeLabels

        return Qwen3CodeLabels(self.training_config.speech_codes)

    def forward(self, model, batch):
        from types import SimpleNamespace

        from omnifactory.adapters.qwen3_speech import talker_condition, talker_logits
        from omnifactory.training.losses import LossBundle, attach_losses, symbol_losses

        if "examples" in batch:
            outputs = [self.forward(model, row) for row in batch["examples"]]
            return attach_losses(SimpleNamespace(), LossBundle.combine([o.losses for o in outputs]))
        if "codec_labels" not in batch:
            return super().forward(model, batch)
        kwargs = {k: v for k, v in batch.items() if k not in ("codec_labels", "speech_spans")}
        output = model.thinker(**kwargs, use_cache=False, output_hidden_states=True)
        start, end = map(int, batch["speech_spans"][0])
        labels = batch["codec_labels"][0]
        primary, residual = talker_logits(
            model, talker_condition(model, batch, output, start, end), labels
        )
        primary_sum = torch.nn.functional.cross_entropy(
            primary.float(), labels[:, 0], reduction="sum"
        )
        residual_sum = torch.nn.functional.cross_entropy(
            residual.float().flatten(0, 1), labels[:-1, 1:].flatten(), reduction="sum"
        )
        output.text_loss = output.loss
        count = int((labels != -100).sum())
        output.primary_loss = primary_sum / len(labels)
        output.residual_loss = residual_sum / labels[:-1, 1:].numel()
        output.codec_loss = (primary_sum + residual_sum) / count
        text_count = int((batch["labels"][:, 1:] != -100).sum())
        return attach_losses(
            output,
            symbol_losses(
                output.text_loss * text_count,
                text_count,
                codec_sum=primary_sum + residual_sum,
                codec_count=count,
                text_weight=self.training_config.text_loss_weight,
                codec_weight=self.training_config.speech_loss_weight,
            ),
        )

    def generate_speech(self, model, batch, *, max_new_tokens, max_codec_tokens):
        from omnifactory.adapters.qwen3_speech import generate_talker, talker_condition

        ids = self.generate_ids(model, batch, max_new_tokens=max_new_tokens)
        start = batch["input_ids"].shape[1]
        new = ids[0, start:]
        eos = (new == self.processor.tokenizer.eos_token_id).nonzero()
        end = start + (int(eos[0]) if len(eos) else len(new))
        text = self.processor.tokenizer.decode(ids[0, start:end], skip_special_tokens=True)
        codes, end_token = [], None
        if end > start:
            full = {**batch, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
            output = model.thinker(**full, use_cache=False, output_hidden_states=True)
            codes, end_token = generate_talker(
                model, talker_condition(model, full, output, start, end), max_codec_tokens
            )
        return dict(
            text=text,
            codec_tokens=codes,
            text_finished=bool(len(eos)),
            speech_finished=end_token is not None,
            codec_eos_token_id=end_token,
            speaker=model.speech_speaker,
        )

    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None):
        if speech_assets is not None or reference_audio is not None:
            raise OmniFactoryError("E_SPEECH_INFERENCE", "Qwen3 uses native checkpoint speakers")
        from omnifactory.adapters.qwen3_speech import make_decoder

        return make_decoder(self.training_config)


def load_qwen3_thinker(config):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise OmniFactoryError("E_DEVICE", "CUDA was requested but is unavailable")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise OmniFactoryError("E_DEVICE", "MPS was requested but is unavailable")
    if (
        config.device == "cuda"
        and config.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise OmniFactoryError("E_DEVICE", "this CUDA device does not support bfloat16")
    source, _ = PretrainedConfig.get_config_dict(config.model, revision=config.revision)
    if source.get("model_type") not in ("qwen3_omni_moe", "qwen3_omni_moe_thinker"):
        raise OmniFactoryError("E_MODEL_TYPE", "this runtime requires Qwen3-Omni MoE weights")
    native = Qwen3OmniMoeThinkerConfig.from_dict(source.get("thinker_config", source))
    if native.text_config.num_hidden_layers > 48:
        raise OmniFactoryError(
            "E_MODEL_TYPE", "component mapping targets the 48-layer Qwen3 checkpoint"
        )
    if native.text_config.output_router_logits:
        raise OmniFactoryError(
            "E_ROUTER_LOSS", "router auxiliary training needs an explicit objective"
        )
    model, loading = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        config.model,
        revision=config.revision,
        config=native,
        dtype=getattr(torch, config.dtype),
        attn_implementation=config.attention,
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise OmniFactoryError("E_CHECKPOINT_INCOMPLETE", str(loading))
    model.to(config.device)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model
