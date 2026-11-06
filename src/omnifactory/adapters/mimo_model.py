"""Native MiMo weights with explicit patch-causal, multi-codebook SFT.

The architecture and checkpoint names come from the pinned Xiaomi implementation.
Training uses one token-normalized objective across text and all RVQ channels.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from omnifactory._vendor.mimo_audio.modeling_mimo_audio import (
    MiMoAudioArguments,
    MiMoAudioConfig,
    MiMoAudioForCausalLM,
)


class MiMoTrainingModel(MiMoAudioForCausalLM):
    config_class = MiMoAudioConfig

    def apply_input_local_transformer(self, speech_embeddings):
        b, t, g, h = speech_embeddings.shape
        inputs = speech_embeddings.reshape(b * t, g, h)
        # Modern Transformers needs an explicit additive mask for noncausal patches.
        mask = inputs.new_zeros((b * t, 1, g, g)) if self.config.input_full_attention else None
        return self.input_local_transformer(
            inputs_embeds=inputs, attention_mask=mask, use_cache=False
        ).last_hidden_state.reshape(b, t, g, h)

    def _prepare_input_embeds(self, input_ids):
        b, _, frames = input_ids.shape
        g = self.group_size
        text_ids = input_ids[:, 0, ::g]
        speech_ids = input_ids[:, 1:].reshape(b, self.audio_channels, -1, g).transpose(1, 2)
        speech = text_ids == self.args.empty_idx
        embeds = self.speech_embeddings[0].weight.new_zeros(
            b, frames // g, g, self.config.input_local_dim
        )
        for i, table in enumerate(self.speech_embeddings):
            values = speech_ids[:, :, i]
            embeds = embeds + table(values).masked_fill(
                (values == self.speech_empty_ids[i]).unsqueeze(-1), 0
            )
        embeds = embeds * speech[..., None, None]
        embeds = self.apply_input_local_transformer(embeds) * speech[..., None, None]
        audio = self.speech_group_downcast(embeds.reshape(b, frames // g, -1))
        # Keep native text embedding behavior, including the learned empty embedding.
        return self.model.embed_tokens(text_ids) + audio

    def codec_logits(self, context, codes):
        """context [patches,H] predicts codes [patches,C,G], with native channel delay."""
        n, channels, g = codes.shape
        length = g + max(self.delay_pattern)
        embeds = self.speech_embeddings[0].weight.new_zeros(n, length, self.config.input_local_dim)
        for i in range(channels):
            delay = self.delay_pattern[i]
            embeds[:, delay : delay + g] += self.speech_embeddings[i](codes[:, i])
        if self.speech_embeddings_to_local is not None:
            embeds = self.speech_embeddings_to_local(embeds)
        inputs = torch.cat((context[:, None], embeds[:, :-1]), dim=1)
        hidden = self.local_transformer(inputs_embeds=inputs, use_cache=False).last_hidden_state
        return [
            head(hidden[:, delay : delay + g]).float()
            for head, delay in zip(self.local_transformer_lm_heads, self.delay_pattern)
        ]

    def forward(self, input_ids, attention_mask, labels=None, past_key_values=None, **kwargs):
        output = self.model(
            inputs_embeds=self._prepare_input_embeds(input_ids),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=labels is None,
        )
        hidden = output.last_hidden_state
        if labels is None:
            return SimpleNamespace(
                text_logits=self.lm_head(hidden[:, -1:]),
                local_hidden_states=self.hidden_states_downcast(hidden[:, -1:]),
                past_key_values=output.past_key_values,
            )
        g = self.group_size
        text_labels = labels[:, 0, ::g][:, 1:]
        text_logits = self.lm_head(hidden[:, :-1]).float()
        valid = text_labels != -100
        text_count = int(valid.sum())
        text_sum = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        # A global state at patch t-1 predicts both the text marker and patch t's codes.
        speech = (text_labels == self.args.empty_idx) & valid
        codec_sum = hidden.sum() * 0
        codec_count = 0
        if speech.any():
            codes = labels[:, 1:].reshape(labels.size(0), self.audio_channels, -1, g)
            codes = codes.transpose(1, 2)[:, 1:][speech]
            context = self.hidden_states_downcast(hidden[:, :-1][speech])
            for i, logits in enumerate(self.codec_logits(context, codes)):
                codec_sum = codec_sum + F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), codes[:, i].reshape(-1), reduction="sum"
                )
            codec_count = codes.numel()
        tw = getattr(self, "text_loss_weight", 1.0)
        sw = getattr(self, "speech_loss_weight", 1.0)
        count = (text_count if tw else 0) + codec_count
        return SimpleNamespace(
            loss=(text_sum * tw + codec_sum * sw) / max(count, 1),
            codec_loss=codec_sum / max(codec_count, 1),
            codec_tokens=codec_count,
            text_loss=text_sum / max(text_count, 1),
            text_loss_sum=text_sum,
            codec_loss_sum=codec_sum,
        )


def model_args(tokenizer, model_path):
    token = tokenizer.convert_tokens_to_ids
    return MiMoAudioArguments(
        model_name_or_path=str(model_path),
        sosp_idx=token("<|sosp|>"),
        eosp_idx=token("<|eosp|>"),
        sostm_idx=token("<|sostm|>"),
        eostm_idx=token("<|eostm|>"),
        eot_idx=token("<|eot|>"),
        empty_idx=token("<|empty|>"),
    )


def load_mimo(config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    native = MiMoAudioConfig.from_pretrained(config.model, revision=config.revision)
    native._attn_implementation = config.attention
    model = MiMoTrainingModel.from_pretrained(
        config.model,
        revision=config.revision,
        config=native,
        args=model_args(tokenizer, config.model),
        torch_dtype=getattr(torch, config.dtype),
    ).to(config.device)
    model.text_loss_weight = config.text_loss_weight
    model.speech_loss_weight = config.speech_loss_weight
    return model
