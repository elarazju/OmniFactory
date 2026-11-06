"""Qwen Thinker packing with native per-example mRoPE and isolated varlen attention.

CUDA uses FlashAttention's varlen kernel. CPU segmented SDPA is a correctness
reference only; production configuration requires CUDA/bfloat16.
"""

from copy import deepcopy

import torch

from omnifactory.errors import OmniFactoryError

BACKEND = "omnifactory_varlen"


def varlen_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    scaling=None,
    dropout=0.0,
    sliding_window=None,
    packing_cu_seqlens=None,
    packing_max_seqlen=None,
    **kwargs,
):
    if packing_cu_seqlens is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
    if query.shape[0] != 1 or attention_mask is not None or sliding_window is not None:
        raise OmniFactoryError("E_PACKING_ATTENTION", "requires one packed full-attention stream")
    q, k, v = [x.squeeze(0).transpose(0, 1).contiguous() for x in (query, key, value)]
    cu = packing_cu_seqlens
    if query.is_cuda:
        from flash_attn import flash_attn_varlen_func

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu,
            cu,
            packing_max_seqlen,
            packing_max_seqlen,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
            deterministic=True,
        )
    else:
        # Never construct a quadratic block-diagonal mask. This slow CPU path is
        # an oracle for unit tests, not a claimed efficient training backend.
        pieces = []
        for start, end in zip(cu.tolist()[:-1], cu.tolist()[1:]):
            result = torch.nn.functional.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                k[start:end].transpose(0, 1),
                v[start:end].transpose(0, 1),
                is_causal=True,
                dropout_p=dropout,
                scale=scaling,
                enable_gqa=True,
            )
            pieces.append(result.transpose(0, 1))
        output = torch.cat(pieces)
    return output.unsqueeze(0), None


def install_attention(thinker):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(BACKEND, varlen_attention)
    for layer in thinker.model.layers:
        attention = layer.self_attn
        if attention.sliding_window is not None:
            raise OmniFactoryError("E_PACKING_ATTENTION", "sliding attention is not supported")
        if attention.config._attn_implementation != BACKEND:
            # Preserve the serializable parent config and encoder attention.
            attention.config = deepcopy(attention.config)
            attention.config._attn_implementation = BACKEND


def packed_batch(thinker, rows):
    allowed = {
        "input_ids",
        "labels",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "input_features",
        "feature_attention_mask",
    }
    positions, lengths = [], []
    for row in rows:
        if set(row) - allowed or not bool(row["attention_mask"].all()):
            raise OmniFactoryError(
                "E_PACKING_FIELDS", "packing requires unpadded text/image/audio rows"
            )
        lengths.append(row["input_ids"].shape[1])
        pos, _ = thinker.get_rope_index(
            row["input_ids"],
            image_grid_thw=row.get("image_grid_thw"),
            attention_mask=row["attention_mask"],
            audio_seqlens=row["feature_attention_mask"].sum(-1)
            if "feature_attention_mask" in row
            else None,
        )
        positions.append(pos)
    # Media order follows sample order. The native towers retain their own
    # sequence boundaries; only the language stream is concatenated.
    media_rows = [
        {k: v for k, v in r.items() if k not in ("input_ids", "labels", "attention_mask")}
        for r in rows
    ]
    result = {}
    for key in set().union(*(r.keys() for r in media_rows)):
        values = [r[key] for r in media_rows if key in r]
        if key in ("input_features", "feature_attention_mask"):
            longest = max(v.shape[-1] for v in values)
            values = [torch.nn.functional.pad(v, (0, longest - v.shape[-1])) for v in values]
        result[key] = torch.cat(values, dim=0)
    result["input_ids"] = torch.cat([r["input_ids"] for r in rows], dim=1)
    labels = []
    for row in rows:
        label = row["labels"].clone()
        label[:, 0] = -100  # no target across a segment boundary
        labels.append(label)
    result["labels"] = torch.cat(labels, dim=1)
    result["position_ids"] = torch.cat(positions, dim=-1)
    result["attention_mask"] = {"full_attention": None}
    result["packing_cu_seqlens"] = torch.tensor(
        [0, *lengths], device=result["input_ids"].device, dtype=torch.int32
    ).cumsum(0, dtype=torch.int32)
    result["packing_max_seqlen"] = max(lengths)
    install_attention(thinker)
    return result
