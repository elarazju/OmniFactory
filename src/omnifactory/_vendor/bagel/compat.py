"""CPU reference attention; CUDA continues using native FlashAttention."""

import torch
import torch.nn.functional as F


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    if q.device.type == 'cuda':
        from flash_attn import flash_attn_varlen_func as native
        return native(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, **kwargs)
    if dropout_p or kwargs:
        raise ValueError('CPU reference supports plain attention without dropout')
    outputs = []
    for i in range(len(cu_seqlens_q) - 1):
        qi = q[int(cu_seqlens_q[i]):int(cu_seqlens_q[i+1])].transpose(0, 1).unsqueeze(0)
        ki = k[int(cu_seqlens_k[i]):int(cu_seqlens_k[i+1])].transpose(0, 1).unsqueeze(0)
        vi = v[int(cu_seqlens_k[i]):int(cu_seqlens_k[i+1])].transpose(0, 1).unsqueeze(0)
        mask = None
        if causal:
            nq, nk = qi.shape[-2], ki.shape[-2]
            mask = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + nk - nq
        out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=softmax_scale, enable_gqa=True)
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)
