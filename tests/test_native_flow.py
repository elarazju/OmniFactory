"""Actual upstream DiT + conformer gradients; no placeholder denoising network."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("stepaudio2")
import torch
from stepaudio2.cosyvoice2.flow.decoder_dit import DiT
from stepaudio2.cosyvoice2.flow.flow import CausalMaskedDiffWithXvec
from stepaudio2.cosyvoice2.flow.flow_matching import CausalConditionalCFM
from stepaudio2.cosyvoice2.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2

from omnifactory.adapters.minicpm_flow import MiniCPMFlowModel, flow_path, flow_plan
from omnifactory.adapters.supervision import expected_losses, normalization_counts


def tiny_flow():
    encoder = UpsampleConformerEncoderV2(
        input_size=32,
        output_size=32,
        num_blocks=1,
        num_up_blocks=1,
        attention_heads=2,
        linear_units=64,
        dropout_rate=0,
        positional_dropout_rate=0,
        attention_dropout_rate=0,
    )
    estimator = DiT(
        in_channels=320, out_channels=80, depth=1, num_heads=2, head_dim=16, hidden_size=32
    )
    decoder = CausalConditionalCFM(estimator)
    flow = CausalMaskedDiffWithXvec(
        input_size=32,
        output_size=80,
        spk_embed_dim=192,
        vocab_size=32,
        encoder=encoder,
        decoder=decoder,
    )
    for module in flow.modules():
        for key in ("att_cache_buffer", "cnn_cache_buffer"):
            if key in module._buffers:
                module._buffers[key] = torch.empty(0)
    return flow


def flow_batch():
    return {
        "mel": torch.randn(1, 12, 80),
        "condition": torch.zeros(1, 12, 80),
        "codes": torch.randint(0, 32, (1, 6)),
        "code_lengths": torch.tensor([6]),
        "valid": torch.ones(1, 12, dtype=torch.bool),
        "loss_mask": torch.tensor([[False] * 4 + [True] * 7 + [False]]),
        "speaker": torch.randn(1, 192),
    }


def test_conditional_path_derivative():
    clean, noise = torch.randn(2, 80, 4), torch.randn(2, 80, 4)
    t = torch.tensor([0.2, 0.7], requires_grad=True)
    noisy, velocity = flow_path(clean, noise, t, 1e-6)
    actual = torch.autograd.grad(noisy.sum(), t)[0]
    torch.testing.assert_close(actual, velocity.sum((1, 2)))
    endpoint, _ = flow_path(clean, noise, torch.ones(2), 0)
    torch.testing.assert_close(endpoint, clean)


def test_actual_native_flow_checkpointing_and_masked_backward():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = MiniCPMFlowModel(
        tiny_flow(), SimpleNamespace(seed=5, flow_sigma_min=1e-6, flow_cfg_dropout=0)
    )
    model.flow.requires_grad_(False)
    model.flow.decoder.estimator.requires_grad_(True)
    batch = flow_batch()
    noise, time = torch.randn(1, 80, 12), torch.tensor([0.4])
    result = model(batch, noise=noise, time=time)
    plan = flow_plan(batch)
    result.losses.validate_expected(expected_losses([(plan, normalization_counts(plan, batch))]))
    target = batch["mel"].transpose(1, 2) - (1 - 1e-6) * noise
    mask = batch["loss_mask"]
    expected = (
        (result.velocity.transpose(1, 2)[mask] - target.transpose(1, 2)[mask]).square().mean()
    )
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert any(g.abs().sum() > 0 for g in grads.values())
    assert all(n.startswith("flow.decoder.estimator.") for n in grads)
    model.zero_grad()
    model.gradient_checkpointing_enable()
    actual = model(batch, noise=noise, time=time)
    actual.loss.backward()
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n])
    model.eval()
    torch.testing.assert_close(model(batch).loss, model(batch).loss)
