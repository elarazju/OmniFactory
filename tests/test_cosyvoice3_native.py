"""Native CosyVoice3 kernels before integrating the shared training adapter.

These are random CPU fixtures, not pretrained training or deployment evidence.
"""

import importlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("omegaconf")
pytest.importorskip("x_transformers")
import torch
from omegaconf import DictConfig
from transformers import Qwen2Config, Qwen2ForCausalLM

from omnifactory._vendor.cosyvoice3.flow.DiT.dit import DiT
from omnifactory._vendor.cosyvoice3.flow.flow import CausalMaskedDiffWithDiT
from omnifactory._vendor.cosyvoice3.flow.flow_matching import CausalConditionalCFM
from omnifactory._vendor.cosyvoice3.llm.llm import CosyVoice3LM, Qwen2Encoder
from omnifactory._vendor.cosyvoice3.transformer.upsample_encoder import PreLookaheadLayer
from omnifactory._vendor.cosyvoice3.utils.common import ras_sampling


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(2)
    torch.manual_seed(19)


def tiny_llm():
    encoder = Qwen2Encoder.__new__(Qwen2Encoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    return CosyVoice3LM(32, 32, 32, encoder, ras_sampling)


@pytest.mark.parametrize("bistream", [False, True])
def test_native_llm_gradients_and_causal_supervision(monkeypatch, bistream):
    native = importlib.import_module("omnifactory._vendor.cosyvoice3.llm.llm")
    monkeypatch.setattr(native.random, "random", lambda: 0.0 if bistream else 1.0)
    model = tiny_llm().eval()
    batch = dict(
        text_token=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        text_token_len=torch.tensor([6]),
        instruct_token=torch.tensor([[7, 8]]),
        instruct_token_len=torch.tensor([2]),
        speech_token=torch.arange(30).reshape(1, -1),
        speech_token_len=torch.tensor([30]),
    )
    captured = {}
    original = model.criterion_ce.forward

    def capture(logits, labels):
        captured.update(logits=logits.detach(), labels=labels.detach())
        return original(logits, labels)

    monkeypatch.setattr(model.criterion_ce, "forward", capture)
    result = model(batch, torch.device("cpu"))
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert model.llm.model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.speech_embedding.weight.grad.abs().sum() > 0
    assert model.llm_decoder.weight.grad.abs().sum() > 0
    labels, logits = captured["labels"].clone(), captured["logits"].clone()
    assert labels[0, -1] == model.eos_token
    assert (labels == model.fill_token).sum() == int(bistream)
    # Editing the final teacher-forced input must not change earlier predictions.
    changed = {**batch, "speech_token": batch["speech_token"].clone()}
    changed["speech_token"][0, -1] = 1
    model(changed, torch.device("cpu"))
    torch.testing.assert_close(captured["logits"][:, :-1], logits[:, :-1], atol=0, rtol=0)
    assert not torch.equal(captured["logits"][:, -1], logits[:, -1])


@pytest.mark.parametrize("streaming", [False, True])
def test_native_flow_backward_in_both_chunk_modes(monkeypatch, streaming):
    native = importlib.import_module("omnifactory._vendor.cosyvoice3.flow.flow")
    monkeypatch.setattr(native.random, "random", lambda: 0.25 if streaming else 0.75)
    model = CausalMaskedDiffWithDiT(
        input_size=80,
        output_size=80,
        spk_embed_dim=192,
        vocab_size=32,
        input_frame_rate=25,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        pre_lookahead_layer=PreLookaheadLayer(80, 32, 3),
        decoder=CausalConditionalCFM(
            in_channels=240,
            spk_emb_dim=80,
            cfm_params=DictConfig(
                dict(
                    sigma_min=1e-6,
                    solver="euler",
                    t_scheduler="cosine",
                    training_cfg_rate=0.0,
                    inference_cfg_rate=0.7,
                )
            ),
            estimator=DiT(
                dim=64,
                depth=2,
                heads=4,
                dim_head=16,
                ff_mult=2,
                mel_dim=80,
                mu_dim=80,
                spk_dim=80,
                out_channels=80,
                static_chunk_size=6,
                num_decoding_left_chunks=-1,
            ),
        ),
    )
    batch = dict(
        speech_token=torch.randint(0, 32, (2, 9)),
        speech_token_len=torch.tensor([9, 9]),
        speech_feat=torch.randn(2, 18, 80),
        speech_feat_len=torch.tensor([18, 18]),
        embedding=torch.randn(2, 192),
    )
    loss = model(batch, torch.device("cpu"))["loss"]
    assert loss.isfinite() and loss > 0
    loss.backward()
    for component in (
        model.input_embedding,
        model.spk_embed_affine_layer,
        model.pre_lookahead_layer,
        model.decoder.estimator,
    ):
        assert any(
            p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
            for p in component.parameters()
        )
    assert model.decoder.estimator.static_chunk_size == 6
