"""The precision policy preserves BF16 storage and a genuine native batch path."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
import torch

from omnifactory.adapters.contracts import check_adapter_contract
from omnifactory.adapters.minicpm import MiniCPMAdapter, load_minicpm
from omnifactory.adapters.minicpm_precision import configure_minicpm_precision
from omnifactory.adapters.runtime import adapter_spec
from omnifactory.data.io import iter_samples
from omnifactory.datasets.demo import create_demo
from omnifactory.training.config import TrainConfig
from omnifactory.training.policy import apply_policy, merge_lora
from omnifactory.training.tiny_minicpm import create_tiny_minicpm


@pytest.fixture
def precision_runtime(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    create_tiny_minicpm(tmp_path / "model")
    create_demo(tmp_path / "data")
    cfg = TrainConfig(
        adapter="minicpm_o_4_5",
        model=str(tmp_path / "model"),
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "run",
        max_length=512,
        heterogeneous_batches=True,
        components={"language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.001}},
        budget={
            "max_examples": 4,
            "max_llm_tokens": 2048,
            "max_vision_patches": 256,
            "max_audio_frames": 1600,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 2000000,
        },
    )
    # Public configuration is CUDA-only. Exercise the same arithmetic on CPU
    # with a tiny native model, without treating it as formal-weight evidence.
    model, _, _ = apply_policy(load_minicpm(cfg).bfloat16(), cfg)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    policy = SimpleNamespace(minicpm_precision="fp32_decoder_compute")
    configure_minicpm_precision(model, policy)
    return cfg, model, before, policy


def test_precision_config_scope_and_contracts(tmp_path):
    raw = {
        "model": "unused",
        "adapter": "minicpm_o_4_5",
        "dataset": str(tmp_path / "data.jsonl"),
        "output_dir": str(tmp_path / "run"),
        "components": {"language_model": {"mode": "lora", "rank": 8, "alpha": 16, "lr": 0.0001}},
        "budget": {
            "max_examples": 3,
            "max_llm_tokens": 2048,
            "max_vision_patches": 256,
            "max_audio_frames": 1600,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 2000000,
        },
        "device": "cuda",
        "dtype": "bfloat16",
    }
    native = TrainConfig.model_validate(raw)
    stable = TrainConfig.model_validate({**raw, "minicpm_precision": "fp32_decoder_compute"})
    assert native.heterogeneous_batches is False
    assert "minicpm_precision" not in native.execution_contract()
    assert stable.execution_contract()["minicpm_precision"] == "fp32_decoder_compute"
    assert native.encoding_contract() == stable.encoding_contract()
    for change in [
        {"adapter": "qwen2_5_omni"},
        {"dtype": "float32"},
        {"attention": "eager"},
        {"components": {"audio_projector": {"mode": "full", "lr": 0.0001}}},
    ]:
        with pytest.raises(ValueError, match="minicpm_precision requires"):
            TrainConfig.model_validate({**stable.model_dump(), **change})


def test_mlp_oracle_storage_gradients_and_autocast(precision_runtime):
    _, model, before, policy = precision_runtime
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.float(), before[name].float())
        assert parameter.dtype == (torch.float32 if ".lora_" in name else torch.bfloat16)
    layer = model.llm.model.layers[0]
    projection = layer.mlp.up_proj
    x = torch.linspace(-0.98765, 1.23456, projection.in_features * 3).reshape(1, 3, -1)
    x.requires_grad_()
    # A double-precision oracle also catches accidentally re-rounding the input
    # or output, including under an outer autocast scope.
    oracle_x = x.detach().double().requires_grad_()
    oracle = oracle_x @ projection.weight.detach().double().T
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = projection(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), oracle, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    oracle.square().sum().backward()
    torch.testing.assert_close(x.grad.double(), oracle_x.grad, atol=2e-6, rtol=2e-6)
    native = torch.nn.functional.linear(x.detach().bfloat16(), projection.weight).float()
    assert (native - actual.detach()).abs().max() > 1e-5
    assert layer.self_attn.q_proj.base_layer.weight.dtype == torch.bfloat16
    assert layer.self_attn.v_proj.base_layer.weight.dtype == torch.bfloat16
    handles = len(model.llm._forward_pre_hooks)
    configure_minicpm_precision(model, policy)
    assert len(model.llm._forward_pre_hooks) == handles


def test_native_mixed_gradients_and_input_ids_reload(precision_runtime, tmp_path):
    cfg, model, _, _ = precision_runtime
    adapter = MiniCPMAdapter.from_config(cfg)
    examples = [
        adapter.encode(s, s.predictions[0].id, contract_id="precision")
        for s in list(iter_samples(cfg.dataset))[:3]
    ]
    shapes = []
    handle = model.llm.register_forward_pre_hook(
        lambda m, args, kwargs: shapes.append(list(kwargs["inputs_embeds"].shape)), with_kwargs=True
    )
    check_adapter_contract(
        adapter_spec(cfg.adapter),
        cfg.task,
        adapter,
        model,
        examples,
        config=cfg,
        atol=0.005,
        rtol=0.05,
    )
    handle.remove()
    assert shapes[-1][0] == 3
    model.eval()
    ids = examples[0].payload["input_ids"]
    with torch.no_grad():
        a = model.llm(input_ids=ids, use_cache=False).logits
        b = model.llm(ids, use_cache=False).logits
        c = model.llm(inputs_embeds=model.llm.get_input_embeddings()(ids), use_cache=False).logits
        with torch.autocast("cpu", dtype=torch.bfloat16):
            d = model.llm(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a, c, atol=0, rtol=0)
    torch.testing.assert_close(a, d, atol=0, rtol=0)
    assert d.dtype == torch.float32
    batch = adapter.to_device(adapter.collate([e.payload for e in examples]), model, cfg)
    assert len(adapter.generate(model, batch, max_new_tokens=2)) == 3
    # Serialization remains native: runtime policy creates no tensor key and no
    # custom architecture requirement. LoRA merge leaves BF16 base storage.
    merged = merge_lora(model)
    assert all("lora_" not in n for n in merged.state_dict())
    assert not any("precision" in n for n in merged.state_dict())
    merged.save_pretrained(tmp_path / "export")
    reloaded = load_minicpm(cfg.model_copy(update={"model": str(tmp_path / "export")}))
    assert set(reloaded.state_dict()) == set(merged.state_dict())
    assert not hasattr(reloaded.llm, "_omnifactory_precision_hook")


def test_precision_rejects_tf32(precision_runtime):
    _, model, _, policy = precision_runtime
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises(Exception, match="E_MINICPM_PRECISION"):
            configure_minicpm_precision(model, policy)
        with pytest.raises(Exception, match="E_MINICPM_PRECISION"):
            model.llm(input_ids=torch.zeros((1, 2), dtype=torch.long))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
