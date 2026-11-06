# ruff: noqa: E402, F811
"""Cache boundaries preserve trainable projector graphs and reject stale state."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from test_packing import fixture  # noqa: F401

from omnifactory.adapters.qwen2_5 import load_thinker
from omnifactory.errors import OmniFactoryError
from omnifactory.training.feature_cache import cache_stats, install_feature_cache
from omnifactory.training.policy import apply_policy


def test_feature_hits_preserve_loss_and_projector_gradients(fixture, tmp_path):
    config, adapter, rows = fixture
    config = config.model_copy(
        update={
            "components": {
                "vision_projector": config.components["language_model"].model_validate(
                    {"mode": "full", "lr": 0.001}
                )
            },
        }
    )
    model, _, _ = apply_policy(load_thinker(config), config)
    model.train()
    batch = adapter.collate_padded(rows)
    expected = adapter.forward(model, batch).loss
    expected.backward()
    ref = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert all(p.abs().sum() > 0 for p in ref.values())
    model.zero_grad(set_to_none=True)
    cached_cfg = config.model_copy(update={"feature_cache": tmp_path / "cache"})
    install_feature_cache(model, cached_cfg)
    for _ in range(2):
        output = adapter.forward(model, batch).loss
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.backward()
        for n, p in model.named_parameters():
            if n in ref:
                torch.testing.assert_close(p.grad, ref[n], rtol=0, atol=0)
        model.zero_grad(set_to_none=True)
    assert all(s["hits"] == 1 and s["misses"] == 1 for s in cache_stats(model).values())
    # A trainable merger update must be applied on cache hits, not hidden by caching.
    with torch.no_grad():
        model.visual.merger.mlp[-1].bias.add_(0.1)
    assert not torch.equal(adapter.forward(model, batch).loss, expected)
    assert cache_stats(model)["vision_encoder"]["hits"] == 2


def test_cache_input_content_weights_and_policy_invalidation(fixture, tmp_path):
    config, adapter, rows = fixture
    config = config.model_copy(update={"feature_cache": tmp_path / "cache"})
    model, _, _ = apply_policy(load_thinker(config), config)
    row = rows[1]
    adapter.forward(model, row)
    original_contract = cache_stats(model)["vision_encoder"]["contract"]
    altered = deepcopy(row)
    altered["pixel_values"] += 0.25
    adapter.forward(model, altered)
    assert cache_stats(model)["vision_encoder"]["misses"] == 2
    with torch.no_grad():
        model.visual.patch_embed.proj.weight.add_(0.01)
    with pytest.raises(OmniFactoryError, match="E_FEATURE_CACHE_CHANGED"):
        adapter.forward(model, row)
    # A new binding hashes actual frozen weight values, not just a model pathname.
    other, _, _ = apply_policy(
        load_thinker(config.model_copy(update={"feature_cache": None})),
        config.model_copy(update={"feature_cache": None}),
    )
    with torch.no_grad():
        other.visual.patch_embed.proj.weight.add_(0.01)
    install_feature_cache(other, config)
    assert cache_stats(other)["vision_encoder"]["contract"] != original_contract
    other.audio_tower.conv1.weight.requires_grad_(True)
    with pytest.raises(OmniFactoryError, match="E_FEATURE_CACHE_CHANGED"):
        adapter.forward(other, rows[2])


def test_trainable_encoder_rejected_and_disk_payload_integrity(fixture, tmp_path):
    config, adapter, rows = fixture
    cfg = config.model_copy(update={"feature_cache": tmp_path / "cache"})
    with pytest.raises(OmniFactoryError, match="E_FEATURE_CACHE_BOUNDARY"):
        install_feature_cache(load_thinker(config), cfg)
    model, _, _ = apply_policy(load_thinker(cfg), cfg)
    adapter.forward(model, rows[1])
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    file = next((tmp_path / "cache").rglob("*.safetensors"))
    with safe_open(file, framework="pt") as f:
        metadata = f.metadata()
    tensors = load_file(file)
    tensors["hidden"] += 0.1
    save_file(tensors, file, metadata=metadata)
    with pytest.raises(OmniFactoryError, match="E_FEATURE_CACHE_HASH"):
        adapter.forward(model, rows[1])
