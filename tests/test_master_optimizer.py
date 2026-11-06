import copy

import pytest

pytest.importorskip("torch")
import torch

from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.optimizer import MasterAdamW


def test_master_accumulates_updates_lost_by_bf16_and_matches_fp32_adamw():
    native = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    projected = torch.nn.Parameter(native.detach().clone())
    reference = torch.nn.Parameter(native.detach().float())
    low = torch.optim.AdamW([native], lr=1e-4, weight_decay=0)
    full = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0)
    master = MasterAdamW([dict(params=[projected], lr=1e-4)])
    for _ in range(32):
        for p in (native, projected, reference):
            p.grad = torch.ones_like(p)
        for optimizer in (low, master, full):
            optimizer.step()
    assert torch.equal(native, torch.ones_like(native))
    assert not torch.equal(projected, native)
    torch.testing.assert_close(master.inner.param_groups[0]["params"][0], reference, atol=0, rtol=0)
    assert torch.equal(projected, reference.bfloat16())
    assert all(
        s["exp_avg"].dtype == torch.float32 and s["exp_avg_sq"].dtype == torch.float32
        for s in master.inner.state.values()
    )


def test_resume_retains_hidden_master_bits_lr_schedule_and_missing_gradient():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    q = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    opt = MasterAdamW([dict(params=[p], lr=1e-4), dict(params=[q], lr=2e-4)])

    def advance(opt, params, start, end):
        for step in range(start, end):
            for i, (group, param) in enumerate(zip(opt.param_groups, params, strict=True)):
                group["lr"] = (i + 1) * 1e-4 / (step + 1)
                param.grad = None if i == 1 and step % 2 else torch.full_like(param, step / 10 + 1)
            opt.step()
            opt.zero_grad(set_to_none=True)

    advance(opt, [p, q], 0, 7)
    saved = copy.deepcopy(opt.state_dict())
    restored = [torch.nn.Parameter(v.detach().clone()) for v in (p, q)]
    other = MasterAdamW([dict(params=[v], lr=99) for v in restored])
    other.load_state_dict(saved)
    advance(opt, [p, q], 7, 23)
    advance(other, restored, 7, 23)
    for a, b in zip((p, q), restored, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(
        opt.state_dict()["master_weights"], other.state_dict()["master_weights"], strict=True
    ):
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)
    for a, b in zip(opt.inner.state.values(), other.inner.state.values(), strict=True):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(a[key], b[key])
    saved["master_weights"][0][0].add_(1)
    with pytest.raises(OmniFactoryError, match="projection"):
        other.load_state_dict(saved)
    with pytest.raises(OmniFactoryError, match="original master"):
        other.load_state_dict({"state": {}, "param_groups": []})


def test_master_is_execution_only_and_default_preserves_old_contract():
    cfg = TrainConfig.model_validate(
        dict(
            model="tiny",
            dataset="data.jsonl",
            output_dir="out",
            components={"language_model": dict(mode="full", lr=1e-4)},
            budget=dict(
                max_examples=1,
                max_llm_tokens=512,
                max_vision_patches=1,
                max_audio_frames=1,
                max_codec_frames=1,
                max_decoded_media_bytes=1000000,
            ),
        )
    )
    enabled = cfg.model_copy(update={"optimizer_master_weights": True})
    assert cfg.encoding_contract() == enabled.encoding_contract()
    assert "optimizer_master_weights" not in cfg.execution_contract()
    assert enabled.execution_contract()["optimizer_master_weights"] is True


def test_production_checkpoint_restores_masters_and_rejects_native_state(tmp_path):
    from pathlib import Path

    from safetensors.torch import load_file

    from omnifactory.datasets.demo import create_demo
    from omnifactory.training.engine import train
    from omnifactory.training.tiny import create_tiny

    torch.set_num_threads(2)
    create_tiny(tmp_path / "model")
    create_demo(tmp_path / "data")
    cfg = TrainConfig(
        model=str(tmp_path / "model"),
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "native",
        max_steps=4,
        save_steps=2,
        max_length=512,
        components={"language_model": dict(mode="lora", rank=2, alpha=4, lr=1e-3)},
        budget=dict(
            max_examples=2,
            max_llm_tokens=1024,
            max_vision_patches=128,
            max_audio_frames=400,
            max_codec_frames=1,
            max_decoded_media_bytes=1000000,
        ),
    )
    native = train(cfg)
    master_cfg = cfg.model_copy(
        update=dict(output_dir=tmp_path / "master", optimizer_master_weights=True)
    )
    master = train(master_cfg)
    resumed = train(
        master_cfg.model_copy(
            update=dict(
                output_dir=tmp_path / "resumed", resume_from=tmp_path / "master/checkpoint-00000002"
            )
        )
    )
    reference = load_file(Path(master["checkpoint"]) / "trainable.safetensors")
    for result in (native, resumed):
        actual = load_file(Path(result["checkpoint"]) / "trainable.safetensors")
        assert reference.keys() == actual.keys()
        assert all(torch.equal(v, actual[k]) for k, v in reference.items())
    state = torch.load(Path(resumed["checkpoint"]) / "optimizer.pt", weights_only=True)["optimizer"]
    assert state["format"] == "omnifactory-fp32-master-adamw-v1"
    with pytest.raises(OmniFactoryError, match="E_RESUME_CONTRACT"):
        train(
            master_cfg.model_copy(
                update=dict(
                    output_dir=tmp_path / "wrong",
                    resume_from=tmp_path / "native/checkpoint-00000002",
                )
            )
        )
