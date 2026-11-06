"""Native tiny-model mixed batching: loss, gradients, isolation and rejection."""

import copy
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
import torch

from omnifactory.adapters.contracts import (
    check_adapter_contract,
    collate_task,
    validate_batch,
    validate_encoded,
)
from omnifactory.adapters.runtime import adapter_class, adapter_spec, load_model
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.demo import create_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.tiny import create_tiny
from omnifactory.training.tiny_minicpm import create_tiny_minicpm


@pytest.fixture(params=["qwen2_5_omni", "minicpm_o_4_5"])
def native(request, tmp_path):
    torch.set_num_threads(2)
    name = request.param
    (create_tiny if name == "qwen2_5_omni" else create_tiny_minicpm)(tmp_path / "model")
    create_demo(tmp_path / "data")
    cfg = TrainConfig(
        adapter=name,
        model=str(tmp_path / "model"),
        dataset=tmp_path / "data/train.jsonl",
        output_dir=tmp_path / "run",
        max_length=512,
        heterogeneous_batches=True,
        components={"language_model": {"mode": "full", "lr": 0.001}},
        budget={
            "max_examples": 4,
            "max_llm_tokens": 2048,
            "max_vision_patches": 256,
            "max_audio_frames": 1600,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 2000000,
        },
    )
    adapter, model = adapter_class(name).from_config(cfg), load_model(cfg).eval()
    samples = list(iter_samples(cfg.dataset))[:3]
    raw = samples[2].model_dump(mode="json")
    audio = next(b for b in raw["messages"][0]["content"] if b["type"] == "audio")
    raw["messages"][0]["content"].append(
        {**audio, "id": "second-audio", "media": {**audio["media"], "end_us": 150000}}
    )
    raw["messages"][-1]["content"][0]["text"] += " with a longer answer."
    samples.append(CanonicalSample.model_validate(raw))
    examples = [adapter.encode(s, "p0", contract_id="mixed-native") for s in samples]
    return cfg, adapter, model, samples, examples


def test_native_mixed_loss_gradients_and_forward_group(native, tmp_path):
    cfg, adapter, model, _, examples = native
    calls = []
    decoder = model if cfg.adapter == "qwen2_5_omni" else model.llm
    handle = decoder.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(
            len(kwargs.get("input_ids", kwargs.get("inputs_embeds")))
        ),
        with_kwargs=True,
    )
    report = check_adapter_contract(
        adapter_spec(cfg.adapter),
        cfg.task,
        adapter,
        model,
        examples,
        config=cfg,
        atol=2e-5,
        rtol=2e-5,
    )
    handle.remove()
    assert calls == [1, 1, 1, 1, 4]
    assert report["execution"]["forward_groups"] == [4]
    (tmp_path / "numerics.json").write_text(json.dumps(report, indent=2))
    print(cfg.adapter, json.dumps(report))


def test_no_cross_sample_attention_media_or_target_leak(native):
    cfg, adapter, model, samples, examples = native
    rows = [e.payload for e in examples]
    with torch.no_grad():
        baseline = adapter.forward(model, adapter.collate(rows))
        changed = copy.deepcopy(rows)
        changed[-1]["input_ids"] = changed[-1]["input_ids"].clone()
        # Perturb another sample's ordinary tokens and targets without disturbing media placeholders.
        changed[-1]["input_ids"][0, -2] = changed[-1]["input_ids"][0, -3]
        changed[-1]["labels"].fill_(-100)
        if cfg.adapter == "qwen2_5_omni":
            changed[-1]["input_features"] *= 0
        else:
            changed[-1]["audio_features"] *= 0
        perturbed = adapter.forward(model, adapter.collate(changed))
    torch.testing.assert_close(baseline.logits[:-1], perturbed.logits[:-1], atol=0, rtol=0)
    raw = samples[1].model_dump(mode="json")
    raw["messages"][-1]["content"][0]["text"] = "different target"
    before = adapter.encode(samples[1], "p0", contract_id="inference", inference=True)
    after = adapter.encode(
        CanonicalSample.model_validate(raw), "p0", contract_id="inference", inference=True
    )
    validate_encoded(adapter_spec(cfg.adapter), cfg.task, before, inference=True)
    for key, value in before.tensors.items():
        torch.testing.assert_close(value, after.tensors[key], atol=0, rtol=0)


def test_negative_batch_declarations_masks_contracts(native, monkeypatch):
    cfg, adapter, _, _, examples = native
    bad = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1], [0, 1, 1]]),
        "labels": torch.ones(2, 3, dtype=torch.long),
    }
    with pytest.raises(OmniFactoryError, match="E_BATCH_MASK"):
        validate_batch(bad)
    costs = [e.cost for e in examples]
    costs[-1] = costs[-1].model_copy(update={"contract_id": "other"})
    with pytest.raises(OmniFactoryError, match="E_BATCH_CONTRACT"):
        collate_task(adapter, cfg, [e.payload for e in examples], costs)
    monkeypatch.setattr(adapter, "collate", lambda rows: {"examples": rows})
    with pytest.raises(OmniFactoryError, match="E_BATCH_CAPABILITY"):
        collate_task(adapter, cfg, [e.payload for e in examples], [e.cost for e in examples])


def test_continuous_and_timed_shared_validation_and_normalization(monkeypatch):
    from dataclasses import replace

    from test_supervision_runtime import ReferenceAdapter, aligned_example

    from omnifactory.adapters import runtime
    from omnifactory.adapters.batching import TaskBatchCapability

    source = adapter_spec("minicpm_o_4_5")
    cap = TaskBatchCapability(
        mode="padded", heterogeneous=True, supervision=("discrete", "continuous"), timed=True
    )
    spec = replace(
        source,
        name="contract_mixed_fixture",
        loaders={"understanding_sft": source.loaders["understanding_sft"]},
        loss_components={"understanding_sft": ("language_model",)},
        batch_capabilities={"understanding_sft": cap},
    )
    monkeypatch.setitem(runtime._REGISTRY, spec.name, spec)
    a = aligned_example()
    # Include actual timed streams in shared validation; the linear probe has no attention.
    validate_encoded(spec, "understanding_sft", a)
    bad = replace(spec, batch_capabilities={"understanding_sft": replace(cap, timed=False)})
    with pytest.raises(OmniFactoryError, match="E_ADAPTER_SUPERVISION"):
        validate_encoded(bad, "understanding_sft", a)
    bad = replace(
        spec, batch_capabilities={"understanding_sft": replace(cap, supervision=("discrete",))}
    )
    with pytest.raises(OmniFactoryError, match="E_ADAPTER_SUPERVISION"):
        validate_encoded(bad, "understanding_sft", a)
    # Two unequal records retain separate CE symbols and duration-weighted latent measures.
    from test_supervision_runtime import mixed_plan

    from omnifactory.adapters.base import EncodedExample

    examples = []
    for i, n in enumerate((3, 5)):
        tensors = {
            "features": torch.randn(n, 3),
            "symbols": torch.randint(0, 3, (n,)),
            "text_valid": torch.arange(n) < i + 1,
            "latents": torch.randn(n, 2),
            "audio_valid": torch.ones(n, dtype=torch.bool),
            "duration": torch.arange(1, n + 1, dtype=torch.float64) * (0.1 if i else 0.7),
        }
        examples.append(
            EncodedExample(
                tensors=tensors,
                cost=a.cost.model_copy(
                    update={"sample_id": str(i), "latent_elements": n * 2, "stream_events": 0}
                ),
                supervision=mixed_plan(tensors),
            )
        )
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        adapter=spec.name,
        task="understanding_sft",
        packing="none",
        duplex_schedule=None,
        heterogeneous_batches=True,
    )
    report = check_adapter_contract(
        spec, cfg.task, ReferenceAdapter(), torch.nn.Linear(3, 5), examples, config=cfg
    )
    assert report["execution"]["forward_groups"] == [2]
    assert report["gradient_max_absolute_error"] < 2e-5


def test_gradient_contract_does_not_pass_disconnected_parameters(native):
    cfg, adapter, _, _, examples = native
    model = torch.nn.Linear(1, 1)

    class DetachedAdapter:
        collate = adapter.collate
        to_device = staticmethod(lambda batch, model, config: batch)

        def forward(self, model, batch):
            from types import SimpleNamespace

            from omnifactory.training.losses import symbol_losses

            count = int((batch["labels"][:, 1:] != -100).sum())
            unrelated = torch.tensor(1.0, requires_grad=True)
            return SimpleNamespace(losses=symbol_losses(unrelated * count, count))

    with pytest.raises(OmniFactoryError, match="E_ADAPTER_GRADIENT"):
        check_adapter_contract(
            adapter_spec(cfg.adapter), cfg.task, DetachedAdapter(), model, examples, config=cfg
        )


def test_multicodebook_layout_negative_masks_and_axes():
    from types import SimpleNamespace

    adapter = SimpleNamespace(g=2, channels=3)
    batch = {
        "input_ids": torch.zeros(1, 4, 6, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "labels": torch.full((1, 4, 6), -100, dtype=torch.long),
    }
    validate_batch(batch, layout="mimo_codebooks", adapter=adapter)
    bad = {**batch, "input_ids": batch["input_ids"][:, :3]}
    with pytest.raises(OmniFactoryError, match="E_BATCH_SHAPE"):
        validate_batch(bad, layout="mimo_codebooks", adapter=adapter)
    batch["labels"][0, 1, -1] = 1
    with pytest.raises(OmniFactoryError, match="E_BATCH_MASK"):
        validate_batch(batch, layout="mimo_codebooks", adapter=adapter)
    from omnifactory.adapters.qwen3_tts import pack_training

    # Use a small standalone native config; no weights or tokenizer are required.
    cfg = SimpleNamespace(
        tts_pad_token_id=1,
        tts_bos_token_id=2,
        tts_eos_token_id=3,
        talker_config=SimpleNamespace(
            codec_nothink_id=4,
            codec_think_bos_id=5,
            codec_think_eos_id=6,
            codec_pad_id=7,
            codec_bos_id=8,
            codec_eos_token_id=9,
        ),
    )
    batch = pack_training(
        torch.ones(1, 8, dtype=torch.long), torch.ones(2, 16, dtype=torch.long), cfg
    )
    validate_batch(batch, layout="qwen3_tts_pairs")
    bad = {**batch, "residual_labels": batch["residual_labels"][:, :14]}
    with pytest.raises(OmniFactoryError, match="E_BATCH_SHAPE"):
        validate_batch(bad, layout="qwen3_tts_pairs")


def flow_fixture(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from test_supervision_runtime import ReferenceAdapter, mixed_plan

    from omnifactory.adapters import runtime
    from omnifactory.adapters.base import EncodedExample
    from omnifactory.adapters.batching import TaskBatchCapability
    from omnifactory.training.losses import LossBundle

    source = adapter_spec("minicpm_o_4_5")
    cap = TaskBatchCapability(
        mode="padded", heterogeneous=True, supervision=("discrete", "continuous")
    )
    spec = replace(
        source,
        name="flow_probe_fixture",
        loaders={"understanding_sft": source.loaders["understanding_sft"]},
        loss_components={"understanding_sft": ("language_model",)},
        batch_capabilities={"understanding_sft": cap},
    )
    monkeypatch.setitem(runtime._REGISTRY, spec.name, spec)
    examples = []
    for i, n in enumerate((2, 5)):
        tensors = {
            "features": torch.randn(n, 3),
            "symbols": torch.randint(0, 3, (n,)),
            "text_valid": torch.ones(n, dtype=torch.bool),
            "latents": torch.randn(n, 2),
            "audio_valid": torch.ones(n, dtype=torch.bool),
            "duration": torch.ones(n, dtype=torch.float64),
            "noise": torch.randn(n, 3),
            "time": torch.rand(n, 1),
        }
        plan = mixed_plan(tensors)
        objectives = tuple(
            o.model_copy(update={"kind": "flow_matching"}) if o.id == "latent_mse" else o
            for o in plan.objectives
        )
        plan = plan.model_copy(update={"objectives": objectives})
        from omnifactory.planning.batching import SampleCost

        examples.append(
            EncodedExample(
                tensors,
                SampleCost(
                    sample_id=str(i),
                    prediction_id="p",
                    contract_id="fixed-flow",
                    modalities=("audio",),
                    latent_elements=n * 2,
                ),
                supervision=plan,
            )
        )
    cfg = SimpleNamespace(
        adapter=spec.name,
        task="understanding_sft",
        packing="none",
        duplex_schedule=None,
        heterogeneous_batches=True,
    )
    adapter, model = ReferenceAdapter(), torch.nn.Linear(3, 5)
    with pytest.raises(OmniFactoryError, match="E_ADAPTER_PROBE"):
        check_adapter_contract(spec, cfg.task, adapter, model, examples, config=cfg)

    def probe_factory(bound_adapter):
        def controlled(model, batch):
            # Bind only this adapter; the model is always the explicit call argument.
            fixed = {
                **batch,
                "features": batch["features"] * batch["time"]
                + batch["noise"] * (1 - batch["time"]),
            }
            result = bound_adapter.forward(model, fixed)
            result.losses = LossBundle(
                tuple(
                    replace(term, kind="flow_matching") if term.id == "latent_mse" else term
                    for term in result.losses.terms
                ),
                result.losses.counts,
                result.losses.units,
            )
            return result

        return controlled

    return spec, cfg, adapter, model, examples, probe_factory


def test_controlled_flow_probe_required_and_checked(monkeypatch):
    spec, cfg, adapter, model, examples, probe_factory = flow_fixture(monkeypatch)
    report = check_adapter_contract(
        spec,
        cfg.task,
        adapter,
        model,
        examples,
        config=cfg,
        forward_probe=probe_factory(adapter),
        reload_probe_factory=probe_factory,
    )
    assert report["gradient_max_absolute_error"] < 2e-5


def test_controlled_flow_reload_uses_new_adapter_and_model(monkeypatch):
    from test_supervision_runtime import ReferenceAdapter

    spec, cfg, adapter, model, examples, probe_factory = flow_fixture(monkeypatch)
    reload_calls = []

    class ReloadAdapter(ReferenceAdapter):
        def forward(self, reloaded, batch):
            assert reloaded is not model
            reload_calls.append(id(reloaded))
            return super().forward(reloaded, batch)

    def reload_model(changed=False):
        reloaded = copy.deepcopy(model)
        if changed:
            with torch.no_grad():
                reloaded.weight.add_(10)
                reloaded.bias.add_(5)
        return ReloadAdapter(), reloaded

    rng = torch.get_rng_state().clone()
    previous = torch.ones_like(model.weight)
    model.weight.grad = previous
    model.train()
    report = check_adapter_contract(
        spec,
        cfg.task,
        adapter,
        model,
        examples,
        config=cfg,
        forward_probe=probe_factory(adapter),
        reload_probe_factory=probe_factory,
        export_reload=reload_model,
    )
    assert report["export_reload"] == "passed" and len(reload_calls) == len(examples)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training and model.weight.grad is previous
    with pytest.raises(AssertionError):
        check_adapter_contract(
            spec,
            cfg.task,
            adapter,
            model,
            examples,
            config=cfg,
            forward_probe=probe_factory(adapter),
            reload_probe_factory=probe_factory,
            export_reload=lambda: reload_model(changed=True),
        )
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training and model.weight.grad is previous
    with pytest.raises(OmniFactoryError, match="reload_probe_factory"):
        check_adapter_contract(
            spec,
            cfg.task,
            adapter,
            model,
            examples,
            config=cfg,
            forward_probe=probe_factory(adapter),
            export_reload=reload_model,
        )


def test_shared_native_export_reload_contract(native, tmp_path):
    cfg, adapter, model, _, examples = native
    target = tmp_path / "native-reload"

    def reload_native():
        model.save_pretrained(target)
        adapter.save_processor(target)
        restored = cfg.model_copy(update={"model": str(target)})
        return adapter_class(cfg.adapter).from_config(restored), load_model(restored)

    report = check_adapter_contract(
        adapter_spec(cfg.adapter),
        cfg.task,
        adapter,
        model,
        examples,
        config=cfg,
        export_reload=reload_native,
    )
    assert report["export_reload"] == "passed"


def test_calibrated_budget_production_train_backward(native, tmp_path):
    """CPU native integration validates artifact plumbing, not a real VRAM calibration."""
    from omnifactory.planning.batching import MemoryProfile
    from omnifactory.training.benchmark import runtime_hashes
    from omnifactory.training.calibration import configured_planner, recipe_signature
    from omnifactory.training.efficiency import environment
    from omnifactory.training.engine import train
    from omnifactory.training.prepare import source_fingerprint

    cfg, _, _, _, _ = native
    profile_path = tmp_path / "memory-profile.json"
    # Public config validation requires CUDA for physical memory calibration. Here
    # model_copy deliberately runs its accepted-artifact production path on CPU.
    cfg = cfg.model_copy(
        update={
            "memory_profile": profile_path,
            "budget": cfg.budget.model_copy(update={"max_gpu_bytes": 10**12}),
            "max_steps": 2,
            "save_steps": 1,
        }
    )
    source = source_fingerprint(cfg.model, cfg.revision)
    profile = MemoryProfile(
        profile_id="cpu-calibration-plumbing-fixture",
        contract_id="rebound-at-load",
        hardware="cpu-fixture",
        provenance="calibrated",
        fixed_bytes=100,
        bytes_per_llm_token=2,
        bytes_per_vision_patch=3,
        bytes_per_audio_frame=4,
        bytes_per_codec_frame=1,
        bytes_per_attention_pair=1,
    )
    artifact = {
        "format": "omnifactory-memory-calibration-v1",
        "accepted": True,
        "profile": profile.model_dump(mode="json"),
        "recipe_signature": recipe_signature(cfg, source),
        "environment": environment(cfg.device),
        "runtime_hashes": runtime_hashes(),
    }
    profile_path.write_text(json.dumps(artifact))
    result = train(cfg)
    assert result["steps"] == 2
    rows = [
        json.loads(line) for line in (cfg.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    for row in rows:
        assert row["loss"] > 0 and row["grad_norm"] > 0
        for execution in row["batch_execution"]:
            assert execution["budget_cost"]["estimated_gpu_bytes"] > 100
            assert execution["budget_cost"]["examples"] == execution["examples"]
    # All pre-existing production gates remain enforced before a forward.
    profile_path.unlink()
    with pytest.raises(FileNotFoundError):
        configured_planner(cfg, "contract", source)
    bad = copy.deepcopy(artifact)
    for key, value, match in [
        ("accepted", False, "accepted calibration"),
        ("recipe_signature", "wrong", "recipe differs"),
        ("environment", {}, "environment changed"),
        ("runtime_hashes", {}, "source changed"),
    ]:
        profile_path.write_text(json.dumps({**bad, key: value}))
        with pytest.raises(OmniFactoryError, match=match):
            configured_planner(cfg, "contract", source)
    missing = cfg.model_copy(update={"memory_profile": None})
    with pytest.raises(OmniFactoryError, match="E_MEMORY_PROFILE"):
        configured_planner(missing, "contract", source)
