"""Native random-model tests; not pretrained or realtime serving evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sphn")
import torch
import torch.nn.functional as F

from omnifactory._vendor.personaplex.models.lm import LMGen, _delay_sequence, _undelay_sequence
from omnifactory._vendor.personaplex.modules.transformer import RingKVCache
from omnifactory._vendor.personaplex.utils.compile import no_compile
from omnifactory.adapters.personaplex_model import duplex_loss
from omnifactory.adapters.personaplex_supervision import aligned_duplex_supervision
from omnifactory.adapters.supervision import expected_losses, normalization_counts
from omnifactory.data.supervision import SupervisionPlan
from omnifactory.errors import OmniFactoryError
from omnifactory.training.tiny_personaplex import tiny_personaplex


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(2)
    torch.manual_seed(123)


def codes():
    value = torch.randint(0, 32, (1, 17, 6))
    value[:, 0] %= 64
    return value


def test_ring_cache_positions_through_multiple_wraps():
    cache = RingKVCache(1, 1, 1, 4, device=torch.device("cpu"), dtype=torch.float32)
    for step in range(13):
        x = torch.tensor([[[[float(step)]]]])
        result = cache.complete(x, -x)
        valid = result.positions >= 0
        assert sorted(result.positions[valid].tolist()) == list(range(max(0, step - 3), step + 1))
        torch.testing.assert_close(result.keys[0, 0, valid, 0], result.positions[valid].float())


def test_vendor_manifest():
    from omnifactory._vendor import personaplex

    root = Path(personaplex.__file__).parent
    manifest = json.loads((root / "provenance.json").read_text())
    assert len(manifest["modifications"]) == 1
    for name, record in manifest["files"].items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert digest == record["vendored_sha256"]
        if name != "modules/transformer.py":
            assert digest == record["upstream_sha256"]


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_loss_all_gradient_parity_with_native_masks(use_checkpoint):
    model = tiny_personaplex().train()
    reference = copy.deepcopy(model)
    if use_checkpoint:
        model.gradient_checkpointing_enable()
    value = codes()
    supervised = torch.ones_like(value, dtype=torch.bool)
    supervised[:, :, :2] = False  # Prompt context contributes no supervised labels.
    loss = duplex_loss(model, value, supervised, user_weight=0.3)
    with no_compile():
        native = reference.forward_train(value)
    expected = 0
    for logits, mask, selection, weight in (
        (native.text_logits, native.text_mask, slice(0, 1), 1),
        (native.logits[:, :8], native.mask[:, :8], slice(1, 9), 1),
        (native.logits[:, 8:], native.mask[:, 8:], slice(9, 17), 0.3),
    ):
        mask = mask & supervised[:, selection]
        expected = expected + F.cross_entropy(logits[mask], value[:, selection][mask]) * weight
    assert torch.isnan(native.logits[:, 1, -1]).all()
    torch.testing.assert_close(loss.loss, expected, atol=0, rtol=0)
    assert loss.counts == {"text_symbols": 4, "agent_symbols": 25, "user_symbols": 25}
    loss.loss.backward()
    expected.backward()
    for (name, a), (other, b) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other
        assert a.grad is not None and torch.isfinite(a.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0, msg=name)


def test_native_training_matches_streaming_teacher_forcing():
    model = tiny_personaplex().eval()
    value = codes()
    initial = model._get_initial_token()
    delayed = _delay_sequence(model.delays, value, initial)
    inputs = torch.cat([initial, delayed[:, :, :-1]], -1)
    with torch.no_grad(), no_compile():
        hidden, text = model.forward_codes(inputs)
        depth = model.forward_depformer_training(delayed, hidden)
        streamed_text, streamed_depth = [], []
        with model.streaming(1):
            for t in range(value.shape[-1]):
                state, text_step = model.forward_codes(inputs[:, :, t : t + 1])
                streamed_text.append(text_step)
                frame = []
                with model.depformer.streaming(1):
                    for cb in range(16):
                        frame.append(
                            model.forward_depformer(cb, delayed[:, cb : cb + 1, t : t + 1], state)
                        )
                streamed_depth.append(torch.cat(frame, 1))
    torch.testing.assert_close(text, torch.cat(streamed_text, 2), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(depth, torch.cat(streamed_depth, 2), atol=1e-6, rtol=1e-5)


def test_future_user_audio_does_not_leak_into_earlier_agent_outputs():
    model = tiny_personaplex().eval()
    value = codes()
    changed = value.clone()
    changed[:, 9:, 3:] = (changed[:, 9:, 3:] + 1) % model.card
    with no_compile():
        a, b = model.forward_train(value), model.forward_train(changed)
    # Semantic audio/text have zero delay; residual codebooks have one frame delay.
    torch.testing.assert_close(a.text_logits[:, :, :4], b.text_logits[:, :, :4], atol=0, rtol=0)
    torch.testing.assert_close(a.logits[:, :1, :4], b.logits[:, :1, :4], atol=0, rtol=0)
    torch.testing.assert_close(a.logits[:, 1:8, :3], b.logits[:, 1:8, :3], atol=0, rtol=0)
    assert not torch.equal(a.text_logits[:, :, 4:], b.text_logits[:, :, 4:])


def test_empty_masks_and_input_only_user_streams():
    model = tiny_personaplex()
    value = codes()
    none = torch.zeros_like(value, dtype=torch.bool)
    empty = duplex_loss(model, value, none)
    assert empty.loss.item() == 0
    empty.loss.backward()
    model.zero_grad(set_to_none=True)
    valid = torch.ones_like(value, dtype=torch.bool)
    loss = duplex_loss(model, value, valid)
    loss.loss.backward()
    assert model.emb[8].weight.grad.abs().sum() > 0  # User input still conditions the agent.
    assert model.linears[8].weight.grad.count_nonzero() == 0  # Input-only user stream.
    value[:, 1, 0] = model.card
    with pytest.raises(OmniFactoryError, match="E_DUPLEX_CODES"):
        duplex_loss(model, value, valid)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("voice_prefix", [False, True])
def test_agent_depth_prefix_preserves_logits_loss_and_gradients(checkpointing, voice_prefix):
    model = tiny_personaplex().train()
    reference = copy.deepcopy(model)
    if checkpointing:
        model.gradient_checkpointing_enable()
        reference.gradient_checkpointing_enable()
    voice = None
    if voice_prefix:
        voice = dict(
            embeddings=torch.randn(3, 1, 1, model.dim),
            cache=torch.zeros(1, 17, 4, dtype=torch.long),
        )
    value = codes()
    mask = torch.ones_like(value, dtype=torch.bool)
    mask[:, :, :2] = False
    with no_compile():
        full = reference.forward_train(value, voice_state=voice)
        prefix = model.forward_train(value, voice_state=voice, agent_only_depth=True)
    assert prefix.logits.shape[1] == 8 and model.dep_q == 16
    torch.testing.assert_close(prefix.text_logits, full.text_logits, atol=0, rtol=0)
    torch.testing.assert_close(
        prefix.logits, full.logits[:, :8], atol=1e-6, rtol=1e-5, equal_nan=True
    )
    a = duplex_loss(model, value, mask, voice_state=voice, agent_only_depth=True)
    b = duplex_loss(reference, value, mask, voice_state=voice)
    assert a.counts == b.counts
    torch.testing.assert_close(a.loss, b.loss, atol=1e-6, rtol=1e-5)
    a.loss.backward()
    b.loss.backward()
    for (name, p), (other, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other
        actual = p.grad if p.grad is not None else torch.zeros_like(p)
        expected = q.grad if q.grad is not None else torch.zeros_like(q)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5, msg=name)
    assert model.emb[8].weight.grad.abs().sum() > 0  # User input conditioning remains active.
    assert model.depformer_in[8].weight.grad is None
    assert model.linears[8].weight.grad is None


def test_agent_depth_never_executes_user_projections_and_rejects_user_loss():
    model = tiny_personaplex().train()

    def forbidden(*args):
        pytest.fail("input-only user projection executed")

    handles = [
        layer.register_forward_pre_hook(forbidden)
        for layers in (model.depformer_in, model.linears)
        for layer in layers[8:]
    ]
    value = codes()
    mask = torch.ones_like(value, dtype=torch.bool)
    loss = duplex_loss(model, value, mask, agent_only_depth=True)
    loss.loss.backward()
    for h in handles:
        h.remove()
    with pytest.raises(OmniFactoryError, match="E_DUPLEX_DEPTH"):
        duplex_loss(model, value, mask, agent_only_depth=True, user_weight=0.3)


def test_lora_on_executed_projections_and_native_merge():
    pytest.importorskip("peft")
    from peft import LoraConfig, inject_adapter_in_model

    from omnifactory.training.policy import merge_lora

    model = tiny_personaplex()
    names = [f"transformer.layers.{i}.self_attn.out_proj" for i in range(2)]
    names += [f"depformer_in.{i}" for i in range(8)] + [f"linears.{i}" for i in range(8)]
    model.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=4, target_modules=names), model)
    value = codes()
    loss = duplex_loss(model, value, torch.ones_like(value, dtype=torch.bool))
    loss.loss.backward()
    for name in names:
        layer = model.get_submodule(name)
        assert layer.lora_B["default"].weight.grad.abs().sum() > 0, name
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    optimizer.step()
    with torch.no_grad(), no_compile():
        before = model.forward_train(value)
        merged = merge_lora(model)
        plain = tiny_personaplex()
        plain.load_state_dict(merged.state_dict(), strict=True)
        after = plain.forward_train(value)
    torch.testing.assert_close(before.text_logits, after.text_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(before.logits, after.logits, atol=1e-6, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("user_weight", [0.0, 0.3])
def test_aligned_supervision_roundtrip_and_loss_measures(user_weight):
    value = codes()
    mask = torch.ones_like(value, dtype=torch.bool)
    mask[:, :, :2] = False
    plan, tensors = aligned_duplex_supervision(
        value, mask, text_card=64, audio_card=32, user_weight=user_weight, offset_us=240_000
    )
    restored = SupervisionPlan.model_validate_json(plan.model_dump_json())
    counts = normalization_counts(restored, tensors)
    loss = duplex_loss(tiny_personaplex(), value, mask, user_weight=user_weight)
    loss.validate_expected(expected_losses([(restored, counts)]))
    assert tensors["duplex/intervals"][0].tolist() == [240_000, 320_000]
    assert restored.streams[-1].direction == "input"
    assert restored.streams[-1].values.shape == (1, 8, 6)
    assert not restored.alignments  # Native delay/depth schedule owns attention.
    assert counts["agent_symbols"] == 25


def test_voice_prefix_matches_native_replay_and_teacher_forced_suffix():
    model = tiny_personaplex().eval()
    voice_codes = codes()
    generator = LMGen(model, "cpu", use_sampling=False, check=True)
    embeddings = []
    with torch.no_grad(), no_compile(), generator.streaming(1):
        for t in range(voice_codes.shape[-1]):
            out = generator.step(
                voice_codes[:, 9:, t : t + 1],
                voice_codes[:, 1:9, t : t + 1],
                voice_codes[:, 0, t],
                return_embeddings=True,
            )
            if out is not None and out[1] is not None:
                embeddings.append(out[1])
        voice = {
            "embeddings": torch.stack(embeddings),
            "cache": generator._streaming_state.cache.clone(),
        }
    # Replay official-shaped embeddings/cache, as native offline inference does.
    replay = LMGen(model, "cpu", use_sampling=False, check=True, return_logits=True)
    replay.voice_prompt_embeddings = voice["embeddings"]
    replay.voice_prompt_cache = voice["cache"]
    suffix = codes()
    text, audio = [], []
    with torch.no_grad(), no_compile(), replay.streaming(1):
        replay._step_voice_prompt(None)
        for t in range(suffix.shape[-1]):
            _, (a, b) = replay.step(
                suffix[:, 9:, t : t + 1], suffix[:, 1:9, t : t + 1], suffix[:, 0, t]
            )
            text.append(a)
            audio.append(b.unsqueeze(2))
        expected_audio, valid = _undelay_sequence(model.delays[1:], torch.cat(audio, 2))
    with no_compile():
        actual = model.forward_train(suffix, voice_state=voice)
    torch.testing.assert_close(actual.text_logits, torch.cat(text, 2), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.logits[valid], expected_audio[valid], atol=1e-6, rtol=1e-5)
    model.train().gradient_checkpointing_enable()
    loss = duplex_loss(model, suffix, torch.ones_like(suffix, dtype=torch.bool), voice_state=voice)
    loss.loss.backward()
    assert model.transformer.layers[0].self_attn.out_proj.weight.grad.abs().sum() > 0
