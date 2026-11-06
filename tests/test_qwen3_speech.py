"""Native Qwen3 teacher-forcing invariants; tiny random weights are not quality evidence."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
from safetensors.torch import load_file

from omnifactory.adapters.qwen3 import Qwen3Adapter
from omnifactory.adapters.qwen3_speech import (
    CODEC,
    CODEC_EOS,
    load_qwen3_speech,
    make_decoder,
    talker_condition,
    talker_logits,
)
from omnifactory.adapters.qwen_speech import speech_target, text_hash
from omnifactory.adapters.speech import file_sha256
from omnifactory.adapters.supervision import normalization_counts
from omnifactory.data.io import iter_samples
from omnifactory.datasets.omni_demo import create_omni_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import digest
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import apply_policy, merge_lora
from omnifactory.training.tiny_qwen3 import create_tiny_qwen3


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("qwen3-speech")
    create_tiny_qwen3(root / "model", speech=True)
    create_omni_demo(root / "data")
    samples = list(iter_samples(root / "data/train.jsonl"))
    cfg = TrainConfig(
        adapter="qwen3_omni",
        model=str(root / "model"),
        task="speech_sft",
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        speech_speaker="Ethan",
        speech_codes=root / "codes.jsonl",
        max_steps=2,
        save_steps=1,
        max_length=1024,
        max_audio_seconds=1,
        max_codec_tokens=10,
        gradient_accumulation_steps=2,
        components={
            k: dict(mode="lora", rank=2, alpha=4, lr=0.001)
            for k in ("language_model", "speech_generator", "code_predictor")
        },
        budget=dict(
            max_examples=2,
            max_llm_tokens=2048,
            max_vision_patches=4096,
            max_audio_frames=1000,
            max_codec_frames=32,
            max_decoded_media_bytes=67108864,
        ),
    )
    adapter = Qwen3Adapter.from_config(cfg)
    # Synthetic unit-test records exercise manifest and mask handling only.
    # Formal GPU acceptance must obtain codes AND audio from native teacher synthesis.
    rows = []
    for i, sample in enumerate(samples):
        text, block = speech_target(sample, "p0")
        rows.append(
            dict(
                codec=CODEC,
                sample_id=sample.id,
                prediction_id="p0",
                audio_block_id=block.id,
                audio_sha256=file_sha256(
                    adapter.resolver.resolve(sample.assets[block.media.asset_id]).path
                ),
                transcript_sha256=text_hash(text),
                sample_sha256=digest(sample.model_dump(mode="json")),
                speaker="Ethan",
                start_us=block.media.start_us,
                end_us=block.media.end_us,
                origin="native_teacher",
                source_fingerprint="synthetic-unit-test-only",
                speech_finished=True,
                eos_token_id=CODEC_EOS,
                codes=torch.arange((3 + i) * 16).reshape(-1, 16).tolist(),
            )
        )
    cfg.speech_codes.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root, cfg, adapter, samples


def encoded(runtime, index=0):
    _, _, adapter, samples = runtime
    return adapter.encode(samples[index], "p0", contract_id="test")


def test_modalities_and_exact_supervision_counts(runtime):
    _, cfg, adapter, samples = runtime
    model = load_qwen3_speech(cfg).eval()
    for i, sample in enumerate(samples):
        row = encoded(runtime, i)
        codes = row.payload["codec_labels"][0]
        assert codes.shape == (4 + i, 16)
        assert codes[-1, 0] == CODEC_EOS and (codes[-1, 1:] == -100).all()
        assert row.cost.codec_frames == len(codes)
        counts = normalization_counts(row.supervision, row.tensors)
        expected = int((row.payload["labels"][:, 1:] != -100).sum()) + (3 + i) * 16 + 1
        assert counts == {"symbols": expected}
        with torch.no_grad():
            output = adapter.forward(model, row.payload)
        assert output.losses.counts == counts and torch.isfinite(output.loss)
        inference = adapter.encode(sample, "p0", contract_id="test", inference=True)
        assert "codec_labels" not in inference.payload and not inference.supervision.targets


def test_teacher_forcing_matches_native_cached_steps(runtime):
    _, cfg, _, _ = runtime
    model = load_qwen3_speech(cfg).eval()
    batch = encoded(runtime).payload
    labels = batch["codec_labels"][0]
    with torch.no_grad():
        out = model.thinker(
            **{k: v for k, v in batch.items() if k not in ("codec_labels", "speech_spans")},
            output_hidden_states=True,
        )
        condition = talker_condition(model, batch, out, *map(int, batch["speech_spans"][0]))
        primary, residual = talker_logits(model, condition, labels)
        prefix, trailing, pad, pos, _, _ = condition
        native = model.talker(
            inputs_embeds=prefix,
            position_ids=pos,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        for t in range(len(labels)):
            torch.testing.assert_close(primary[t], native.logits[0, -1], atol=2e-6, rtol=2e-5)
            if t == len(labels) - 1:
                break
            frame = labels[t]
            predictor = model.talker.code_predictor
            # Independent slow path: native wrapper selects the head by prefix length.
            inputs = [
                native.hidden_states[0][-1][:, -1:],
                model.talker.get_input_embeddings()(frame[0].reshape(1, 1)),
            ]
            for k in range(15):
                step = predictor(inputs_embeds=torch.cat(inputs, 1), use_cache=False)
                torch.testing.assert_close(residual[t, k], step.logits[0, -1], atol=2e-6, rtol=2e-5)
                inputs.append(predictor.get_input_embeddings()[k](frame[k + 1].reshape(1, 1)))
            previous = sum(inputs[1:])
            text = trailing[:, t : t + 1] if t < trailing.shape[1] else pad
            native = model.talker(
                inputs_embeds=previous + text,
                position_ids=(pos.max() + t + 1).expand(3, 1, 1),
                past_key_values=native.past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                generation_step=t,
            )


def test_frame_and_codebook_causality(runtime):
    _, cfg, _, _ = runtime
    model = load_qwen3_speech(cfg).eval()
    batch = encoded(runtime).payload
    with torch.no_grad():
        output = model.thinker(
            **{k: v for k, v in batch.items() if k not in ("codec_labels", "speech_spans")},
            output_hidden_states=True,
        )
        condition = talker_condition(model, batch, output, *map(int, batch["speech_spans"][0]))
        labels = batch["codec_labels"][0]
        p, r = talker_logits(model, condition, labels)
        altered = labels.clone()
        altered[1] += 17
        p2, r2 = talker_logits(model, condition, altered)
        torch.testing.assert_close(p[:2], p2[:2], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(r[:1], r2[:1], atol=1e-7, rtol=1e-6)
        assert not torch.equal(p[2:], p2[2:])
        altered = labels.clone()
        altered[0, 5] += 23
        p3, r3 = talker_logits(model, condition, altered)
        torch.testing.assert_close(p[0], p3[0], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(r[0, :5], r3[0, :5], atol=1e-7, rtol=1e-6)
        assert not torch.equal(r[0, 5:], r3[0, 5:])
    # MoE expert row grouping can change FP32 GEMM rounding under perturbation.
    # In addition to numeric comparisons, prove exact zero dependency gradients.
    captured = {}

    def capture(name):
        def hook(module, args, kwargs):
            captured[name] = kwargs["inputs_embeds"]

        return hook

    handles = [
        model.talker.model.register_forward_pre_hook(capture("frame"), with_kwargs=True),
        model.talker.code_predictor.model.register_forward_pre_hook(
            capture("book"), with_kwargs=True
        ),
    ]
    try:
        primary, residual = talker_logits(model, condition, labels)
        frame_grad, book_grad = torch.autograd.grad(
            primary[0, 5] + residual[0, 4, 9], (captured["frame"], captured["book"])
        )
        assert torch.count_nonzero(frame_grad[:, condition[0].shape[1] :]) == 0
        assert torch.count_nonzero(book_grad[0, 6:]) == 0
        assert torch.count_nonzero(book_grad[1:]) == 0
        assert torch.count_nonzero(book_grad[0, :6]) > 0
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    "component,index", [("speech_projection", 0), ("code_predictor", 0), ("audio_projector", 2)]
)
def test_frozen_stage_gradient_paths(runtime, component, index):
    _, cfg, adapter, _ = runtime
    config = TrainConfig.model_validate(
        {
            **cfg.model_dump(),
            "text_loss_weight": 0,
            "gradient_checkpointing": True,
            "components": {component: dict(mode="full", lr=0.001)},
        }
    )
    local = Qwen3Adapter.from_config(config)
    model, _, plan = apply_policy(load_qwen3_speech(config), config)
    batch = encoded(runtime, index).payload
    local.forward(model, batch).loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for n, p in model.named_parameters() if ".mlp.gate." in n)
    assert "speech_generator" in plan.gradient_components


def test_resume_native_export_and_decoder(runtime):
    root, cfg, adapter, _ = runtime
    trained = train(cfg)
    resumed = train(
        cfg.model_copy(
            update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
        )
    )
    a = load_file(trained["checkpoint"] + "/trainable.safetensors")
    b = load_file(resumed["checkpoint"] + "/trainable.safetensors")
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
    exported = export_checkpoint(Path(trained["checkpoint"]), root / "export")
    assert exported["format"] == "merged"
    native = load_qwen3_speech(cfg.model_copy(update={"model": str(root / "export")})).eval()
    original, _, _, _ = load_checkpoint(Path(trained["checkpoint"]), for_export=True)
    original = merge_lora(original).eval()
    batch = encoded(runtime).payload
    with torch.no_grad():
        torch.testing.assert_close(
            adapter.forward(original, batch).loss, adapter.forward(native, batch).loss
        )
    codes = batch["codec_labels"][0][:-1].tolist()
    wav, rate = make_decoder(cfg.model_copy(update={"model": str(root / "export")}))(codes)
    assert rate == 24000 and len(wav) > 0
    from omnifactory.training.evaluation import infer_export

    inference = infer_export(
        root / "export",
        root / "data/validation.jsonl",
        output=root / "generated.jsonl",
        generate_audio=True,
        max_new_tokens=2,
        max_codec_tokens=3,
        speaker="Ethan",
    )
    assert inference["audio_predictions"] == 4
    from transformers import Qwen3OmniMoeForConditionalGeneration

    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(root / "export")
    torch.testing.assert_close(full.talker.codec_head.weight, native.talker.codec_head.weight)


def test_wrong_codec_and_truncated_labels_rejected(runtime):
    _, cfg, adapter, samples = runtime
    from omnifactory.adapters.qwen3_speech import Qwen3CodeLabels

    labels = Qwen3CodeLabels(cfg.speech_codes)
    row = labels.rows[(samples[0].id, "p0")]
    for key, value in (
        ("codec", "qwen2_5_omni_v1"),
        ("speech_finished", False),
        ("eos_token_id", True),
        ("codes", [[0] * 15]),
    ):
        old = row[key]
        row[key] = value
        with pytest.raises(OmniFactoryError):
            labels.get(samples[0], "p0", adapter.resolver, "Ethan", 10)
        row[key] = old


def test_native_generation_collects_complete_frames(runtime, monkeypatch):
    from transformers import LogitsProcessor

    from omnifactory.adapters.qwen3_speech import generate_talker

    _, cfg, _, _ = runtime
    model = load_qwen3_speech(cfg).eval()
    batch = encoded(runtime).payload
    with torch.no_grad():
        output = model.thinker(
            **{k: v for k, v in batch.items() if k not in ("codec_labels", "speech_spans")},
            output_hidden_states=True,
        )
        condition = talker_condition(model, batch, output, *map(int, batch["speech_spans"][0]))

    # Force only the primary sequence to exercise EOS handling deterministically;
    # residual generation remains the actual native 15-head generator.
    class FixturePrimary(LogitsProcessor):
        def __call__(self, input_ids, scores):
            token = [17, 23, 9, CODEC_EOS][input_ids.shape[1]]
            scores.fill_(-torch.inf)
            scores[:, token] = 0
            return scores

    native_generate = model.talker.generate

    def generate(**kwargs):
        return native_generate(**kwargs, logits_processor=[FixturePrimary()])

    monkeypatch.setattr(model.talker, "generate", generate)
    residual_steps = []
    handle = model.talker.code_predictor.register_forward_hook(
        lambda module, args, output: residual_steps.append(output.generation_steps)
    )
    codes, eos = generate_talker(model, condition, 6)
    handle.remove()
    assert residual_steps == list(range(1, 16)) * 3

    assert eos == CODEC_EOS and [frame[0] for frame in codes] == [17, 23, 9]
    assert all(len(frame) == 16 and all(0 <= v < 2048 for v in frame) for frame in codes)
    codes, eos = generate_talker(model, condition, 2)
    assert eos is None and len(codes) == 1  # Final primary has no residual frame yet.
