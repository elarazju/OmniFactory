"""Real native MiMo architecture/codec tests, never mock gradients or tokenizers."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from omnifactory.adapters.mimo import MiMoAdapter
from omnifactory.adapters.mimo_model import load_mimo
from omnifactory.data.io import iter_samples
from omnifactory.datasets.mimo_demo import create_mimo_demo
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import evaluate, infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.tiny_mimo import create_tiny_mimo


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("mimo")
    fixture = create_tiny_mimo(root / "model")
    create_mimo_demo(root / "data")
    config = TrainConfig(
        adapter="mimo_audio",
        task="speech_sft",
        model=str(root / "model"),
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        speech_tokenizer=root / "model/assets/mimo_codec",
        speech_tokenizer_sha256=fixture["codec_fingerprint"],
        max_steps=2,
        save_steps=1,
        max_length=512,
        components={
            name: {"mode": "lora", "rank": 2, "alpha": 4, "lr": 0.01}
            for name in ("language_model", "speech_generator")
        },
        budget={
            "max_examples": 2,
            "max_llm_tokens": 1024,
            "max_vision_patches": 1,
            "max_audio_frames": 128,
            "max_codec_frames": 128,
            "max_decoded_media_bytes": 1000000,
        },
    )
    return root, config, MiMoAdapter.from_config(config), list(iter_samples(config.dataset))


def test_codec_multichannel_backward_and_padding(runtime):
    _, config, adapter, samples = runtime
    encoded = [adapter.encode(s, "p0", contract_id="test") for s in samples]
    model = load_mimo(config).eval()
    rows = [e.payload for e in encoded]
    for e in encoded:
        assert e.tensors["input_ids"].shape[1] == 9
        assert e.tensors["input_ids"].shape[-1] == e.cost.llm_tokens * 4
        assert (e.tensors["labels"][:, :, : e.prompt_length] == -100).all()
    singles = [adapter.forward(model, row) for row in rows]
    batch = adapter.forward(model, adapter.collate(rows))
    expected = sum(o.loss * e.target_tokens for o, e in zip(singles, encoded)) / sum(
        e.target_tokens for e in encoded
    )
    torch.testing.assert_close(batch.loss, expected)
    batch.loss.backward()
    for head in model.local_transformer_lm_heads:
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in adapter.codec.model.parameters())


def test_target_free_inference(runtime):
    _, _, adapter, samples = runtime
    raw = samples[0].model_dump(mode="json")
    original = adapter.encode(samples[0], "p0", contract_id="test", inference=True)
    raw["assets"]["target"]["uri"] = "missing-target-file.wav"
    raw["messages"][1]["content"][0]["text"] = "Changed answer."
    from omnifactory.data.schema import CanonicalSample

    changed = adapter.encode(
        CanonicalSample.model_validate(raw), "p0", contract_id="test", inference=True
    )
    assert "labels" not in changed.payload
    torch.testing.assert_close(original.payload["input_ids"], changed.payload["input_ids"])


def test_native_speech_prefix_shared_by_training_and_generation(runtime):
    _, _, adapter, samples = runtime
    trained = adapter.encode(samples[0], "p0", contract_id="test")
    prompt = adapter.encode(samples[0], "p0", contract_id="test", inference=True)
    ids = prompt.payload["input_ids"]
    torch.testing.assert_close(ids, trained.payload["input_ids"][:, :, : trained.prompt_length])
    text_ids = trained.payload["input_ids"][0, 0, :: adapter.g]
    sostm = adapter.processor.convert_tokens_to_ids("<|sostm|>")
    assert int(ids[0, 0, -adapter.g]) == sostm
    assert int((text_ids == sostm).sum()) == 1
    assert (trained.payload["labels"][:, :, : trained.prompt_length] == -100).all()
    assert (trained.payload["labels"][:, :, trained.prompt_length :] != -100).any()


def test_codec_loss_causality(runtime):
    _, config, adapter, samples = runtime
    model = load_mimo(config).eval()
    context = torch.randn(1, model.config.local_dim)
    codes = torch.zeros(1, 8, 4, dtype=torch.long)
    first = model.codec_logits(context, codes)
    changed = codes.clone()
    changed[:, 0, 3] = 1
    second = model.codec_logits(context, changed)
    # Current/future frame cannot affect its own prediction.
    torch.testing.assert_close(first[0], second[0])
    # It can affect later channels at their delayed positions.
    assert not torch.equal(first[-1][:, -1], second[-1][:, -1])


def test_train_resume_export_and_infer(runtime):
    root, config, adapter, samples = runtime
    result = train(config)
    checkpoint = Path(result["checkpoint"])
    evaluation = evaluate(checkpoint, root / "data/validation.jsonl")
    assert evaluation["codec_tokens"] > 0
    export_checkpoint(checkpoint, root / "export", format="vllm-omni")
    assert (root / "export/assets/mimo_codec/config.json").exists()
    assert json.loads((root / "export/config.json").read_text())["architectures"] == [
        "MiMoAudioModel"
    ]
    prediction = infer_export(
        root / "export",
        root / "data/validation.jsonl",
        output=root / "predictions.jsonl",
        max_new_tokens=3,
    )
    assert prediction["predictions"] == 3
    # Exported native parameters preserve trained forward behavior after LoRA merge.
    loaded = load_mimo(config.model_copy(update={"model": str(root / "export")})).eval()
    trained_model = load_checkpoint(checkpoint)[0]
    batch = adapter.encode(samples[0], "p0", contract_id="parity").payload
    torch.testing.assert_close(loaded(**batch).loss, trained_model(**batch).loss)
    # The export is self-contained: decoding must not reach back to the source codec.
    original = root / "model/assets/mimo_codec"
    moved = original.with_name("codec-away")
    original.rename(moved)
    try:
        standalone = MiMoAdapter.from_config(
            config.model_copy(update={"model": str(root / "export")})
        )
        waveform, rate = standalone.speech_decoder(loaded)([[0] * 8 for _ in range(8)])
        assert rate == 24000 and len(waveform) > 0
    finally:
        moved.rename(original)
    # Direct decoder uses real multi-codebook native RVQ and vocoder.
    waveform, rate = adapter.speech_decoder(None)([[0] * 8 for _ in range(8)])
    assert rate == 24000 and len(waveform) > 0
    resumed = config.model_copy(
        update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
    )
    resumed_result = train(resumed)
    left = load_checkpoint(checkpoint)[0]
    right = load_checkpoint(Path(resumed_result["checkpoint"]))[0]
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


def test_delayed_logits_match_native_cached_generation(runtime):
    _, config, _, _ = runtime
    model = load_mimo(config).eval()
    context = torch.randn(1, 1, model.config.local_dim)

    class Recorder:
        def __init__(self):
            self.logits = []

        def sample(self, scores, removed_tokens):
            self.logits.append(scores.clone())
            scores = scores.clone()
            scores[:, removed_tokens] = -torch.inf
            return scores.argmax(-1)

    recorder = Recorder()
    codes = model.local_forward(context, torch.long, context.device, local_sampler=recorder)
    parallel = model.codec_logits(context[:, 0], codes.transpose(1, 2))
    k = 0
    for t in range(model.group_size + max(model.delay_pattern)):
        for i, delay in enumerate(model.delay_pattern):
            if delay <= t < delay + model.group_size:
                torch.testing.assert_close(
                    recorder.logits[k], parallel[i][:, t - delay], atol=1e-6, rtol=1e-5
                )
                k += 1


def test_cached_global_logits_match_full_context(runtime):
    _, config, adapter, samples = runtime
    model = load_mimo(config).eval()
    batch = adapter.encode(samples[1], "p0", contract_id="test").payload
    ids = batch["input_ids"]
    with torch.no_grad():
        full = model(input_ids=ids, attention_mask=batch["attention_mask"])
        prefix = model(input_ids=ids[:, :, :-4], attention_mask=batch["attention_mask"][:, :-1])
        last = model(
            input_ids=ids[:, :, -4:],
            attention_mask=batch["attention_mask"],
            past_key_values=prefix.past_key_values,
        )
    torch.testing.assert_close(full.text_logits, last.text_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        full.local_hidden_states, last.local_hidden_states, atol=1e-6, rtol=1e-5
    )


def test_codec_hash_and_unsupported_inputs(runtime):
    _, config, _, samples = runtime
    from omnifactory.data.schema import CanonicalSample
    from omnifactory.errors import OmniFactoryError

    bad = MiMoAdapter.from_config(config.model_copy(update={"speech_tokenizer_sha256": "0" * 64}))
    with pytest.raises(OmniFactoryError, match="E_CODEC_HASH"):
        bad.encode(samples[0], "p0", contract_id="test")
    raw = samples[0].model_dump(mode="json")
    raw["assets"]["picture"] = {"modality": "image", "uri": "missing.png"}
    raw["messages"][0]["content"].append({"id": "picture", "type": "image", "asset_id": "picture"})
    with pytest.raises(OmniFactoryError, match="E_RUNTIME_MODALITY"):
        MiMoAdapter.from_config(config).encode(
            CanonicalSample.model_validate(raw), "p0", contract_id="test"
        )


def test_windowed_codec_attention_matches_dense_reference():
    from omnifactory._vendor.mimo_audio.codec.modeling_audio_tokenizer import flash_attn_varlen_func

    q, k, v = (torch.randn(7, 2, 4) for _ in range(3))
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    for causal in (True, False):
        result = flash_attn_varlen_func(q, k, v, cu, cu, 4, 4, causal=causal, window_size=[2, 1])
        expected = []
        for start, end in ((0, 3), (3, 7)):
            n = end - start
            delta = torch.arange(n)[:, None] - torch.arange(n)[None]
            allowed = (delta <= 2) & (delta >= -1)
            if causal:
                allowed &= delta >= 0
            scores = torch.einsum("ihd,jhd->hij", q[start:end], k[start:end]) / 2
            scores = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
            expected.append(torch.einsum("hij,jhd->ihd", scores, v[start:end]))
        torch.testing.assert_close(result, torch.cat(expected))


def test_understanding_audio_inputs_use_shared_training_interface(runtime):
    _, config, _, samples = runtime
    config = config.model_copy(update={"task": "understanding_sft"})
    adapter = MiMoAdapter.from_config(config)
    model = load_mimo(config).eval()
    rows = [adapter.encode(sample, "p0", contract_id="understanding") for sample in samples]
    result = adapter.forward(model, adapter.collate([row.payload for row in rows]))
    assert torch.isfinite(result.loss) and result.codec_tokens == 0
    assert rows[1].cost.audio_frames > 0 and rows[1].cost.codec_frames == 0
    result.loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.local_transformer_lm_heads[0].weight.grad is None


def test_speech_only_loss_counts_codebook_symbols(runtime):
    _, config, _, samples = runtime
    config = config.model_copy(update={"text_loss_weight": 0})
    adapter = MiMoAdapter.from_config(config)
    model = load_mimo(config).eval()
    row = adapter.encode(samples[0], "p0", contract_id="codec-only")
    assert row.target_tokens == row.cost.codec_frames * 8
    result = adapter.forward(model, row.payload)
    torch.testing.assert_close(result.loss, result.codec_loss)
