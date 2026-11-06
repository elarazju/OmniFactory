"""Native Qwen video/Talker acceptance; no downloads and no substitute codec."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
from safetensors.torch import load_file

from omnifactory.adapters.qwen2_5 import Qwen25Adapter, load_thinker
from omnifactory.adapters.qwen_speech import (
    CODEC_EOS,
    load_qwen_speech,
    talker_condition,
    talker_logits,
)
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.omni_demo import create_omni_demo
from omnifactory.datasets.qwen_speech import synthesize_dataset
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import evaluate, infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import apply_policy
from omnifactory.training.tiny import create_tiny


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required for native video integration")
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("qwen-omni")
    create_tiny(root / "model", speech=True)
    create_omni_demo(root / "seed")
    config = TrainConfig(
        model=str(root / "model"),
        task="speech_sft",
        speech_codes=root / "data/speech_codes.jsonl",
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        video_fps=2,
        max_steps=4,
        save_steps=2,
        gradient_accumulation_steps=2,
        max_length=512,
        components={
            "language_model": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.01},
            "speech_generator": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.01},
        },
        budget={
            "max_examples": 2,
            "max_llm_tokens": 1024,
            "max_vision_patches": 128,
            "max_audio_frames": 400,
            "max_codec_frames": 128,
            "max_decoded_media_bytes": 1000000,
        },
    )
    synthesize_dataset(
        config,
        root / "data",
        datasets=[root / "seed/train.jsonl", root / "seed/validation.jsonl"],
        max_codec_tokens=8,
        allow_truncated=True,
    )
    return root, config, list(iter_samples(config.dataset))


def text_only(sample):
    raw = sample.model_dump(mode="json")
    raw["predictions"][0]["target_block_ids"] = [raw["speech_pairs"][0]["text_block_id"]]
    return CanonicalSample.model_validate(raw)


def test_native_video_layout_timing_and_audio_policy(runtime):
    _, cfg, samples = runtime
    cfg = cfg.model_copy(update={"task": "understanding_sft"})
    adapter = Qwen25Adapter.from_config(cfg)
    row = adapter.encode(text_only(samples[-1]), "p0", contract_id="video")
    assert "pixel_values_videos" in row.payload and "pixel_values" not in row.payload
    assert row.payload["use_audio_in_video"] is True
    assert row.payload["video_grid_thw"].tolist() == [[1, 4, 4]]
    assert row.payload["video_second_per_grid"].tolist() == [1.0]
    assert row.cost.vision_patches == 16
    assert row.metadata["video"][0]["frames"] == [
        {"start_us": 0, "end_us": 500000},
        {"start_us": 500000, "end_us": 1000000},
    ]
    raw = text_only(samples[-1]).model_dump(mode="json")
    raw["messages"][0]["content"][0]["audio_mode"] = "exclude"
    silent = adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="video")
    assert "input_features" not in silent.payload and not silent.payload["use_audio_in_video"]
    model = load_thinker(cfg).eval()
    mixed = adapter.collate([row.payload, silent.payload])
    expected = (
        adapter.forward(model, row.payload).loss * row.target_tokens
        + adapter.forward(model, silent.payload).loss * silent.target_tokens
    ) / (row.target_tokens + silent.target_tokens)
    torch.testing.assert_close(adapter.forward(model, mixed).loss, expected)
    adapter.forward(model, row.payload).loss.backward()
    assert model.visual.merger.mlp[2].weight.grad.abs().sum() > 0
    adapter.video_options["max_frames"] = 1
    with pytest.raises(OmniFactoryError, match="E_DECODE_BUDGET"):
        adapter.encode(text_only(samples[-1]), "p0", contract_id="video")


def test_codec_causality_and_cached_native_logits(runtime):
    _, cfg, samples = runtime
    adapter, model = Qwen25Adapter.from_config(cfg), load_qwen_speech(cfg).eval()
    batch = adapter.encode(samples[-1], "p0", contract_id="speech").payload
    inputs = {k: v for k, v in batch.items() if k not in ("codec_labels", "speech_spans")}
    output = model.thinker(**inputs, output_hidden_states=True, use_cache=False)
    condition = talker_condition(model, batch, output, *map(int, batch["speech_spans"][0]))
    codes = torch.tensor([11, 22, 33, CODEC_EOS])
    logits = talker_logits(model, condition, codes)
    changed = talker_logits(model, condition, torch.tensor([11, 22, 99, CODEC_EOS]))
    torch.testing.assert_close(logits[:, :3], changed[:, :3], rtol=0, atol=0)
    prefix, remaining, positions = condition
    # Compare manual teacher-forcing prefix against the untouched native prefill.
    raw_prefix = prefix.clone()
    raw_prefix[:, -2] -= model.talker.get_input_embeddings()(
        torch.tensor([model.talker.codec_pad_token])
    )
    raw_prefix[:, -1] -= model.talker.get_input_embeddings()(
        torch.tensor([model.talker.codec_bos_token])
    )
    start = int(batch["speech_spans"][0][0])
    text_ids = torch.cat(
        [
            batch["input_ids"][:, :start],
            torch.tensor([[model.speaker_bos]]),
            batch["input_ids"][:, start : start + 1],
        ],
        dim=1,
    )
    native = model.talker(
        inputs_embeds=raw_prefix,
        input_text_ids=text_ids,
        attention_mask=torch.ones_like(text_ids),
        video_grid_thw=batch.get("video_grid_thw"),
        use_audio_in_video=batch.get("use_audio_in_video", False),
        audio_feature_lengths=batch["feature_attention_mask"].sum(-1),
        video_second_per_grid=batch.get("video_second_per_grid"),
        use_cache=False,
        return_dict=True,
    )
    torch.testing.assert_close(native.logits[:, -1], logits[:, 0], atol=1e-6, rtol=1e-5)
    cached = model.talker(
        inputs_embeds=prefix, position_ids=positions, use_cache=True, return_dict=True
    )
    torch.testing.assert_close(cached.logits[:, -1], logits[:, 0], atol=1e-6, rtol=1e-5)
    for i in range(len(codes) - 1):
        embedding = (
            model.talker.get_input_embeddings()(codes[i : i + 1].view(1, 1))
            + remaining[:, i : i + 1]
        )
        cached = model.talker(
            inputs_embeds=embedding,
            position_ids=(positions.max() + i + 1).expand(3, 1, 1),
            past_key_values=cached.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        torch.testing.assert_close(cached.logits[:, -1], logits[:, i + 1], atol=1e-6, rtol=1e-5)
    torch.nn.functional.cross_entropy(logits[0], codes).backward()
    assert model.talker.codec_head.weight.grad.abs().sum() > 0


def test_joint_batch_gradient_and_decoder_not_loaded(runtime):
    _, cfg, samples = runtime
    adapter = Qwen25Adapter.from_config(cfg)
    model, _, _ = apply_policy(load_qwen_speech(cfg), cfg)
    assert not hasattr(model, "token2wav")
    model.eval()
    examples = [adapter.encode(s, "p0", contract_id="joint") for s in samples]
    # Vary codec lengths to exercise normalization rather than average per-example loss.
    rows = [e.payload for e in examples]
    rows[0]["codec_labels"] = [rows[0]["codec_labels"][0][:3]]
    counts = [
        int((row["labels"][:, 1:] != -100).sum()) + sum(len(c) for c in row["codec_labels"])
        for row in rows
    ]
    for row, n in zip(rows, counts):
        (adapter.forward(model, row).loss * n / sum(counts)).backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    output = adapter.forward(model, adapter.collate(rows))
    output.loss.backward()
    for name, p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], atol=2e-6, rtol=5e-4)
    assert any("talker" in n and "lora_B" in n and g.abs().sum() > 0 for n, g in expected.items())
    assert any("thinker" in n and "lora_B" in n and g.abs().sum() > 0 for n, g in expected.items())


def test_speech_only_gradient_through_frozen_talker_to_video(runtime):
    _, cfg, samples = runtime
    cfg = TrainConfig.model_validate(
        {
            **cfg.model_dump(),
            "text_loss_weight": 0,
            "components": {"vision_projector": {"mode": "full", "lr": 0.001}},
        }
    )
    adapter = Qwen25Adapter.from_config(cfg)
    model, _, plan = apply_policy(load_qwen_speech(cfg), cfg)
    encoded = adapter.encode(samples[-1], "p0", contract_id="speech")
    assert encoded.target_tokens == len(encoded.payload["codec_labels"][0])
    adapter.forward(model, encoded.payload).loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if n.startswith("thinker.visual.merger")
    )
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith("talker."))
    assert "speech_generator" in plan.gradient_components


@pytest.mark.parametrize(
    "field,value",
    [
        ("codec", "minicpm_s3"),
        ("speaker", "Ethan"),
        ("transcript_sha256", "0" * 64),
        ("audio_sha256", "0" * 64),
        ("codes", [8192]),
        ("codes", [True]),
        ("sample_sha256", "0" * 64),
    ],
)
def test_code_manifest_rejects_incompatible_or_stale_labels(runtime, tmp_path, field, value):
    _, cfg, samples = runtime
    rows = [json.loads(line) for line in cfg.speech_codes.read_text().splitlines()]
    rows[0][field] = value
    path = tmp_path / "codes.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    adapter = Qwen25Adapter.from_config(cfg.model_copy(update={"speech_codes": path}))
    with pytest.raises(OmniFactoryError, match="E_CODEC"):
        adapter.encode(samples[0], "p0", contract_id="bad")


def test_inference_has_no_target_or_manifest_dependency(runtime):
    _, cfg, samples = runtime
    adapter = Qwen25Adapter.from_config(
        cfg.model_copy(update={"speech_codes": Path("/missing/native-codes.jsonl")})
    )
    before = adapter.encode(samples[-1], "p0", contract_id="prompt", inference=True)
    raw = samples[-1].model_dump(mode="json")
    raw["assets"]["target-wave"]["uri"] = "/missing/target.wav"
    raw["messages"][-1]["content"][0]["text"] = "A completely different <|im_end|> target."
    after = adapter.encode(
        CanonicalSample.model_validate(raw), "p0", contract_id="prompt", inference=True
    )
    for key in before.tensors:
        torch.testing.assert_close(before.tensors[key], after.tensors[key])


def test_truncated_teacher_labels_do_not_supervise_eos(runtime):
    _, cfg, samples = runtime
    adapter = Qwen25Adapter.from_config(cfg)
    encoded = adapter.encode(samples[0], "p0", contract_id="codes")
    row = json.loads(cfg.speech_codes.read_text().splitlines()[0])
    assert row["origin"] == "native_teacher" and not row["speech_finished"]
    assert encoded.payload["codec_labels"][0].tolist() == row["codes"]


def test_resume_export_decoder_preservation_and_wav(runtime, tmp_path):
    root, cfg, samples = runtime
    result = train(cfg)
    checkpoint = Path(result["checkpoint"])
    resumed = train(
        cfg.model_copy(
            update={
                "output_dir": tmp_path / "resumed",
                "resume_from": root / "run/checkpoint-00000002",
            }
        )
    )
    for name, tensor in load_file(checkpoint / "trainable.safetensors").items():
        torch.testing.assert_close(
            tensor,
            load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")[name],
            rtol=0,
            atol=0,
        )
    metrics = evaluate(checkpoint, root / "data/validation.jsonl")
    assert (
        metrics["codec_tokens"] == 32
        and metrics["codec_loss"] > 0
        and metrics["perplexity"] is None
    )
    exported = tmp_path / "export"
    report = export_checkpoint(checkpoint, exported)
    assert report["format"] == "merged" and (exported / "spk_dict.pt").exists()
    before, adapter, _, _ = load_checkpoint(checkpoint)
    after = load_qwen_speech(cfg.model_copy(update={"model": str(exported)})).eval()
    encoded = adapter.encode(samples[-1], "p0", contract_id="merge")
    with torch.no_grad():
        torch.testing.assert_close(
            adapter.forward(before, encoded.payload).loss,
            adapter.forward(after, encoded.payload).loss,
            rtol=2e-5,
            atol=2e-5,
        )
    base = load_file(root / "model/model.safetensors")
    mapping = json.loads((exported / "model.safetensors.index.json").read_text())["weight_map"]
    shard = {file: load_file(exported / file) for file in set(mapping.values())}
    for name in base:
        if name.startswith("token2wav."):
            torch.testing.assert_close(base[name], shard[mapping[name]][name], rtol=0, atol=0)
    assert not torch.equal(
        base["talker.model.layers.0.self_attn.q_proj.weight"],
        shard[mapping["talker.model.layers.0.self_attn.q_proj.weight"]][
            "talker.model.layers.0.self_attn.q_proj.weight"
        ],
    )
    infer_export(
        exported,
        root / "data/validation.jsonl",
        output=tmp_path / "speech.jsonl",
        max_new_tokens=8,
        generate_audio=True,
        max_codec_tokens=8,
    )
    import soundfile as sf

    rows = [json.loads(line) for line in (tmp_path / "speech.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        wave = sf.info(row["audio"])
        assert wave.frames > 0 and wave.samplerate == 24000
        assert all(0 <= c < 8192 for c in row["codec_tokens"])
    with pytest.raises(OmniFactoryError, match="E_EXPORT_FORMAT"):
        export_checkpoint(checkpoint, tmp_path / "bad-export", format="thinker")


@pytest.mark.parametrize("stop", [8292, 8294])
def test_completed_labels_preserve_native_stop_identity(runtime, tmp_path, stop):
    _, cfg, samples = runtime
    rows = [json.loads(line) for line in cfg.speech_codes.read_text().splitlines()]
    rows[0].update(speech_finished=True, eos_token_id=stop)
    path = tmp_path / "complete.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    adapter = Qwen25Adapter.from_config(cfg.model_copy(update={"speech_codes": path}))
    encoded = adapter.encode(samples[0], "p0", contract_id="complete")
    assert encoded.payload["codec_labels"][0].tolist() == rows[0]["codes"] + [stop]
