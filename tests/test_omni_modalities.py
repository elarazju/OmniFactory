"""Video, native TTS causality and optional real S3/decoder integration."""

import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from omnifactory.adapters.minicpm import MiniCPMAdapter, load_minicpm
from omnifactory.adapters.speech import file_sha256, generate_codes, speech_condition, speech_logits
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.datasets.omni_demo import create_omni_demo
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint, train
from omnifactory.training.evaluation import evaluate, infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import apply_policy
from omnifactory.training.tiny_minicpm import create_tiny_minicpm


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for actual video fixtures")
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("full-omni")
    create_tiny_minicpm(root / "model", speech=True)
    create_omni_demo(root / "data")
    # Structural TTS tests do not need an ONNX download; real integration below does.
    config = TrainConfig(
        model=str(root / "model"),
        adapter="minicpm_o_4_5",
        task="speech_sft",
        speech_tokenizer=root / "not-used.onnx",
        speech_tokenizer_sha256="0" * 64,
        dataset=root / "data/train.jsonl",
        output_dir=root / "run",
        max_steps=2,
        save_steps=1,
        max_length=512,
        components={"speech_generator": {"mode": "lora", "rank": 4, "alpha": 8, "lr": 0.01}},
        budget={
            "max_examples": 2,
            "max_llm_tokens": 1024,
            "max_vision_patches": 128,
            "max_audio_frames": 400,
            "max_codec_frames": 128,
            "max_decoded_media_bytes": 1000000,
        },
    )
    return root, config, list(iter_samples(config.dataset))


def text_target(sample):
    raw = sample.model_dump(mode="json")
    raw["predictions"][0]["target_block_ids"] = [raw["speech_pairs"][0]["text_block_id"]]
    return CanonicalSample.model_validate(raw)


def test_video_soundtrack_policy_and_frame_budget(runtime):
    _, config, samples = runtime
    cfg = config.model_copy(update={"task": "understanding_sft"})
    adapter = MiniCPMAdapter.from_config(cfg)
    sample = text_target(samples[3])
    encoded = adapter.encode(sample, "p0", contract_id="video")
    assert encoded.cost.modalities == ("audio", "text", "video")
    assert encoded.cost.vision_patches > 0 and encoded.cost.audio_frames > 0
    assert encoded.metadata["video"][0]["frames"] == [{"start_us": 0, "end_us": 1000000}]
    model = load_minicpm(cfg)
    adapter.forward(model, encoded.payload).loss.backward()
    assert model.vpm.embeddings.patch_embedding.weight.grad is not None
    raw = sample.model_dump(mode="json")
    raw["messages"][0]["content"][0]["audio_mode"] = "exclude"
    excluded = adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="video")
    assert excluded.cost.audio_frames == 0
    assert excluded.cost.vision_patches == encoded.cost.vision_patches
    adapter.video_options["fps"] = 2
    adapter.video_options["max_frames"] = 1
    with pytest.raises(OmniFactoryError, match="E_DECODE_BUDGET"):
        adapter.encode(sample, "p0", contract_id="video")


def test_video_partial_slice(runtime):
    _, config, samples = runtime
    adapter = MiniCPMAdapter.from_config(config.model_copy(update={"task": "understanding_sft"}))
    raw = text_target(samples[3]).model_dump(mode="json")
    raw["messages"][0]["content"][0]["media"].update(start_us=250000, end_us=750000)
    e = adapter.encode(CanonicalSample.model_validate(raw), "p0", contract_id="slice")
    assert e.metadata["video"][0]["frames"] == [{"start_us": 250000, "end_us": 750000}]
    assert e.payload["audio_feature_lens"][0].tolist() == [50]


def test_tts_teacher_forcing_causal_and_incremental_parity(runtime):
    _, config, _ = runtime
    model = load_minicpm(config).eval()
    tts = model.tts
    ids = torch.tensor([30, 31, 32])
    hidden = torch.randn(3, 128, requires_grad=True)
    condition = speech_condition(tts, ids, hidden)
    targets = torch.tensor([10, 20, 30, 6561])
    logits = speech_logits(tts, condition, targets)
    changed = speech_logits(tts, condition, torch.tensor([10, 20, 99, 6561]))
    # Changing code[2] cannot affect predictions for code[0], code[1], code[2].
    torch.testing.assert_close(logits[:3], changed[:3], atol=1e-6, rtol=1e-6)
    prefix = tts.model(inputs_embeds=condition.unsqueeze(0), use_cache=False)
    torch.testing.assert_close(
        logits[0], tts.head_code[0](prefix.last_hidden_state[0, -1]), atol=1e-6, rtol=1e-6
    )
    F.cross_entropy(logits, targets).backward()
    assert hidden.grad.abs().sum() > 0
    assert tts.emb_code[0].weight.grad[10].abs().sum() > 0
    assert tts.head_code[0].parametrizations.weight.original1.grad.abs().sum() > 0
    codes, finished = generate_codes(tts, condition.detach(), 4)
    assert all(0 <= x < 6561 for x in codes)
    assert len(codes) == 4 or finished
    # Independent native generation checks our sampling/cache stepping. The
    # upstream implementation drops its last code when truncated, so compare
    # the complete prefix it exposes, not its off-by-one output length.
    tts.eval()
    torch.manual_seed(42)
    ours, _ = generate_codes(tts, condition.detach(), 5)
    torch.manual_seed(42)

    def current_cache_position(module, args, kwargs):
        # The pinned upstream generator passes an old [1,T] cache_position;
        # Transformers 4.57 requires [T]. Keep this compatibility shim local
        # to the independent reference call.
        kwargs["cache_position"] = kwargs["cache_position"].reshape(-1)
        return args, kwargs

    handle = tts.model.register_forward_pre_hook(current_cache_position, with_kwargs=True)
    try:
        native = tts.generate(
            condition.detach().unsqueeze(0),
            eos_token=torch.tensor([tts.num_audio_tokens - 1]),
            max_new_token=5,
            show_tqdm=False,
        )
    finally:
        handle.remove()
    assert native.new_ids[0, :, 0].tolist() == ours[: len(native.new_ids[0])]


def test_speech_inference_never_opens_target_audio(runtime):
    _, config, samples = runtime
    adapter = MiniCPMAdapter.from_config(config)
    a = adapter.encode(samples[0], "p0", contract_id="infer", inference=True)
    raw = samples[0].model_dump(mode="json")
    raw["assets"]["target-wave"]["uri"] = "does-not-exist.wav"
    raw["messages"][-1]["content"][0]["text"] = "a different target"
    b = adapter.encode(
        CanonicalSample.model_validate(raw), "p0", contract_id="infer", inference=True
    )
    assert "codec_labels" not in a.payload and adapter.codec is None
    for key in a.tensors:
        torch.testing.assert_close(a.tensors[key], b.tensors[key])


@pytest.fixture(scope="module")
def real_codec_config(runtime):
    root, config, _ = runtime
    path = os.environ.get("OMNIFACTORY_SPEECH_ASSETS")
    if not path:
        pytest.skip("set OMNIFACTORY_SPEECH_ASSETS for native 1.23 GB asset integration")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("s3tokenizer")
    assets = Path(path).resolve()
    return config.model_copy(
        update={
            "speech_tokenizer": assets / "speech_tokenizer_v2_25hz.onnx",
            "speech_tokenizer_sha256": file_sha256(assets / "speech_tokenizer_v2_25hz.onnx"),
            "speech_assets": assets,
        }
    )


def test_native_s3_batch_loss_and_gradient(runtime, real_codec_config):
    _, _, samples = runtime
    config = real_codec_config
    adapter = MiniCPMAdapter.from_config(config)
    model, _, plan = apply_policy(load_minicpm(config), config)
    model.eval()
    varied = list(samples)
    raw = samples[1].model_dump(mode="json")
    raw["messages"][-1]["content"][-1]["media"]["end_us"] = 150000
    varied[1] = CanonicalSample.model_validate(raw)
    rows = [adapter.encode(s, "p0", contract_id="real") for s in varied]
    assert rows[0].payload["codec_labels"][0].tolist() == [
        3515,
        4299,
        4218,
        4218,
        4218,
        4218,
        4161,
        6561,
    ]
    assert "speech_generator" in plan.gradient_components
    counts = [r.target_tokens for r in rows]
    assert len(set(counts)) > 1
    singles = [adapter.forward(model, r.payload).loss for r in rows]
    batch = adapter.collate([r.payload for r in rows])
    together = adapter.forward(model, batch)
    torch.testing.assert_close(
        together.loss,
        sum(n * x for n, x in zip(counts, singles)) / sum(counts),
        atol=2e-5,
        rtol=2e-5,
    )
    together.loss.backward()
    assert model.tts.model.layers[0].self_attn.q_proj.lora_B["default"].weight.grad.abs().sum() > 0
    assert model.llm.model.layers[0].self_attn.q_proj.weight.grad is None


def test_speech_resume_merge_and_waveform_inference(runtime, real_codec_config, tmp_path):
    root, _, samples = runtime
    config = real_codec_config
    result = train(config)
    checkpoint = Path(result["checkpoint"])
    resumed = train(
        config.model_copy(
            update={
                "output_dir": tmp_path / "resumed",
                "resume_from": root / "run/checkpoint-00000001",
            }
        )
    )
    saved = load_file(checkpoint / "trainable.safetensors")
    restored = load_file(Path(resumed["checkpoint"]) / "trainable.safetensors")
    for key in saved:
        torch.testing.assert_close(saved[key], restored[key], atol=0, rtol=0)
    metrics = evaluate(checkpoint, root / "data/validation.jsonl")
    assert metrics["codec_tokens"] == 32 and metrics["codec_loss"] > 0
    assert metrics["perplexity"] is None
    export_checkpoint(checkpoint, tmp_path / "export")
    before, adapter, _, _ = load_checkpoint(checkpoint)
    after = load_minicpm(config.model_copy(update={"model": str(tmp_path / "export")})).eval()
    e = adapter.encode(samples[3], "p0", contract_id="real")
    with torch.no_grad():
        torch.testing.assert_close(
            adapter.forward(before, e.payload).loss,
            adapter.forward(after, e.payload).loss,
            atol=2e-5,
            rtol=2e-5,
        )
    # Restrict synthesis to one video example to keep native flow/HiFT integration small.
    validation = list(iter_samples(root / "data/validation.jsonl"))[-1]
    data = tmp_path / "one.jsonl"
    data.write_text(validation.model_dump_json() + "\n")
    report = infer_export(
        tmp_path / "export",
        data,
        root=root / "data",
        output=tmp_path / "speech.jsonl",
        max_new_tokens=4,
        generate_audio=True,
        max_codec_tokens=8,
        reference_audio=root / "data/seed/media/validation-002.wav",
    )
    assert report["predictions"] == 1
    row = json.loads((tmp_path / "speech.jsonl").read_text())
    import soundfile as sf

    audio, rate = sf.read(row["audio"])
    assert rate == 24000 and len(audio) > 0
    assert all(0 <= code < 6561 for code in row["codec_tokens"])
    assert "speech_finished" in row


def test_codec_hash_rejected_before_session_load(tmp_path):
    from omnifactory.adapters.speech import S3Tokenizer

    path = tmp_path / "codec.onnx"
    path.write_bytes(b"wrong model")
    with pytest.raises(OmniFactoryError, match="E_CODEC_HASH"):
        S3Tokenizer(path, "0" * 64)


def test_native_export_replaces_trained_tts(runtime, tmp_path):
    from omnifactory.training.export import assemble_native

    _, config, _ = runtime
    model = load_minicpm(config)
    original = model.tts.model.layers[0].self_attn.q_proj.weight.detach().clone()
    with torch.no_grad():
        model.tts.model.layers[0].self_attn.q_proj.weight.add_(0.125)
    output = tmp_path / "native"
    output.mkdir()
    assemble_native(Path(config.model), model, output, adapter="minicpm_o_4_5")
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    key = "tts.model.layers.0.self_attn.q_proj.weight"
    weights = load_file(output / index[key])
    torch.testing.assert_close(weights[key], original + 0.125)


def test_joint_speech_loss_reaches_trainable_vision_projector(runtime, real_codec_config):
    _, _, samples = runtime
    config = TrainConfig.model_validate(
        {
            **real_codec_config.model_dump(),
            "text_loss_weight": 0,
            "components": {"vision_projector": {"mode": "full", "lr": 0.001}},
        }
    )
    model, _, plan = apply_policy(load_minicpm(config), config)
    adapter = MiniCPMAdapter.from_config(config)
    row = adapter.encode(samples[1], "p0", contract_id="speech-gradient")
    assert row.target_tokens == len(row.payload["codec_labels"][0])
    output = adapter.forward(model, row.payload)
    output.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.resampler.parameters())
    assert all(p.grad is None for p in model.llm.parameters())
    assert all(p.grad is None for p in model.tts.parameters())
    assert "speech_generator" in plan.gradient_components
