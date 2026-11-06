"""Real small native editing models; execution evidence, not pretrained quality."""

import copy
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("diffusers")
import torch
from diffusers import QwenImageTransformer2DModel
from PIL import Image
from safetensors.torch import load_file

from omnifactory.adapters.qwen_image_edit import QwenImageEditAdapter
from omnifactory.adapters.qwen_image_edit_model import load_qwen_image_edit
from omnifactory.data.schema import CanonicalSample
from omnifactory.errors import OmniFactoryError
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train
from omnifactory.training.evaluation import infer_export
from omnifactory.training.export import export_checkpoint
from omnifactory.training.policy import apply_policy
from omnifactory.training.tiny_qwen_image_edit import create_tiny_qwen_image_edit


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("qwen-image-edit")
    create_tiny_qwen_image_edit(root / "model")
    rows = []
    for i, color in enumerate(("blue", "green", "yellow")):
        Image.new("RGB", (32, 32), "red").save(root / f"source-{i}.png")
        Image.new("RGB", (32, 32), color).save(root / f"target-{i}.png")
        rows.append(
            CanonicalSample.model_validate(
                dict(
                    schema_version="0.1",
                    id=f"edit-{i}",
                    provenance=dict(
                        dataset="native-fixture",
                        source_id=str(i),
                        group_id=str(i),
                        split="validation" if i == 2 else "train",
                    ),
                    assets=dict(
                        source=dict(uri=f"source-{i}.png", modality="image"),
                        target=dict(uri=f"target-{i}.png", modality="image"),
                    ),
                    messages=[
                        dict(
                            id="user",
                            role="user",
                            content=[
                                dict(id="source", type="image", asset_id="source"),
                                dict(id="instruction", type="text", text=f"Change red to {color}."),
                            ],
                        ),
                        dict(
                            id="assistant",
                            role="assistant",
                            content=[
                                dict(id="target", type="image", asset_id="target"),
                            ],
                        ),
                    ],
                    predictions=[
                        dict(id="edit", target_message_id="assistant", target_block_ids=["target"])
                    ],
                )
            )
        )
    for split in ("train", "validation"):
        (root / f"{split}.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in rows if r.provenance.split == split)
        )
    config = TrainConfig(
        model=str(root / "model"),
        dataset=root / "train.jsonl",
        output_dir=root / "run",
        adapter="qwen_image_edit",
        task="image_sft",
        image_height=32,
        image_width=32,
        max_steps=2,
        save_steps=1,
        max_length=8192,
        max_attention_pairs=8192**2,
        components={
            "image_generator": dict(
                mode="lora",
                rank=2,
                alpha=4,
                lr=1e-3,
                target_modules=["to_q", "to_v", "add_q_proj", "add_v_proj"],
            )
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=8192,
            max_vision_patches=16384,
            max_audio_frames=1,
            max_codec_frames=1,
            max_latent_elements=524288,
            max_decoded_media_bytes=134217728,
        ),
    )
    adapter = QwenImageEditAdapter(config)
    encoded = adapter.encode(rows[0], "edit", contract_id="test")
    return root, config, rows, adapter, encoded


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree_equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_native_encoding_and_target_isolation(runtime):
    root, config, rows, adapter, encoded = runtime
    batch = encoded.payload
    assert encoded.normalization_counts == {"latent_elements": 4 * 64}
    assert batch["condition"].shape == (1, 4096, 64)
    assert batch["clean"].shape == (1, 4, 64)
    assert encoded.cost.latent_elements == 4100 * 64
    assert batch["img_shapes"] == [[(1, 2, 2), (1, 64, 64)]] or batch["img_shapes"] == [
        [[1, 2, 2], [1, 64, 64]]
    ]
    other = rows[0].model_copy(deep=True)
    other.assets["target"] = rows[1].assets["target"]
    changed = adapter.encode(other, "edit", contract_id="test").payload
    assert_tree_equal(batch["prompt"], changed["prompt"])
    assert_tree_equal(batch["condition"], changed["condition"])
    assert not torch.equal(batch["clean"], changed["clean"])
    frontend = adapter.frontend()
    native = frontend.pipe
    with torch.no_grad():
        source = Image.new("RGB", (1024, 1024), "red")
        prompt, mask = native.encode_prompt("Change red to blue.", image=source, device="cpu")
        assert_tree_equal(batch["prompt"], prompt)
        assert_tree_equal(batch["prompt_mask"], mask)
        pixels = native.image_processor.preprocess(source, 1024, 1024).unsqueeze(2)
        z = native._encode_vae_image(pixels, None)
        packed = native._pack_latents(z, 1, 16, 128, 128)
        assert_tree_equal(batch["condition"], packed)
    assert all(not p.requires_grad for p in frontend.pipe.text_encoder.parameters())
    assert all(not p.requires_grad for p in frontend.pipe.vae.parameters())


def test_inference_never_reads_target(runtime):
    _, _, rows, adapter, _ = runtime
    original = adapter.encode(rows[0], "edit", contract_id="test", inference=True)
    hidden = rows[0].model_copy(deep=True)
    hidden.assets["target"] = hidden.assets["target"].model_copy(update={"uri": "unavailable.png"})
    other = adapter.encode(hidden, "edit", contract_id="test", inference=True)
    assert_tree_equal(original.payload, other.payload)
    assert not original.supervision.targets
    assert "clean" not in original.payload


def test_flow_target_and_gradients_match_native(runtime):
    _, cfg, _, adapter, encoded = runtime
    actual = load_qwen_image_edit(cfg)
    native = copy.deepcopy(actual)
    b = encoded.payload
    clean = b["clean"]
    torch.manual_seed(18)
    loss = adapter.forward(actual, b).losses.loss
    loss.backward()
    torch.manual_seed(18)
    noise = torch.randn_like(clean)
    index = (torch.rand(1) * adapter.scheduler.config.num_train_timesteps).long()
    t = adapter.scheduler.timesteps[index]
    sigma = adapter.scheduler.sigmas[index].view(-1, 1, 1)
    noisy = (1 - sigma) * clean + sigma * noise
    prediction = native(
        hidden_states=torch.cat((noisy, b["condition"]), dim=1),
        encoder_hidden_states=b["prompt"],
        encoder_hidden_states_mask=b["prompt_mask"],
        timestep=t / 1000,
        img_shapes=b["img_shapes"],
        return_dict=False,
    )[0][:, : clean.shape[1]]
    reference = (prediction.float() - (noise - clean).float()).square().mean()
    reference.backward()
    torch.testing.assert_close(loss, reference, atol=0, rtol=0)
    for (name, p), (_, q) in zip(actual.named_parameters(), native.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
    assert actual.transformer_blocks[0].attn.to_v.weight.grad.abs().sum() > 0


def test_budgets_checked_before_frozen_models(runtime):
    _, cfg, rows, _, _ = runtime
    limited = QwenImageEditAdapter(cfg.model_copy(update={"max_length": 128}))
    with pytest.raises(OmniFactoryError, match="latent sequence exceeds"):
        limited.encode(rows[0], "edit", contract_id="test")
    assert limited._frontend is None
    with pytest.raises(ValueError, match="freezes"):
        TrainConfig.model_validate(
            {**cfg.model_dump(), "components": {"vision_encoder": {"mode": "full", "lr": 1e-4}}}
        )


def test_train_resume_native_export_and_generation(runtime):
    root, cfg, rows, _, encoded = runtime
    model, _, _ = apply_policy(load_qwen_image_edit(cfg), cfg)
    assert model.is_gradient_checkpointing
    adapter = QwenImageEditAdapter(cfg)
    assert adapter._frontend is None
    adapter.forward(model, encoded.payload).losses.loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None for p in trainable)
    del model, adapter
    train(cfg)
    resumed = cfg.model_copy(
        update={"output_dir": root / "resumed", "resume_from": root / "run/checkpoint-00000001"}
    )
    train(resumed)
    first = load_file(root / "run/checkpoint-00000002/trainable.safetensors")
    second = load_file(root / "resumed/checkpoint-00000002/trainable.safetensors")
    assert_tree_equal(first, second)
    exported = root / "export"
    export_checkpoint(root / "run/checkpoint-00000002", exported)
    native = QwenImageTransformer2DModel.from_pretrained(
        exported / "transformer", local_files_only=True
    )
    assert not any("lora_" in name for name, _ in native.named_parameters())
    for folder in ("text_encoder", "vae", "processor", "tokenizer", "scheduler"):
        for source in (root / "model" / folder).rglob("*"):
            if source.is_file():
                assert (
                    source.read_bytes()
                    == (
                        exported / folder / source.relative_to(root / "model" / folder)
                    ).read_bytes()
                )
    assert (
        json.loads((exported / "transformer/config.json").read_text())["_class_name"]
        == "QwenImageTransformer2DModel"
    )
    result = infer_export(
        exported,
        root / "validation.jsonl",
        output=root / "prediction.jsonl",
        generate_image=True,
        image_steps=2,
    )
    assert result["image_predictions"] == 1
    output = json.loads((root / "prediction.jsonl").read_text())
    assert (output["width"], output["height"]) == (32, 32)


def test_edit_http_multipart_sends_only_source(runtime, tmp_path, monkeypatch):
    import base64
    import hashlib
    import io
    import threading
    from email import policy
    from email.parser import BytesParser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from omnifactory import serving

    root, cfg, rows, _, _ = runtime
    export = tmp_path / "export"
    export.mkdir()
    (export / "omnifactory_export.json").write_text(
        json.dumps(
            dict(
                adapter="qwen_image_edit",
                inference_config=cfg.model_dump(mode="json"),
            )
        )
    )
    monkeypatch.setattr(
        serving,
        "verify_export",
        lambda path: dict(
            adapter="qwen_image_edit",
            artifact_id="transport-fixture",
        ),
    )
    hidden = rows[0].model_copy(deep=True)
    hidden.assets["source"] = hidden.assets["source"].model_copy(
        update={"uri": str(root / "source-0.png")}
    )
    hidden.assets["target"] = hidden.assets["target"].model_copy(
        update={"uri": "private-missing.png"}
    )
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(hidden.model_dump_json() + "\n")
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers["Content-Type"], raw))
            image = Image.new("RGB", (32, 32), "white")
            image.putpixel((0, 0), (0, 0, 0))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}]}
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = serving.probe(
            export,
            dataset,
            tmp_path / "http",
            f"http://127.0.0.1:{server.server_port}",
            "edit-fixture",
            size=32,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    endpoint, content_type, body = received[0]
    assert endpoint == "/v1/images/edits"
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    parts = {
        part.get_param("name", header="content-disposition"): part for part in message.iter_parts()
    }
    image = Image.open(io.BytesIO(parts["image"].get_payload(decode=True)))
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert parts["prompt"].get_content() == "Change red to blue."
    assert parts["true_cfg_scale"].get_content() == "4.0"
    assert "target" not in parts and result["requests"] == 1
    assert result["results"][0]["request_body_sha256"] == hashlib.sha256(body).hexdigest()
    assert (tmp_path / "http/request-00000.request.body").read_bytes() == body
