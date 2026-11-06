"""Native CosyVoice3 LLM/flow acceptance on chapter-held-out recordings."""

import argparse
import json
from pathlib import Path

import torch
from validate_gpu_qwen3_tts import prepare_data, release, validate_waveforms

from omnifactory.acceptance import attach_gate, run_acceptance
from omnifactory.adapters.cosyvoice3_model import (
    build_flow,
    build_talker,
    load_weights,
    native_config,
)
from omnifactory.artifacts import sha256, verify_export
from omnifactory.environment import capture_environment
from omnifactory.training.artifacts import write_json
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_checkpoint
from omnifactory.training.policy import merge_lora


def native_parity(checkpoint, export, output):
    model, adapter, cfg, _ = load_checkpoint(checkpoint, for_export=True)
    model = merge_lora(model)
    native = native_config(export)
    key, stage = ("llm", "talker") if cfg.task == "speech_sft" else ("flow", "flow")
    module = build_talker(export, native["llm"]) if key == "llm" else build_flow(native["flow"])
    load_weights(module, export / f"{key}.pt")
    expected, actual = getattr(model, stage).state_dict(), module.state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], atol=0, rtol=0)
    frozen = [
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v3.onnx",
        "flow.pt" if key == "llm" else "llm.pt",
    ]
    assert all(sha256(export / name) == sha256(adapter.root / name) for name in frozen)
    write_json(
        output,
        dict(
            status="passed",
            artifact_id=verify_export(export)["artifact_id"],
            tensor_count=len(actual),
            merged_tensor_max_abs=0,
            frozen_files=frozen,
            trained_stage=key,
            scope="native strict state reload and frozen bytes; not speech quality",
        ),
    )
    del model, adapter, module, expected, actual
    release()


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("speech_sft", "flow_sft"), required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA BF16 required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = prepare_data(args.corpus.resolve(), root / "data")
    reference_text = None
    for batch in pq.ParquetFile(args.corpus / "clean/validation/0000.parquet").iter_batches(
        batch_size=32
    ):
        for row in batch.to_pylist():
            if row["id"] == reference.stem:
                reference_text = row["text"]
                break
        if reference_text:
            break
    assert reference_text
    component = "speech_generator" if args.task == "speech_sft" else "audio_decoder"
    cfg = TrainConfig(
        model=str(args.model.resolve()),
        adapter="cosyvoice3",
        task=args.task,
        dataset=root / "data/train.jsonl",
        validation_dataset=root / "data/validation.jsonl",
        output_dir=root / "run",
        speech_reference=reference,
        speech_reference_sha256=sha256(reference),
        speech_reference_text=reference_text,
        device="cuda",
        dtype="bfloat16",
        attention="sdpa",
        gradient_checkpointing=True,
        deterministic=args.task == "flow_sft",
        max_steps=args.steps,
        save_steps=max(1, args.steps // 2),
        eval_steps=max(1, args.steps // 2),
        gradient_accumulation_steps=2,
        max_length=1024,
        max_audio_seconds=8,
        max_codec_tokens=256,
        components={
            component: dict(
                mode="lora",
                rank=8,
                alpha=16,
                lr=1e-4,
                target_modules=("q_proj", "v_proj")
                if args.task == "speech_sft"
                else ("to_q", "to_v"),
            )
        },
        budget=dict(
            max_examples=1,
            max_llm_tokens=1024,
            max_codec_frames=256,
            max_audio_frames=512,
            max_vision_patches=1,
            max_decoded_media_bytes=16777216,
            max_latent_elements=40960,
        ),
    )
    write_json(root / "recipe.json", cfg.model_dump(mode="json"))
    capture_environment(root / "environment")
    report = run_acceptance(cfg, root / "acceptance")
    waveform = validate_waveforms(root)
    native_parity(
        Path(report.gates["train"].details["checkpoint"]),
        root / "acceptance/export",
        root / "native-reference.json",
    )
    attach_gate(
        root / "acceptance/acceptance.json", root / "native-reference.json", name="native_reference"
    )
    print(
        json.dumps(dict(task=args.task, waveforms=waveform, native_reference="passed")), flush=True
    )
    if waveform["status"] != "passed":
        raise RuntimeError(waveform["error"])


if __name__ == "__main__":
    main()
