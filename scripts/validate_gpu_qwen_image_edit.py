"""Original Qwen-Image-Edit pretrained acceptance using explicit synthetic edit pairs.

This checks execution, exact resume, native merged tensors and frozen assets.
It does not measure editing quality. HTTP serving remains a separate gate.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from diffusers import QwenImageTransformer2DModel

from omnifactory.acceptance import attach_gate, run_acceptance
from omnifactory.artifacts import sha256, verify_export
from omnifactory.datasets.image_demo import create_image_demo
from omnifactory.environment import capture_environment
from omnifactory.training.artifacts import write_json
from omnifactory.training.config import TrainConfig, read_config
from omnifactory.training.engine import load_checkpoint
from omnifactory.training.policy import merge_lora


def native_parity(checkpoint, export, output):
    expected, adapter, _, _ = load_checkpoint(checkpoint, for_export=True)
    expected = merge_lora(expected)
    native, info = QwenImageTransformer2DModel.from_pretrained(
        export / "transformer",
        local_files_only=True,
        output_loading_info=True,
        torch_dtype=torch.bfloat16,
    )
    assert not any(
        info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), info
    a, b = expected.state_dict(), native.state_dict()
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name].to(b[name].dtype), b[name], atol=0, rtol=0, msg=name)
    frozen = {}
    for folder in ("vae", "text_encoder", "processor", "tokenizer", "scheduler"):
        for path in sorted((adapter.root / folder).rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            relative = str(path.relative_to(adapter.root))
            digest = sha256(path)
            assert sha256(export / relative) == digest, relative
            frozen[relative] = digest
    write_json(
        output,
        dict(
            status="passed",
            artifact_id=verify_export(export)["artifact_id"],
            tensor_count=len(a),
            merged_tensor_max_abs=0,
            frozen_files=frozen,
            scope="native strict DiT reload and BF16 merge parity; frozen pipeline byte parity; not quality",
        ),
    )
    del expected, adapter, native, a, b
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA BF16 required")
    if args.steps < 2 or args.size < 16 or args.size > 1024 or args.size % 16:
        parser.error("steps >= 2 and output size a multiple of 16 in [16, 1024] required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    create_image_demo(root / "data", size=args.size, editing=True)
    template = read_config(
        Path(__file__).resolve().parents[1] / "examples/train-qwen-image-edit.yaml"
    )
    config = TrainConfig.model_validate(
        {
            **template.model_dump(),
            "model": str(args.model.resolve()),
            "revision": None,
            "dataset": root / "data/train.jsonl",
            "dataset_root": root / "data",
            "validation_dataset": root / "data/validation.jsonl",
            "validation_root": root / "data",
            "output_dir": root / "run",
            "max_steps": args.steps,
            "save_steps": args.steps // 2,
            "eval_steps": args.steps // 2,
            "gradient_accumulation_steps": 1,
            "image_height": args.size,
            "image_width": args.size,
            "deterministic": True,
        }
    )
    write_json(root / "recipe.json", config.model_dump(mode="json"))
    capture_environment(root / "environment")
    report = run_acceptance(config, root / "acceptance")
    native_parity(
        Path(report.gates["train"].details["checkpoint"]),
        root / "acceptance/export",
        root / "native-reference.json",
    )
    attach_gate(
        root / "acceptance/acceptance.json", root / "native-reference.json", name="native_reference"
    )
    print(
        json.dumps(
            dict(
                status="passed",
                artifact_id=report.artifact_id,
                native_reference="passed",
                quality="not_run",
                http="not_run",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
