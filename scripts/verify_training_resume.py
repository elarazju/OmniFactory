"""Resume an intermediate checkpoint and compare weights, optimizer and RNG exactly."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from omnifactory.artifacts import sha256
from omnifactory.training.artifacts import write_json
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import train


def equal_tree(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        if (
            not isinstance(b, torch.Tensor)
            or a.dtype != b.dtype
            or a.shape != b.shape
            or not torch.equal(a, b)
        ):
            raise ValueError(f"tensor mismatch: {path}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(f"mapping mismatch: {path}")
        for key in a:
            equal_tree(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError(f"sequence mismatch: {path}")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            equal_tree(x, y, f"{path}[{i}]")
    elif type(a) is not type(b) or a != b:
        raise ValueError(f"value mismatch: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    prior = json.loads((args.checkpoint / "checkpoint.json").read_text())
    expected = json.loads((args.expected / "checkpoint.json").read_text())
    if prior["contract_id"] != expected["contract_id"]:
        raise ValueError("checkpoint contracts differ")
    cfg = TrainConfig.model_validate(
        {
            **prior["config"],
            "output_dir": args.output,
            "resume_from": args.checkpoint,
            "max_steps": expected["state"]["step"],
        }
    )
    result = train(cfg, progress=lambda r: print(json.dumps(r), flush=True))
    actual = Path(result["checkpoint"])
    equal_tree(
        load_file(args.expected / "trainable.safetensors"),
        load_file(actual / "trainable.safetensors"),
    )
    equal_tree(
        torch.load(args.expected / "optimizer.pt", map_location="cpu", weights_only=True),
        torch.load(actual / "optimizer.pt", map_location="cpu", weights_only=True),
    )
    write_json(
        args.output / "resume-parity.json",
        dict(
            status="passed",
            scope="all trainable tensors, hidden master weights, optimizer moments/groups and CPU/CUDA RNG",
            expected=str(args.expected.resolve()),
            actual=str(actual.resolve()),
            code=runtime_hashes(),
            evidence={
                str(p.resolve()): sha256(p)
                for root in (args.expected, actual)
                for p in (
                    root / "trainable.safetensors",
                    root / "optimizer.pt",
                    root / "checkpoint.json",
                )
            },
        ),
    )


if __name__ == "__main__":
    main()
