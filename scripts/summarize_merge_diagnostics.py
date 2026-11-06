"""Compare two completed fixed-input merge diagnostic arms without hiding drift."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from omnifactory.artifacts import sha256
from omnifactory.training.artifacts import write_json


def difference(a, b):
    if a.shape != b.shape:
        raise ValueError(f"different fixed-input tensor shapes: {a.shape} vs {b.shape}")
    a, b = a.float(), b.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite snapshot")
    delta = a - b
    return dict(
        max_abs=float(delta.abs().max()),
        rms=float(delta.square().mean().sqrt()),
        reference_rms=float(a.square().mean().sqrt()),
        unequal=int((a != b).sum()),
        elements=a.numel(),
    )


def compare(root, left, right):
    def read(p):
        return json.loads(p.read_text())

    dirs = [root / left, root / right]
    evidence = {}
    identities = []
    for folder in dirs:
        assert read(folder / "status.json")["status"] == "passed"
        identities.append(read(folder / "identity.json"))
    for key in (
        "artifact_id",
        "checkpoint_sha256",
        "weights_sha256",
        "dataset_sha256",
        "script_sha256",
        "code",
        "cases",
    ):
        if identities[0][key] != identities[1][key]:
            raise ValueError(f"arm identity mismatch: {key}")
    if identities[0].get("sample_ids_sha256") != identities[1].get("sample_ids_sha256"):
        raise ValueError("arm identity mismatch: selected samples")
    cases = []
    for i, (a, b) in enumerate(zip(*(read(d / "cases.json") for d in dirs), strict=True)):
        assert (a["sample_id"], a["prediction_id"]) == (b["sample_id"], b["prediction_id"])
        tensor_paths = [d / f"case-{i:03d}.safetensors" for d in dirs]
        tensors = [load_file(p) for p in tensor_paths]
        assert tensors[0].keys() == tensors[1].keys()
        stats = {k: difference(tensors[0][k], tensors[1][k]) for k in tensors[0]}
        for key in ("logits", "stop"):
            if key in tensors[0]:
                arg_a, arg_b = (t[key].argmax(-1) for t in tensors)
                stats[key]["argmax_changed"] = int((arg_a != arg_b).sum())
                stats[key]["positions"] = arg_a.numel()
        generation = dict(left_finished=a["finished"], right_finished=b["finished"])
        if "tokens" in a:
            count = min(len(a["tokens"]), len(b["tokens"]))
            prefix = next((j for j in range(count) if a["tokens"][j] != b["tokens"][j]), count)
            generation.update(
                exact=a["tokens"] == b["tokens"],
                common_prefix_tokens=prefix,
                left_tokens=len(a["tokens"]),
                right_tokens=len(b["tokens"]),
                left_text=a["prediction"],
                right_text=b["prediction"],
            )
        else:
            paths = [d / f"generation-{i:03d}.safetensors" for d in dirs]
            latents = [load_file(p)["latents"] for p in paths]
            n = min(len(x) for x in latents)
            generation.update(
                left_frames=len(latents[0]),
                right_frames=len(latents[1]),
                first_patch=difference(latents[0][:4], latents[1][:4]),
                common_length_prefix=difference(latents[0][:n], latents[1][:n]),
            )
            tensor_paths += paths
        cases.append(
            dict(
                sample_id=a["sample_id"],
                left_loss=a["loss"],
                right_loss=b["loss"],
                teacher=stats,
                generation=generation,
            )
        )
        for path in tensor_paths:
            evidence[str(path.relative_to(root))] = sha256(path)
    if len(cases) != identities[0]["cases"]:
        raise ValueError("incomplete arm")
    for d in dirs:
        for name in ("identity.json", "cases.json", "runtime.json", "status.json"):
            evidence[str((d / name).relative_to(root))] = sha256(d / name)
    return dict(
        status="completed",
        left=left,
        right=right,
        artifact_id=identities[0]["artifact_id"],
        cases=cases,
        evidence=evidence,
        scope="fixed teacher inputs, per-case RNG and repeated deterministic forwards; cross-dtype noise is not compared; generated latent prefix errors are not perceptual scores",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose fresh output")
    torch.set_num_threads(4)
    write_json(args.output, compare(args.root, args.left, args.right))


if __name__ == "__main__":
    main()
