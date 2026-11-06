"""Check agent-only depth against the full native depth on real prepared sessions.

Uses a trained checkpoint so both LoRA factors have meaningful gradients. Checks
every supplied prepared training record, deterministic reference repeats, logits,
losses and every trainable gradient. Forward/backward timings exclude optimizer.
"""

import argparse
import json
from pathlib import Path

import torch

from omnifactory._vendor.personaplex.utils.compile import no_compile
from omnifactory.artifacts import sha256
from omnifactory.training.artifacts import write_json
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.control import deterministic_training
from omnifactory.training.efficiency import StepMeasurement, environment
from omnifactory.training.engine import load_checkpoint, load_tensors
from omnifactory.training.prepare import load_prepared


def compare(reference, candidate, *, exact=False):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("different tensor shape/dtype")
    a, b = reference.float(), candidate.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("comparison needs finite selected values")
    delta = (a - b).abs()
    return dict(
        close=bool(torch.allclose(a, b, rtol=0 if exact else 1e-4, atol=0 if exact else 1e-6)),
        max_abs=float(delta.max()) if delta.numel() else 0.0,
        reference_max_abs=float(a.abs().max()) if a.numel() else 0.0,
        unequal=int((a != b).sum()),
        elements=a.numel(),
    )


def step(model, adapter, cfg, batch, *, optimized):
    adapter.config = TrainConfig.model_validate(
        {**cfg.model_dump(), "personaplex_agent_only_depth": optimized}
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(cfg.seed)
    meter = StepMeasurement(cfg.device).start()
    output = adapter.forward(model, batch)
    output.loss.backward()
    metric = meter.finish()
    metric["loss"] = float(output.loss.detach())
    grads = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                raise ValueError(f"selected parameter did not execute: {name}")
            grads[name] = p.grad.detach().cpu().clone()
    return metric, grads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, adapter, cfg, _ = load_checkpoint(args.checkpoint)
    if cfg.adapter != "personaplex" or cfg.device != "cuda" or cfg.personaplex_user_loss_weight:
        p.error("requires PersonaPlex CUDA checkpoint with input-only user stream")
    cfg = TrainConfig.model_validate({**cfg.model_dump(), "deterministic": True})
    manifest = load_prepared(cfg, args.prepared)
    model.train()
    cases = []
    write_json(
        root / "identity.json",
        dict(
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=sha256(args.checkpoint / "checkpoint.json"),
            trainable_sha256=sha256(args.checkpoint / "trainable.safetensors"),
            prepared_manifest_sha256=sha256(args.prepared / "manifest.json"),
            code=runtime_hashes(),
            script_sha256=sha256(Path(__file__)),
            environment=environment(cfg.device),
            rtol=1e-4,
            atol=1e-6,
        ),
    )
    with deterministic_training(cfg):
        for index, entry in enumerate(manifest["entries"]):
            batch = adapter.to_device(load_tensors(args.prepared, entry), model, cfg)
            # Same retained model; reference repeats separate kernel nondeterminism
            # from changing the causal depth prefix. No optimizer updates here.
            step(model, adapter, cfg, batch, optimized=False)
            baseline, ref_grads = step(model, adapter, cfg, batch, optimized=False)
            repeated, repeated_grads = step(model, adapter, cfg, batch, optimized=False)
            optimized, opt_grads = step(model, adapter, cfg, batch, optimized=True)
            jitter = {n: compare(g, repeated_grads[n], exact=True) for n, g in ref_grads.items()}
            gradients = {n: compare(g, opt_grads[n]) for n, g in ref_grads.items()}
            del ref_grads, repeated_grads, opt_grads
            model.zero_grad(set_to_none=True)
            logits = {}
            with torch.no_grad(), no_compile():
                before = model.forward_train(batch["codes"], voice_state=batch["voice_state"])
                ref_text = before.text_logits[before.text_mask].cpu()
                agent_mask = before.mask[:, :8].cpu()
                ref_audio = before.logits[:, :8][before.mask[:, :8]].cpu()
                del before
                after = model.forward_train(
                    batch["codes"], voice_state=batch["voice_state"], agent_only_depth=True
                )
                if not torch.equal(after.mask.cpu(), agent_mask):
                    raise ValueError("different native valid frames")
                logits["text"] = compare(ref_text, after.text_logits[after.text_mask].cpu())
                logits["agent"] = compare(ref_audio, after.logits[after.mask].cpu())
                del after, ref_audio, ref_text, agent_mask
            loss_close = (
                abs(baseline["loss"] - optimized["loss"]) <= 1e-6 + abs(baseline["loss"]) * 1e-4
            )
            passed = (
                baseline["loss"] == repeated["loss"]
                and loss_close
                and all(
                    v["close"] for v in [*jitter.values(), *gradients.values(), *logits.values()]
                )
            )
            case = dict(
                sample_id=entry["sample_id"],
                prediction_id=entry["prediction_id"],
                file_sha256=entry["sha256"],
                status="passed" if passed else "failed",
                baseline=baseline,
                repeated=repeated,
                optimized=optimized,
                baseline_jitter=jitter,
                gradients=gradients,
                logits=logits,
                loss_close=loss_close,
            )
            write_json(root / f"case-{index:03d}.json", case)
            cases.append(
                dict(
                    sample_id=entry["sample_id"],
                    status=case["status"],
                    max_gradient_delta=max(g["max_abs"] for g in gradients.values()),
                    max_logit_delta=max(g["max_abs"] for g in logits.values()),
                )
            )
            print(json.dumps(cases[-1]), flush=True)
            del batch
    report = dict(
        status="passed" if cases and all(c["status"] == "passed" for c in cases) else "failed",
        cases=cases,
        scope="trained CUDA weights, real prepared sessions, logits/loss/trainable gradients; not quality",
    )
    write_json(root / "report.json", report)
    if report["status"] != "passed":
        raise ValueError("depth comparison failed; retain case evidence and investigate")


if __name__ == "__main__":
    main()
