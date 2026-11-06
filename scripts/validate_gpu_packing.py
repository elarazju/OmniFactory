"""Compare pretrained CUDA BF16 packed training against independent native samples."""

import argparse
import json
import os
from pathlib import Path

import torch

from omnifactory.adapters.packing import packed_batch
from omnifactory.adapters.qwen2_5 import Qwen25Adapter, load_thinker
from omnifactory.training.artifacts import write_json
from omnifactory.training.config import TrainConfig
from omnifactory.training.engine import load_tensors
from omnifactory.training.policy import apply_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.benchmark_dir.resolve()
    config = TrainConfig.model_validate(json.loads((root / "config.json").read_text()))
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    adapter = Qwen25Adapter.from_config(config)
    torch.manual_seed(42)
    model, _, _ = apply_policy(load_thinker(config), config)
    model.train()
    # Nonzero B also exercises A gradients, not just the zero-initialized adapter.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.001)
    entries = [e for e in manifest["entries"] if e["sample_id"] in ("text-0", "image-5", "audio-4")]
    assert len(entries) == 3
    rows = [adapter.to_device(load_tensors(root / "prepared", e), model, config) for e in entries]
    counts = [int((r["labels"][:, 1:] != -100).sum()) for r in rows]
    expected = 0.0
    for row, n in zip(rows, counts):
        loss = model(**row, use_cache=False).loss * n / sum(counts)
        expected += float(loss.detach())
        loss.backward()
    ref = {n: p.grad.float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    packed = packed_batch(model, rows)
    output = adapter.forward(model, packed)
    actual = float(output.loss.detach())
    output.loss.backward()
    errors = {}
    for name, parameter in model.named_parameters():
        if name in ref:
            delta = parameter.grad.float().cpu() - ref[name]
            errors[name] = float(delta.norm() / ref[name].norm().clamp_min(1e-12))
    print(
        json.dumps(
            {
                "sdpa_independent_loss": expected,
                "fa_packed_loss": actual,
                "sdpa_gradient_relative_l2_max": max(errors.values()),
            }
        ),
        flush=True,
    )
    # Compare independent execution under the SAME kernel too. BF16 SDPA vs FA
    # and differently sized GEMMs have their own numerical error floor.
    model.zero_grad(set_to_none=True)
    fa_expected = 0.0
    for row, n in zip(rows, counts):
        loss = model(**packed_batch(model, [row]), use_cache=False).loss * n / sum(counts)
        fa_expected += float(loss.detach())
        loss.backward()
    fa_ref = {
        n: p.grad.float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    model.zero_grad(set_to_none=True)
    adapter.forward(model, packed_batch(model, rows)).loss.backward()
    fa_errors = {
        n: float((p.grad.float().cpu() - fa_ref[n]).norm() / fa_ref[n].norm().clamp_min(1e-12))
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    print(
        json.dumps(
            {
                "fa_independent_loss": fa_expected,
                "fa_gradient_relative_l2_max": max(fa_errors.values()),
            }
        ),
        flush=True,
    )
    model.zero_grad(set_to_none=True)
    padded_output = adapter.forward(model, adapter.collate_padded(rows))
    padded_loss = float(padded_output.loss.detach())
    padded_output.loss.backward()

    def grad_metrics(reference):
        total, norm, worst = 0.0, 0.0, 0.0
        for name, parameter in model.named_parameters():
            if name in reference:
                delta = parameter.grad.float().cpu() - reference[name]
                total += float(delta.square().sum())
                norm += float(reference[name].square().sum())
                worst = max(worst, float(delta.norm() / reference[name].norm().clamp_min(1e-12)))
        return {"global_relative_l2": (total / norm) ** 0.5, "worst_relative_l2": worst}

    padded_metrics = grad_metrics(ref)
    model.zero_grad(set_to_none=True)
    adapter.forward(model, packed_batch(model, rows)).loss.backward()
    packed_metrics = grad_metrics(ref)
    packed_fa_metrics = grad_metrics(fa_ref)
    print(
        json.dumps(
            {
                "padded_loss": padded_loss,
                "padded_gradients": padded_metrics,
                "packed_gradients": packed_metrics,
                "packed_fa_gradients": packed_fa_metrics,
            }
        ),
        flush=True,
    )
    # Diagnostic: retain all numerical evidence before judging acceptance.
    args.output.with_suffix(".diagnostics.json").write_text(
        json.dumps(
            {
                "sdpa_loss": expected,
                "fa_loss": fa_expected,
                "padded_loss": padded_loss,
                "packed_loss": actual,
                "padded_gradients": padded_metrics,
                "packed_gradients": packed_metrics,
                "packed_fa_gradients": packed_fa_metrics,
                "errors": errors,
                "fa_errors": fa_errors,
            },
            indent=2,
        )
    )
    # Keep all GEMMs/media features identical and replace only the varlen call
    # with independent per-segment FA calls. This isolates boundary/kernel logic
    # from BF16 GEMM-shape sensitivity (also measured for native padded SDPA).
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from omnifactory.adapters.packing import BACKEND, varlen_attention

    def segmented_reference(module, query, key, value, mask, **kwargs):
        cu = kwargs.pop("packing_cu_seqlens")
        kwargs.pop("packing_max_seqlen")
        outs = []
        for start, end in zip(cu.tolist()[:-1], cu.tolist()[1:]):
            local = torch.tensor([0, end - start], device=query.device, dtype=torch.int32)
            result, _ = varlen_attention(
                module,
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                None,
                packing_cu_seqlens=local,
                packing_max_seqlen=end - start,
                **kwargs,
            )
            outs.append(result)
        return torch.cat(outs, dim=1), None

    model.zero_grad(set_to_none=True)
    segmented = packed_batch(model, rows)
    ALL_ATTENTION_FUNCTIONS.register(BACKEND, segmented_reference)
    segment_output = adapter.forward(model, segmented)
    segment_loss = float(segment_output.loss.detach())
    segment_output.loss.backward()
    segment_grad = {
        n: p.grad.float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    model.zero_grad(set_to_none=True)
    packed = packed_batch(model, rows)  # restores real varlen backend
    adapter.forward(model, packed).loss.backward()
    segment_metrics = grad_metrics(segment_grad)
    print(
        json.dumps(
            {
                "segmented_loss": segment_loss,
                "varlen_loss": actual,
                "kernel_only_gradients": segment_metrics,
            }
        ),
        flush=True,
    )
    assert abs(actual - segment_loss) < 0.001, (actual, segment_loss)
    assert segment_metrics["global_relative_l2"] < 0.02, segment_metrics
    assert packed_metrics["global_relative_l2"] <= padded_metrics["global_relative_l2"]

    # Changing an earlier text sample cannot affect later image/audio segments.
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        original = model(**packed, use_cache=False).logits
        changed_inputs = {**packed, "input_ids": packed["input_ids"].clone()}
        changed_inputs["input_ids"][0, 10:20] = 100
        changed = model(**changed_inputs, use_cache=False).logits
        boundary = rows[0]["input_ids"].shape[1]
        isolation = float((original[:, boundary:] - changed[:, boundary:]).abs().max())
    assert isolation == 0, isolation
    write_json(
        args.output,
        dict(
            passed=True,
            segmented_loss=segment_loss,
            kernel_only_gradients=segment_metrics,
            native_padded_gradients=padded_metrics,
            independent_to_packed_gradients=packed_metrics,
            bf16_shape_sensitive=True,
            acceptance_scope="CPU FP32 native loss/all gradients; CUDA identical-GEMM segmented-vs-varlen loss and gradient; exact cross-sample isolation",
            kernel_gradient_relative_l2_tolerance=0.02,
            independent_bf16_equivalence=False,
            independent_loss=expected,
            fa_independent_loss=fa_expected,
            fa_gradient_errors=fa_errors,
            packed_loss=actual,
            maximum_gradient_relative_l2=max(errors.values()),
            gradient_errors=errors,
            cross_sample_logit_delta=isolation,
            lengths=[r["input_ids"].shape[1] for r in rows],
            gradient_checkpointing=config.gradient_checkpointing,
            kernel="flash_attn_varlen_func",
            model=config.model,
            torch=torch.__version__,
        ),
    )
    print(
        json.dumps(
            {
                "passed": True,
                "loss_delta": actual - expected,
                "gradient_relative_l2_max": max(errors.values()),
                "isolation_delta": isolation,
            }
        )
    )


if __name__ == "__main__":
    main()
