"""Score the paired ASR checkpoint after CPU FP32 LoRA merge and BF16 rounding.

This isolates the deployment-precision transformation without writing another
70 GB full model. It is an in-memory precision check, not an exported-model
reload or HTTP quality benchmark. Run after run_asr_baseline.py has exited.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from run_asr_baseline import check_splits, predictions

from omnifactory.adapters.runtime import load_model
from omnifactory.data.io import iter_samples
from omnifactory.training.artifacts import write_json
from omnifactory.training.asr_metrics import corpus_wer
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.engine import load_checkpoint
from omnifactory.training.policy import merge_lora


def rounded_native_state(model):
    """Round checkpoint tensors only; native construction owns transient buffers."""
    return {
        key: value.to(torch.bfloat16) if value.is_floating_point() else value
        for key, value in model.state_dict().items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality-run", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error("this precision comparison requires CUDA BF16")
    source = args.quality_run
    paired = json.loads((source / "report.json").read_text())
    inputs = json.loads((source / "evaluation-inputs.json").read_text())
    recipe = json.loads((source / "config.json").read_text())
    if recipe["adapter"] != "qwen3_omni" or recipe["quantization"] != "none":
        p.error("this protocol targets the unquantized Qwen3 understanding pilot")
    if runtime_hashes() != json.loads((source / "tested-code.json").read_text()):
        p.error("runtime source changed since the paired ASR run")
    if (
        hashlib.sha256(args.validation.read_bytes()).hexdigest()
        != inputs["validation_jsonl_sha256"]
    ):
        p.error("validation data changed since the paired ASR run")
    args.output.mkdir(parents=True, exist_ok=False)
    model, adapter, config, _ = load_checkpoint(
        Path(paired["training"]["checkpoint"]), for_export=True
    )
    if hashlib.sha256(config.dataset.read_bytes()).hexdigest() != inputs["train_jsonl_sha256"]:
        raise ValueError("training data changed since the paired ASR run")
    samples = list(iter_samples(args.validation))
    check_splits(list(iter_samples(config.dataset)), samples)
    # Same arithmetic/storage path as native assembly: FP32 CPU merge, then
    # BF16 weights. Do not merge LoRA using BF16 matmul before rounding.
    model = merge_lora(model)
    rounded = rounded_native_state(model)
    del model
    gc.collect()
    config = config.model_copy(update={"device": "cuda", "dtype": "bfloat16"})
    # Fresh native construction preserves nonpersistent FP32 RoPE buffers. A
    # blanket model.bfloat16() would cast those too, unlike checkpoint reload.
    model = load_model(config).eval()
    model.load_state_dict(rounded, strict=True)
    del rounded
    gc.collect()
    adapter.resolver.root = args.validation.resolve().parent
    rows = predictions(model, adapter, config, samples, max_new_tokens=inputs["max_new_tokens"])
    previous = json.loads((source / "tuned-predictions.json").read_text())

    def identity(row):
        return (row["sample_id"], row["prediction_id"], row["reference"])

    if list(map(identity, rows)) != list(map(identity, previous)):
        raise ValueError("prediction/reference identity differs from the paired run")
    metrics = corpus_wer(rows)
    report = dict(
        precision_path="CPU FP32 LoRA merge -> BF16 parameters -> CUDA BF16 inference",
        baseline=paired["baseline"],
        unmerged_adapter=paired["finetuned"],
        merged_bf16=metrics,
        wer_delta_vs_unmerged=metrics["wer"] - paired["finetuned"]["wer"],
        wer_delta_vs_base=metrics["wer"] - paired["baseline"]["wer"],
        truncated=sum(not row["finished"] for row in rows),
        identical_decoded_predictions=sum(
            a["prediction"] == b["prediction"] for a, b in zip(rows, previous)
        ),
        validation_jsonl_sha256=inputs["validation_jsonl_sha256"],
        runtime_matches_paired_run=True,
        native_export_reload_verified=False,
        http_quality_verified=False,
        official_test_benchmark=False,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write_json(args.output / "predictions.json", rows)
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
