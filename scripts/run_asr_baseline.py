"""Paired base/finetuned ASR pilot, with speaker-group and waveform overlap checks."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.data.io import iter_samples
from omnifactory.training.artifacts import write_json
from omnifactory.training.asr_metrics import corpus_wer
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import read_config
from omnifactory.training.efficiency import environment
from omnifactory.training.engine import load_checkpoint, train


def check_splits(training, validation):
    def groups(rows):
        return {(s.provenance.dataset, s.provenance.group_id) for s in rows}

    def hashes(rows):
        result = {a.sha256 for s in rows for a in s.assets.values()}
        if None in result:
            raise ValueError("quality fixtures require hashes for all media")
        return result

    if not training or not validation:
        raise ValueError("nonempty train/validation datasets required")
    if any(s.provenance.split != "train" for s in training):
        raise ValueError("training rows must declare train")
    if any(s.provenance.split != "validation" for s in validation):
        raise ValueError("validation rows must declare validation")
    if groups(training) & groups(validation) or hashes(training) & hashes(validation):
        raise ValueError("train/eval speaker group or waveform overlap")


def predictions(model, adapter, config, samples, *, max_new_tokens):
    results = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            for prediction in sample.predictions:
                target = next(m for m in sample.messages if m.id == prediction.target_message_id)
                reference = "\n".join(
                    b.text for b in target.content if b.id in prediction.target_block_ids
                )
                encoded = adapter.encode(
                    sample, prediction.id, contract_id="asr-inference", inference=True
                )
                encoded.validate(inference=True)
                batch = adapter.to_device(encoded.payload, model, config)
                ids = adapter.generate_ids(model, batch, max_new_tokens=max_new_tokens)
                new = ids[0, batch["input_ids"].shape[1] :]
                finished = bool((new == adapter.processor.tokenizer.eos_token_id).any())
                text = adapter.processor.tokenizer.decode(new, skip_special_tokens=True)
                row = dict(
                    sample_id=sample.id,
                    prediction_id=prediction.id,
                    reference=reference,
                    prediction=text,
                    finished=finished,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args()
    config = read_config(args.config)
    if config.adapter not in ("qwen2_5_omni", "qwen3_omni") or config.task != "understanding_sft":
        p.error("this ASR pilot currently uses Qwen understanding adapters")
    training, validation = list(iter_samples(config.dataset)), list(iter_samples(args.validation))
    check_splits(training, validation)
    # Retain completed baseline predictions if later training fails.
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config.model_dump(mode="json"))
    write_json(root / "tested-code.json", runtime_hashes())
    write_json(
        root / "evaluation-inputs.json",
        {
            "train_jsonl_sha256": hashlib.sha256(config.dataset.read_bytes()).hexdigest(),
            "validation_jsonl_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
            "normalization": "English NFKC lowercase; retain apostrophes; punctuation to spaces; no number expansion",
        },
    )
    adapter = adapter_class(config.adapter).from_config(config)
    adapter.resolver.root = args.validation.resolve().parent
    model = load_model(config)
    before = predictions(model, adapter, config, validation, max_new_tokens=args.max_new_tokens)
    write_json(root / "base-predictions.json", before)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    result = train(config, progress=lambda row: print(json.dumps(row), flush=True))
    gc.collect()
    torch.cuda.empty_cache()
    model, adapter, restored, _ = load_checkpoint(Path(result["checkpoint"]))
    adapter.resolver.root = args.validation.resolve().parent
    after = predictions(model, adapter, restored, validation, max_new_tokens=args.max_new_tokens)
    write_json(root / "tuned-predictions.json", after)
    base_metrics, tuned_metrics = corpus_wer(before), corpus_wer(after)
    report = dict(
        environment=environment(config.device),
        training=result,
        baseline=base_metrics,
        finetuned=tuned_metrics,
        wer_delta=tuned_metrics["wer"] - base_metrics["wer"],
        baseline_truncated=sum(not r["finished"] for r in before),
        finetuned_truncated=sum(not r["finished"] for r in after),
        disjoint_speaker_groups=True,
        official_test_benchmark=False,
        quality_improved=tuned_metrics["wer"] < base_metrics["wer"],
        claim="paired small real-speech pilot; no convergence or generalization claim",
    )
    write_json(root / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
