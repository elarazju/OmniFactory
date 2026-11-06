"""Native declared-batch regression; synthetic fixtures establish no quality claim."""

import argparse
import json
from pathlib import Path

import torch

from omnifactory.adapters.contracts import check_adapter_contract
from omnifactory.adapters.runtime import adapter_class, adapter_spec, load_model
from omnifactory.data.io import iter_samples
from omnifactory.training.config import read_config
from omnifactory.training.policy import apply_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    args = parser.parse_args()
    cfg = read_config(args.config)
    torch.set_num_threads(2)
    adapter = adapter_class(cfg.adapter).from_config(cfg)
    model, _, _ = apply_policy(load_model(cfg), cfg)
    examples = []
    for sample in iter_samples(cfg.dataset):
        example = adapter.encode(sample, sample.predictions[0].id, contract_id="native-batch-probe")
        examples.append(example)
        if len(examples) == args.examples:
            break
    calls = []
    decoder = model.llm if cfg.adapter == "minicpm_o_4_5" else model

    def observe(module, positional, kwargs):
        tensor = kwargs.get("input_ids", kwargs.get("inputs_embeds"))
        calls.append(None if tensor is None else list(tensor.shape))

    handle = decoder.register_forward_pre_hook(observe, with_kwargs=True)
    report = {
        "status": "failed",
        "adapter": cfg.adapter,
        "dtype": cfg.dtype,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    try:
        report.update(
            check_adapter_contract(
                adapter_spec(cfg.adapter),
                cfg.task,
                adapter,
                model,
                examples,
                config=cfg,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        handle.remove()
        report["observed_decoder_shapes"] = calls
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
