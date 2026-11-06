"""Standalone native acceptance, including public training/export/reload contracts."""


def main():
    import argparse
    import json
    from pathlib import Path

    import torch

    from omnifactory.acceptance import run_acceptance
    from omnifactory.adapters.contracts import check_adapter_contract
    from omnifactory.adapters.runtime import adapter_class, adapter_spec, load_model
    from omnifactory.data.io import iter_samples
    from omnifactory.datasets.demo import create_demo
    from omnifactory.training.config import TrainConfig
    from omnifactory.training.policy import apply_policy
    from omnifactory.training.tiny import create_tiny

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    create_tiny(root / "model")
    create_demo(root / "data")
    config = TrainConfig(
        adapter="example_qwen_fixture",
        model=str(root / "model"),
        dataset=root / "data/train.jsonl",
        validation_dataset=root / "data/validation.jsonl",
        validation_root=root / "data",
        output_dir=root / "run",
        max_steps=2,
        save_steps=1,
        max_length=512,
        heterogeneous_batches=True,
        components={"language_model": {"mode": "lora", "rank": 2, "alpha": 4, "lr": 0.001}},
        budget={
            "max_examples": 3,
            "max_llm_tokens": 1536,
            "max_vision_patches": 128,
            "max_audio_frames": 800,
            "max_codec_frames": 1,
            "max_decoded_media_bytes": 1000000,
        },
    )
    adapter = adapter_class(config.adapter).from_config(config)
    model, _, _ = apply_policy(load_model(config), config)
    examples = [
        adapter.encode(s, "p0", contract_id="external-fixture")
        for s in list(iter_samples(config.dataset))[:3]
    ]
    contract = check_adapter_contract(
        adapter_spec(config.adapter), config.task, adapter, model, examples, config=config
    )
    report = run_acceptance(config, root / "acceptance")
    assert all(
        report.gates[name].status == "passed"
        for name in ("prepare", "train", "resume", "export", "inference")
    )
    (root / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(
        json.dumps({"contract": contract, "acceptance": report.model_dump(mode="json")}, indent=2)
    )
