"""Trace checkpoint/native-export drift through weights, fixed inputs and generation.

Read-only with respect to the checkpoint/export. Each arm is an independent
process. This diagnostic records differences; generation equality is not assumed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors import safe_open
from safetensors.torch import save_file

from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.artifacts import sha256, verify_export
from omnifactory.data.io import iter_samples
from omnifactory.training.artifacts import write_json
from omnifactory.training.benchmark import runtime_hashes
from omnifactory.training.config import TrainConfig
from omnifactory.training.control import deterministic_training
from omnifactory.training.engine import load_checkpoint
from omnifactory.training.policy import merge_lora


def exported_tensors(root):
    index = root / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    with safe_open(root / "model.safetensors", framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


@torch.no_grad()
def audit_weights(checkpoint, exported, output):
    model, _, cfg, _ = load_checkpoint(checkpoint, for_export=True)
    modules = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, BaseTunerLayer):
            continue
        base = layer.get_base_layer().weight.detach()
        delta = layer.get_delta_weight("default").detach()
        merged = base + delta
        rounded_base = base.bfloat16().float()
        modules[name] = dict(
            base_dtype=str(base.dtype),
            lora_dtype=str(layer.lora_A["default"].weight.dtype),
            update_max_abs=float(delta.abs().max()),
            update_rms=float(delta.square().mean().sqrt()),
            rounded_update_surviving=int((merged.bfloat16() != base.bfloat16()).sum()),
            elements=base.numel(),
            original_vs_rounded_source_merge_unequal=int(
                (merged.bfloat16() != (rounded_base + delta).bfloat16()).sum()
            ),
        )
    model = merge_lora(model)
    mapping = exported_tensors(exported)
    prefix = (
        "thinker."
        if cfg.adapter == "qwen2_5_omni" and any(k.startswith("thinker.") for k in mapping)
        else ""
    )
    checked = {}
    state = model.state_dict()
    for shard in sorted(set(mapping.values())):
        with safe_open(exported / shard, framework="pt") as f:
            for name, value in state.items():
                if cfg.adapter == "voxcpm2" and name.startswith("audio_vae."):
                    continue
                key = prefix + name
                if key not in mapping:
                    raise ValueError(f"missing native parameter: {key}")
                if mapping[key] != shard:
                    continue
                actual = f.get_tensor(key)
                expected = value.detach().cpu().to(actual.dtype)
                difference = (actual.float() - expected.float()).abs()
                checked[key] = dict(
                    exact=bool(torch.equal(actual, expected)),
                    max_abs=float(difference.max()),
                    dtype=str(actual.dtype),
                )
    if not checked or not all(v["exact"] for v in checked.values()):
        raise ValueError("export does not equal prescribed FP32 merge and native dtype conversion")
    write_json(
        output / "weights.json",
        dict(
            status="passed",
            tensors=checked,
            lora=modules,
            scope="actual exported tensors vs fresh checkpoint FP32 merge; BF16 rounding inventory is descriptive",
        ),
    )


def set_model_dtype(model, cfg):
    # Reconstruct native runtime buffers exactly as normal loading does. A blanket
    # .to(BF16) would also round Qwen's nonpersistent FP32 rotary frequencies.
    native = load_model(cfg)
    native.load_state_dict(model.state_dict(), strict=True)
    return native.eval()


@torch.no_grad()
def probe(args, root, metadata):
    cfg = TrainConfig.model_validate(metadata["config"])
    dtype = "float32" if args.arm.endswith("fp32") else "bfloat16"
    cfg = cfg.model_copy(
        update=dict(
            device="cuda",
            dtype=dtype,
            deterministic=True,
            gradient_checkpointing=False,
            packing="none",
            feature_cache=None,
            dataset=args.data,
            dataset_root=args.data.parent,
        )
    )
    if args.arm.startswith("adapter"):
        model, _, _, _ = load_checkpoint(
            args.checkpoint, device="cuda", for_export=dtype == "float32"
        )
    elif args.arm in ("runtime_merge_bf16", "merged_fp32"):
        model, _, _, _ = load_checkpoint(args.checkpoint, for_export=True)
        # Isolate using the runtime-rounded base instead of the original FP32 base.
        if args.arm == "runtime_merge_bf16":
            for layer in model.modules():
                if isinstance(layer, BaseTunerLayer):
                    weight = layer.get_base_layer().weight
                    weight.data.copy_(weight.data.bfloat16().float())
        model = merge_lora(model)
        model = set_model_dtype(model, cfg)
    else:
        cfg = cfg.model_copy(update=dict(model=str(args.export), revision=None))
        model = load_model(cfg)
    model.eval()
    adapter = adapter_class(cfg.adapter).from_config(cfg)
    all_cases = list(iter_samples(args.data))
    if args.sample_ids:
        ids = json.loads(args.sample_ids.read_text())
        by_id = {s.id: s for s in all_cases}
        if len(ids) != args.cases or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError("explicit IDs must be unique, present and match --cases")
        cases = [by_id[i] for i in ids]
    else:
        cases = all_cases[: args.cases]
    results = []
    with deterministic_training(cfg), torch.jit.optimized_execution(False):
        for i, sample in enumerate(cases):
            pid = sample.predictions[0].id
            batch = adapter.to_device(
                adapter.encode(sample, pid, contract_id="merge-probe").payload, model, cfg
            )

            def snapshot():
                captured = {}
                hooks = []
                if cfg.adapter == "voxcpm2":
                    for key, module in [
                        ("velocity", model.feat_decoder.estimator),
                        ("stop", model.stop_head),
                    ]:
                        hooks.append(
                            module.register_forward_hook(
                                lambda m, a, o, key=key: captured.update(
                                    {key: o.detach().float().cpu().clone()}
                                )
                            )
                        )
                try:
                    torch.manual_seed(1337 + i)
                    result = adapter.forward(model, batch)
                    if cfg.adapter == "qwen2_5_omni":
                        mask = batch["labels"][:, 1:] != -100
                        captured["logits"] = result.logits[:, :-1][mask].float().cpu()
                    return captured, float(result.loss)
                finally:
                    for hook in hooks:
                        hook.remove()

            before, loss = snapshot()
            repeat, repeated_loss = snapshot()
            jitter = {k: float((v - repeat[k]).abs().max()) for k, v in before.items()}
            if any(jitter.values()) or loss != repeated_loss:
                raise ValueError(f"repeat forward is not deterministic: {jitter}")
            save_file(before, root / f"case-{i:03d}.safetensors")
            record = dict(sample_id=sample.id, prediction_id=pid, loss=loss, repeat_max_abs=jitter)
            prompt = adapter.to_device(
                adapter.encode(sample, pid, contract_id="merge-probe", inference=True).payload,
                model,
                cfg,
            )
            seed = int.from_bytes(
                hashlib.sha256(f"{cfg.seed}:{sample.id}:{pid}".encode()).digest()[:4], "little"
            )
            record["generation_seed"] = seed
            torch.manual_seed(seed)
            if cfg.adapter == "voxcpm2":
                generated = adapter.generate_speech(
                    model, prompt, max_new_tokens=256, max_codec_tokens=256
                )
                latent = torch.tensor(generated["audio_latents"])
                save_file({"latents": latent}, root / f"generation-{i:03d}.safetensors")
                record.update(finished=generated["speech_finished"], generated_frames=len(latent))
            else:
                ids = adapter.generate_ids(model, prompt, max_new_tokens=256)
                new = ids[0, prompt["input_ids"].shape[1] :].cpu()
                record.update(
                    tokens=new.tolist(),
                    prediction=adapter.processor.tokenizer.decode(new, skip_special_tokens=True),
                    finished=bool((new == adapter.processor.tokenizer.eos_token_id).any()),
                )
            results.append(record)
            write_json(root / "cases.json", results)
            print(json.dumps(dict(arm=args.arm, **record)), flush=True)
    write_json(
        root / "runtime.json",
        dict(
            model_dtype=dtype,
            trainable_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
            effective_generation_config=getattr(model, "generation_config", None).to_dict()
            if hasattr(model, "generation_config")
            else None,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=int, default=8)
    p.add_argument(
        "--sample-ids",
        type=Path,
        help="JSON list; explicit diagnostic selection, not a quality benchmark",
    )
    p.add_argument(
        "--arm",
        choices=(
            "weights",
            "adapter_bf16",
            "export_bf16",
            "adapter_fp32",
            "export_fp32",
            "runtime_merge_bf16",
            "merged_fp32",
        ),
        required=True,
    )
    args = p.parse_args()
    if args.cases < 1:
        p.error("cases must be positive")
    root = args.output / args.arm
    root.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.checkpoint / "checkpoint.json").read_text())
    if metadata["config"]["adapter"] not in ("qwen2_5_omni", "voxcpm2"):
        p.error("only Qwen2.5 understanding and VoxCPM2 supported")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    artifact = verify_export(args.export)
    write_json(
        root / "identity.json",
        dict(
            artifact_id=artifact["artifact_id"],
            checkpoint_sha256=sha256(args.checkpoint / "checkpoint.json"),
            weights_sha256=sha256(args.checkpoint / "trainable.safetensors"),
            dataset_sha256=sha256(args.data),
            script_sha256=sha256(Path(__file__)),
            code=runtime_hashes(),
            cases=args.cases,
            sample_ids_sha256=sha256(args.sample_ids) if args.sample_ids else None,
            arm=args.arm,
        ),
    )
    if args.arm == "weights":
        audit_weights(args.checkpoint, args.export, root)
    else:
        probe(args, root, metadata)
    write_json(
        root / "status.json",
        dict(
            status="passed",
            scope="diagnostic execution and deterministic repeats; differences are not acceptance failures",
        ),
    )


if __name__ == "__main__":
    main()
