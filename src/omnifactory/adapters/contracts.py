"""Reusable adapter acceptance. Heavy execution dependencies are imported on demand."""

from omnifactory.errors import OmniFactoryError


def validate_encoded(spec, task, example, *, inference=False):
    capability = spec.batch_capability(task)
    if inference and any(
        key in example.payload for key in ("labels", "codec_labels", "speech_spans")
    ):
        raise OmniFactoryError(
            "E_TARGET_LEAK", "inference native payload includes target-only fields"
        )
    example.validate(inference=inference)
    kinds = {t.representation.kind for t in example.supervision.targets}
    kinds.update(s.representation.kind for s in example.supervision.streams)
    if not kinds <= set(capability.supervision):
        raise OmniFactoryError("E_ADAPTER_SUPERVISION", "undeclared supervision representation")
    if example.supervision.streams and not capability.timed:
        raise OmniFactoryError("E_ADAPTER_SUPERVISION", "timed streams are not declared")
    return example


def collate_task(adapter, config, rows, costs):
    """Return actual forward groups; never relabel an adapter loop as a native batch."""
    from omnifactory.adapters.runtime import task_batch_capability

    capability = task_batch_capability(config)
    if not rows or len(rows) != len(costs) or len({r.contract_id for r in costs}) != 1:
        raise OmniFactoryError(
            "E_BATCH_CONTRACT", "nonempty matching rows and one contract required"
        )
    mode, reason = capability.decision(
        costs, packing=config.packing != "none", timed=config.duplex_schedule is not None
    )
    groups = (
        [adapter.collate([r]) for r in rows] if mode == "sequential" else [adapter.collate(rows)]
    )
    if mode != "sequential" and (
        "examples" in groups[0] or "packing_rows" in groups[0] and mode != "varlen"
    ):
        raise OmniFactoryError(
            "E_BATCH_CAPABILITY", "declared native batch returned a per-example loop"
        )
    if mode == "padded" and capability.input_layout != "native":
        if groups[0].get("input_ids") is None or groups[0]["input_ids"].shape[0] != len(rows):
            raise OmniFactoryError(
                "E_BATCH_CAPABILITY", "native batch axis does not match declared group"
            )
    if capability.input_layout != "native":
        for batch in groups:
            validate_batch(batch, layout=capability.input_layout, adapter=adapter)
    return groups, {
        "mode": mode,
        "fallback_reason": reason,
        "media_execution": capability.media_execution,
        "examples": len(rows),
        "forward_groups": [1] * len(rows) if mode == "sequential" else [len(rows)],
        "modalities": [list(r.modalities) for r in costs],
    }


def validate_batch(batch, *, layout="token_sequence", adapter=None):
    """Validate common discrete fields without imposing token axes on latent adapters."""
    import torch

    if "examples" in batch or "packing_rows" in batch:
        for row in batch.get("examples", batch.get("packing_rows", [])):
            validate_batch(row, layout=layout, adapter=adapter)
        return
    ids = batch.get("input_ids")
    if ids is None:
        return
    if layout == "mimo_codebooks":
        g, channels = adapter.g, adapter.channels
        mask, labels = batch.get("attention_mask"), batch.get("labels")
        if (
            not isinstance(ids, torch.Tensor)
            or ids.ndim != 3
            or ids.dtype != torch.long
            or ids.shape[1] != channels + 1
            or ids.shape[-1] % g
            or mask is None
            or mask.shape != (ids.shape[0], ids.shape[-1] // g)
            or not ((mask == 0) | (mask == 1)).all()
        ):
            raise OmniFactoryError(
                "E_BATCH_SHAPE", "invalid MiMo batch/codebook/patch axes or mask"
            )
        if labels is not None and (
            labels.shape != ids.shape
            or labels.dtype != torch.long
            or (
                labels.masked_select(
                    ~mask.bool().repeat_interleave(g, -1)[:, None, :].expand_as(labels)
                )
                != -100
            ).any()
        ):
            raise OmniFactoryError("E_BATCH_MASK", "invalid MiMo labels or supervised padding")
        return
    if layout == "qwen3_tts_pairs":
        if ids.ndim != 3 or ids.shape[-1] != 2 or ids.dtype != torch.long:
            raise OmniFactoryError(
                "E_BATCH_SHAPE", "Qwen3-TTS input requires [batch,time,text/codec]"
            )
        shape = ids.shape[:2]
        for key in ("text_mask", "codec_mask", "frame_mask", "attention_mask"):
            value = batch.get(key)
            if value is None or value.shape != shape or not ((value == 0) | (value == 1)).all():
                raise OmniFactoryError("E_BATCH_MASK", "invalid Qwen3-TTS " + key)
        codes, primary, residual = (
            batch.get(k) for k in ("codec_ids", "primary_labels", "residual_labels")
        )
        if (
            codes is None
            or codes.shape != (*shape, 16)
            or codes.dtype != torch.long
            or primary is None
            or primary.shape != shape
            or primary.dtype != torch.long
            or residual is None
            or residual.shape != (int(batch["frame_mask"].sum()), 15)
            or residual.dtype != torch.long
        ):
            raise OmniFactoryError(
                "E_BATCH_SHAPE", "invalid Qwen3-TTS primary/residual codebook axes"
            )
        if (primary[~batch["attention_mask"].bool()] != -100).any():
            raise OmniFactoryError("E_BATCH_MASK", "Qwen3-TTS padding supervises primary targets")
        return
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.dtype != torch.long:
        raise OmniFactoryError("E_BATCH_SHAPE", "input_ids must be a 2D int64 tensor")
    mask, labels = batch.get("attention_mask"), batch.get("labels")
    if mask is not None and (mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all()):
        raise OmniFactoryError("E_BATCH_MASK", "invalid attention mask")
    if labels is not None:
        if labels.shape != ids.shape or labels.dtype != torch.long:
            raise OmniFactoryError("E_BATCH_SHAPE", "labels must match input_ids")
        if mask is not None and (labels[~mask.bool()] != -100).any():
            raise OmniFactoryError("E_BATCH_MASK", "padding must not supervise targets")


def check_adapter_contract(
    spec,
    task,
    adapter,
    model,
    examples,
    *,
    config,
    atol=2e-5,
    rtol=2e-5,
    export_reload=None,
    forward_probe=None,
    reload_probe_factory=None,
):
    """Check declared batching against independent losses and all trainable gradients.

    Caller supplies native fixtures and optionally an export/reload callback returning
    (adapter, model). Eval mode disables dropout; training gradients remain enabled.
    Existing gradients, model mode and RNG states are restored on return.
    Flow/diffusion/custom objectives require forward_probe(model, batch), which must
    supply fixed per-sample noise/time from fixture identifiers for both passes.
    Arbitrary reseeding cannot establish equivalence for shape-dependent RNG draws.
    With controlled export/reload, reload_probe_factory(reloaded_adapter) must return
    the same controlled objective bound to that adapter and accepting the new model.
    A closure bound to the source adapter/model must never stand in for the reload.
    """
    import torch

    from omnifactory.adapters.supervision import expected_losses
    from omnifactory.training.losses import LossBundle

    if (
        any(
            o.kind not in ("cross_entropy", "mse")
            for e in examples
            for o in e.supervision.objectives
        )
        and forward_probe is None
    ):
        raise OmniFactoryError(
            "E_ADAPTER_PROBE", "stochastic/custom objectives require a controlled forward_probe"
        )
    if forward_probe is not None and export_reload is not None and reload_probe_factory is None:
        raise OmniFactoryError(
            "E_ADAPTER_PROBE", "controlled export/reload requires reload_probe_factory"
        )
    forward = adapter.forward if forward_probe is None else forward_probe
    for example in examples:
        validate_encoded(spec, task, example)
    expected = expected_losses((e.supervision, e.normalization_counts) for e in examples)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not parameters:
        raise OmniFactoryError("E_ADAPTER_CONTRACT", "gradient checks require trainable parameters")
    previous = {n: p.grad for n, p in parameters.items()}
    training = model.training
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        singles = [
            forward(model, adapter.to_device(adapter.collate([e.payload]), model, config)).losses
            for e in examples
        ]
        independent = LossBundle.combine(singles)
        independent.validate_expected(expected)
        if not independent.loss.requires_grad:
            raise OmniFactoryError("E_ADAPTER_GRADIENT", "loss has no trainable path")
        independent.loss.backward()
        if not any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in parameters.values()
        ):
            raise OmniFactoryError("E_ADAPTER_GRADIENT", "no finite nonzero trainable gradients")
        gradients = {
            n: None if p.grad is None else p.grad.detach().clone() for n, p in parameters.items()
        }
        model.zero_grad(set_to_none=True)
        groups, execution = collate_task(
            adapter, config, [e.payload for e in examples], [e.cost for e in examples]
        )
        combined = LossBundle.combine(
            [forward(model, adapter.to_device(b, model, config)).losses for b in groups]
        )
        combined.validate_expected(expected)
        torch.testing.assert_close(combined.loss, independent.loss, atol=atol, rtol=rtol)
        combined.loss.backward()
        max_error = 0.0
        for name, parameter in parameters.items():
            reference = gradients[name]
            if (reference is None) != (parameter.grad is None):
                raise OmniFactoryError("E_ADAPTER_GRADIENT", f"gradient path changed: {name}")
            if reference is not None:
                torch.testing.assert_close(
                    parameter.grad, reference, atol=atol, rtol=rtol, msg=lambda m: name + ": " + m
                )
                max_error = max(max_error, float((parameter.grad - reference).abs().max()))
        report = {
            "execution": execution,
            "loss_absolute_error": float((combined.loss - independent.loss).detach().abs()),
            "gradient_max_absolute_error": max_error,
            "atol": atol,
            "rtol": rtol,
            "export_reload": "not_run",
        }
        if export_reload is not None:
            reloaded_adapter, reloaded = export_reload()
            reload_forward = (
                reloaded_adapter.forward
                if forward_probe is None
                else reload_probe_factory(reloaded_adapter)
            )
            reloaded.eval()
            with torch.no_grad():
                losses = LossBundle.combine(
                    [
                        reload_forward(
                            reloaded,
                            reloaded_adapter.to_device(
                                reloaded_adapter.collate([e.payload]), reloaded, config
                            ),
                        ).losses
                        for e in examples
                    ]
                )
            losses.validate_expected(expected)
            torch.testing.assert_close(losses.loss, independent.loss, atol=atol, rtol=rtol)
            report["export_reload"] = "passed"
        return report
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in parameters.items():
            parameter.grad = previous[name]
        model.train(training)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
