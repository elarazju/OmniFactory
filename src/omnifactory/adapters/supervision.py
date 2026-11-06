"""Tensor validation and measurable supervision, shared by every runtime family."""

import math

import torch
from pydantic import BaseModel

from omnifactory.data.supervision import (
    DiscreteRepresentation,
    LossObjective,
    Normalizer,
    ReductionMeasure,
    SupervisionPlan,
    SupervisionTarget,
    TensorRef,
)
from omnifactory.errors import OmniFactoryError


def tensor_ref(key, value, axes):
    return TensorRef(
        key=key, shape=tuple(value.shape), axes=axes, dtype=str(value.dtype).removeprefix("torch.")
    )


def tensor_references(plan):
    refs = {}

    def visit(item):
        if isinstance(item, TensorRef):
            if item.key in refs and refs[item.key] != item:
                raise OmniFactoryError(
                    "E_TENSOR_REFERENCE", f"conflicting declarations: {item.key}"
                )
            refs[item.key] = item
        elif isinstance(item, BaseModel):
            for name in type(item).model_fields:
                visit(getattr(item, name))
        elif isinstance(item, (tuple, list)):
            for value in item:
                visit(value)

    visit(plan)
    return refs


def _validate_values(values, valid, representation, axes):
    if representation.kind == "continuous":
        if not torch.isfinite(values).all():
            raise OmniFactoryError("E_CONTINUOUS_VALUES", "continuous tensors must be finite")
        return
    if (values[valid] < 0).any() or (values[valid] == representation.ignore_index).any():
        raise OmniFactoryError(
            "E_DISCRETE_VALUES", "supervised symbols must be nonnegative and not ignored"
        )
    sizes = representation.vocabulary_sizes
    if not sizes:
        return
    axis = representation.codebook_axis
    if axis is None:
        if (values[valid] >= sizes[0]).any():
            raise OmniFactoryError("E_DISCRETE_RANGE", "symbol exceeds vocabulary")
    else:
        dim = axes.index(axis)
        for i, size in enumerate(sizes):
            if (values.select(dim, i)[valid.select(dim, i)] >= size).any():
                raise OmniFactoryError("E_DISCRETE_RANGE", f"codebook {i} exceeds vocabulary")


def validate_supervision(plan: SupervisionPlan, tensors):
    for key, ref in tensor_references(plan).items():
        value = tensors.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != ref.shape
            or str(value.dtype) != "torch." + ref.dtype
        ):
            raise OmniFactoryError("E_TENSOR_REFERENCE", f"missing or mismatched tensor: {key}")
    for target in plan.targets:
        values, mask = tensors[target.values.key], tensors[target.mask.key]
        expanded = mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim)).expand_as(values)
        _validate_values(values, expanded, target.representation, target.values.axes)
        if target.measure.weights is not None:
            weights = tensors[target.measure.weights.key]
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise OmniFactoryError("E_MEASURE", "weights must be finite and nonnegative")
    targets = {target.id: target for target in plan.targets}
    for stream in plan.streams:
        values, valid = tensors[stream.values.key], tensors[stream.valid.key]
        shape = [1] * values.ndim
        shape[stream.values.axes.index(stream.event_axis)] = len(valid)
        expanded = valid.reshape(shape).expand_as(values)
        _validate_values(values, expanded, stream.representation, stream.values.axes)
        if stream.target_id is not None:
            mask = tensors[targets[stream.target_id].mask.key]
            supervised = mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim)).expand_as(
                values
            )
            if (supervised & ~expanded).any():
                raise OmniFactoryError(
                    "E_STREAM_TARGET", "supervision cannot include invalid stream events"
                )
        intervals = tensors[stream.intervals_us.key][valid]
        available = tensors[stream.available_at_us.key][valid]
        if (
            (intervals[:, 0] < 0).any()
            or (intervals[:, 1] <= intervals[:, 0]).any()
            or (intervals[1:, 0] < intervals[:-1, 0]).any()
            or (available < intervals[:, 1]).any()
        ):
            raise OmniFactoryError(
                "E_STREAM_TIME",
                "events must be ordered, positive intervals; availability cannot precede their end",
            )


def target_measure(target, tensors):
    mask = tensors[target.mask.key]
    if target.measure.weights is None:
        return float(mask.sum())
    return float(tensors[target.measure.weights.key][mask].double().sum())


def normalization_counts(plan, tensors, *, validate=True):
    if validate:
        validate_supervision(plan, tensors)
    targets = {t.id: t for t in plan.targets}
    # Multiple objectives on one target/group share its denominator, not duplicate it.
    members = {n.id: set() for n in plan.normalizers}
    for objective in plan.objectives:
        if objective.weight:
            members[objective.normalizer].add(objective.target_id)
    return {
        key: sum(target_measure(targets[name], tensors) for name in sorted(ids))
        for key, ids in members.items()
    }


def causal_visibility(plan, tensors, query_stream, context_stream, *, max_pairs=1_000_000):
    """Explicit opt-in visibility mask; never silently applied to a native attention layer."""
    validate_supervision(plan, tensors)
    if not any(
        e.query_stream == query_stream and e.context_stream == context_stream
        for e in plan.alignments
    ):
        raise OmniFactoryError("E_ALIGNMENT", "no declared causal edge")
    streams = {s.id: s for s in plan.streams}
    q, c = streams[query_stream], streams[context_stream]
    starts = tensors[q.intervals_us.key][:, 0]
    available = tensors[c.available_at_us.key]
    if starts.numel() * available.numel() > max_pairs:
        raise OmniFactoryError("E_ALIGNMENT_BUDGET", "dense visibility exceeds max_pairs")
    return (
        (available[None, :] <= starts[:, None])
        & tensors[q.valid.key][:, None]
        & tensors[c.valid.key][None, :]
    )


def discrete_supervision(
    text_labels,
    *,
    codec_labels=None,
    text_weight=1.0,
    codec_weight=1.0,
    codec_vocabularies=(),
    codec_axes=None,
):
    """Declare already-shifted target values. Shifting native inputs remains family-owned."""
    tensors, targets, objectives = {}, [], []
    items = [("text", text_labels, ("batch", "token"), text_weight, ())]
    if codec_labels is not None:
        items.append(
            (
                "codec",
                codec_labels,
                codec_axes or ("symbol",),
                codec_weight,
                tuple(codec_vocabularies),
            )
        )
    for name, values, axes, weight, vocabularies in items:
        if values is None:
            continue
        key = "supervision/" + name
        values = values.detach().clone().long()
        mask = values != -100
        tensors[key + "/values"], tensors[key + "/mask"] = values, mask
        targets.append(
            SupervisionTarget(
                id=name,
                modality="text" if name == "text" else "audio",
                representation=DiscreteRepresentation(
                    vocabulary_sizes=vocabularies,
                    codebook_axis="codebook" if "codebook" in axes else None,
                ),
                values=tensor_ref(key + "/values", values, axes),
                mask=tensor_ref(key + "/mask", mask, axes),
                measure=ReductionMeasure(unit="symbols"),
            )
        )
        objectives.append(
            LossObjective(
                id=name + "_ce",
                target_id=name,
                kind="cross_entropy",
                normalizer="symbols",
                weight=weight,
            )
        )
    plan = SupervisionPlan(
        targets=tuple(targets),
        objectives=tuple(objectives),
        normalizers=(Normalizer(id="symbols", unit="symbols"),) if targets else (),
    )
    validate_supervision(plan, tensors)
    return plan, tensors


def expected_losses(plans_and_counts):
    """Combine persisted contracts for an accumulation window, without loading tensors."""
    specs, counts, units = {}, {}, {}
    for plan, measures in plans_and_counts:
        if set(measures) != {n.id for n in plan.normalizers}:
            raise OmniFactoryError("E_LOSS_COUNTS", "normalization keys differ from plan")
        for normalizer in plan.normalizers:
            key, unit = normalizer.id, normalizer.unit
            value = measures[key]
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise OmniFactoryError("E_LOSS_COUNTS", "invalid normalization measure")
            if key in units and units[key] != unit:
                raise OmniFactoryError("E_LOSS_UNIT", "normalizer unit changed across samples")
            units[key] = unit
            counts[key] = counts.get(key, 0.0) + value
        for objective in plan.objectives:
            spec = (objective.normalizer, objective.weight, objective.kind)
            if objective.id in specs and specs[objective.id] != spec:
                raise OmniFactoryError("E_LOSS_CONTRACT", "objective changed across samples")
            specs[objective.id] = spec
    return specs, counts, units
