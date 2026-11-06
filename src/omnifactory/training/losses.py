"""Differentiable loss sums and explicit per-group measures for exact accumulation."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from omnifactory.adapters.supervision import normalization_counts, validate_supervision
from omnifactory.errors import OmniFactoryError


@dataclass(frozen=True)
class LossTerm:
    id: str
    numerator: torch.Tensor
    normalizer: str
    weight: float = 1.0
    kind: str = "custom"


@dataclass
class LossBundle:
    terms: tuple[LossTerm, ...]
    counts: dict[str, float]
    units: dict[str, str]

    def __post_init__(self):
        if not self.terms or len({t.id for t in self.terms}) != len(self.terms):
            raise OmniFactoryError("E_LOSS_TERMS", "nonempty, unique loss terms required")
        if set(self.counts) != set(self.units):
            raise OmniFactoryError("E_LOSS_COUNTS", "counts and units differ")
        for key, value in self.counts.items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise OmniFactoryError("E_LOSS_COUNTS", f"invalid denominator: {key}")
        for term in self.terms:
            if (
                term.normalizer not in self.counts
                or not math.isfinite(term.weight)
                or term.weight < 0
            ):
                raise OmniFactoryError("E_LOSS_TERM", "invalid normalizer or weight")
            if term.numerator.ndim != 0 or not torch.isfinite(term.numerator):
                raise OmniFactoryError("E_LOSS_NUMERATOR", "loss sums must be finite scalars")
            if (
                term.weight
                and not self.counts[term.normalizer]
                and float(term.numerator.detach()) != 0
            ):
                raise OmniFactoryError("E_LOSS_COUNTS", "nonzero active numerator has zero measure")

    def validate_expected(self, expected):
        specs, counts, units = expected
        actual = {t.id: (t.normalizer, t.weight, t.kind) for t in self.terms}
        if actual != specs or units != self.units or set(counts) != set(self.counts):
            raise OmniFactoryError(
                "E_LOSS_CONTRACT", "forward loss terms differ from encoded contract"
            )
        if any(
            not math.isclose(counts[k], self.counts[k], rel_tol=1e-7, abs_tol=1e-9) for k in counts
        ):
            raise OmniFactoryError(
                "E_LOSS_COUNTS", "forward normalization differs from encoded targets"
            )

    def scaled_loss(self, window_counts=None):
        counts = self.counts if window_counts is None else window_counts
        values = []
        for term in self.terms:
            if not term.weight or not self.counts[term.normalizer]:
                continue
            denominator = counts.get(term.normalizer, 0)
            local = self.counts[term.normalizer]
            if (
                not math.isfinite(denominator)
                or denominator <= 0
                or (
                    denominator < local
                    and not math.isclose(denominator, local, rel_tol=1e-7, abs_tol=1e-9)
                )
            ):
                raise OmniFactoryError(
                    "E_LOSS_COUNTS", "window measure is smaller than local measure"
                )
            values.append(term.numerator * term.weight / denominator)
        if not values:
            # A microbatch may have only masked/disabled targets; another may be active.
            return self.terms[0].numerator * 0
        return sum(values)

    @property
    def loss(self):
        return self.scaled_loss()

    @classmethod
    def combine(cls, bundles):
        terms, counts, units = {}, {}, {}
        for bundle in bundles:
            for key, value in bundle.counts.items():
                if key in units and units[key] != bundle.units[key]:
                    raise OmniFactoryError("E_LOSS_UNIT", "incompatible normalization units")
                units[key] = bundle.units[key]
                counts[key] = counts.get(key, 0) + value
            for term in bundle.terms:
                if term.id in terms:
                    old = terms[term.id]
                    if (old.normalizer, old.kind, old.weight) != (
                        term.normalizer,
                        term.kind,
                        term.weight,
                    ):
                        raise OmniFactoryError("E_LOSS_CONTRACT", "incompatible loss definitions")
                    term = LossTerm(
                        term.id,
                        old.numerator + term.numerator,
                        term.normalizer,
                        term.weight,
                        term.kind,
                    )
                terms[term.id] = term
        return cls(tuple(terms.values()), counts, units)

    def detached(self):
        return LossBundle(
            tuple(
                LossTerm(t.id, t.numerator.detach().double().cpu(), t.normalizer, t.weight, t.kind)
                for t in self.terms
            ),
            dict(self.counts),
            dict(self.units),
        )

    def report(self):
        return {
            "loss": float(self.loss.detach()),
            "normalization_counts": dict(self.counts),
            "normalization_units": dict(self.units),
            "loss_terms": {
                t.id: {
                    "numerator": float(t.numerator.detach()),
                    "normalizer": t.normalizer,
                    "kind": t.kind,
                    "weight": t.weight,
                }
                for t in self.terms
            },
        }


def symbol_losses(
    text_sum, text_count, *, codec_sum=None, codec_count=0, text_weight=1.0, codec_weight=1.0
):
    terms = [LossTerm("text_ce", text_sum, "symbols", text_weight, "cross_entropy")]
    count = text_count if text_weight else 0
    if codec_sum is not None:
        terms.append(LossTerm("codec_ce", codec_sum, "symbols", codec_weight, "cross_entropy"))
        if codec_weight:
            count += codec_count
    return LossBundle(tuple(terms), {"symbols": float(count)}, {"symbols": "symbols"})


def attach_losses(output, bundle):
    output.losses = bundle
    output.loss = bundle.loss
    return output


def supervised_losses(plan, tensors, predictions):
    """Reference CE/MSE reducers with named-axis masks and explicit normalization.

    Flow/diffusion adapters must construct their noisy inputs, time sampling and
    parameterization explicitly, then return LossBundle. This helper never guesses them.
    """
    validate_supervision(plan, tensors)
    if set(predictions) != {o.id for o in plan.objectives}:
        raise OmniFactoryError("E_LOSS_PREDICTIONS", "predictions must match objective IDs")
    targets = {t.id: t for t in plan.targets}
    terms = []
    for objective in plan.objectives:
        target = targets[objective.target_id]
        values, mask = tensors[target.values.key], tensors[target.mask.key]
        predicted = predictions[objective.id]
        if objective.kind == "cross_entropy":
            if predicted.shape[:-1] != values.shape:
                raise OmniFactoryError("E_LOSS_SHAPE", "CE needs value axes plus vocabulary")
            representation = target.representation
            sizes = representation.vocabulary_sizes
            if sizes and predicted.shape[-1] != max(sizes):
                raise OmniFactoryError("E_LOSS_SHAPE", "CE vocabulary differs from declared sizes")
            if representation.codebook_axis is None:
                unit_losses = F.cross_entropy(
                    predicted[mask].float(), values[mask].long(), reduction="none"
                )
            else:
                # Dense logits pad the vocabulary axis to the largest codebook.
                # Padded classes must not enter smaller codebooks' softmax.
                axis = target.values.axes.index(representation.codebook_axis)
                per_symbol = predicted.new_zeros(values.shape, dtype=torch.float32)
                for index, size in enumerate(sizes):
                    valid = mask.select(axis, index)
                    logits = predicted.select(axis, index)[..., :size]
                    labels = values.select(axis, index)
                    per_symbol.select(axis, index)[valid] = F.cross_entropy(
                        logits[valid].float(), labels[valid].long(), reduction="none"
                    )
                unit_losses = per_symbol[mask]
        elif objective.kind == "mse":
            if predicted.shape != values.shape:
                raise OmniFactoryError("E_LOSS_SHAPE", "MSE prediction and target shapes differ")
            # Select first: even NaN predictions in padding must not poison gradients.
            unit_losses = (predicted[mask].float() - values[mask].float()).square()
            feature_dims = tuple(range(1, unit_losses.ndim))
            if feature_dims:
                unit_losses = (
                    unit_losses.mean(feature_dims)
                    if target.measure.feature_reduction == "mean"
                    else unit_losses.sum(feature_dims)
                )
        else:
            raise OmniFactoryError(
                "E_LOSS_IMPLEMENTATION",
                "flow/diffusion/custom losses need an explicit model-specific implementation",
            )
        weights = (
            tensors[target.measure.weights.key][mask] if target.measure.weights is not None else 1
        )
        # Select before reduction; padding contributes neither numerator nor denominator.
        numerator = (unit_losses * weights).sum()
        terms.append(
            LossTerm(
                objective.id, numerator, objective.normalizer, objective.weight, objective.kind
            )
        )
    return LossBundle(
        tuple(terms),
        normalization_counts(plan, tensors, validate=False),
        {n.id: n.unit for n in plan.normalizers},
    )
