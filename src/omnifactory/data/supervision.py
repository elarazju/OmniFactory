"""Serializable, framework-independent contracts for encoded supervision and aligned streams.

Representations describe values. Streams add time/availability to either representation.
Tensor references address the flat safetensors store, never arbitrary Python objects.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from omnifactory.data.schema import Identifier, Modality, Record

Positive = Annotated[int, Field(gt=0, strict=True)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
MeasureUnit = Literal["symbols", "elements", "frames", "seconds", "examples"]


class TensorRef(Record):
    key: Identifier
    axes: tuple[Identifier, ...]
    shape: tuple[Annotated[int, Field(ge=0, strict=True)], ...]
    dtype: Literal["bool", "int32", "int64", "float16", "bfloat16", "float32", "float64"]

    @model_validator(mode="after")
    def dimensions(self):
        if len(self.axes) != len(self.shape) or len(set(self.axes)) != len(self.axes):
            raise ValueError("E_SUPERVISION_AXES: unique named axes must match tensor rank")
        return self


class DiscreteRepresentation(Record):
    kind: Literal["discrete"] = "discrete"
    vocabulary_sizes: tuple[Positive, ...] = ()
    codebook_axis: str | None = None
    ignore_index: int = -100

    @model_validator(mode="after")
    def vocabularies(self):
        if len(self.vocabulary_sizes) > 1 and self.codebook_axis is None:
            raise ValueError("E_CODEBOOK_AXIS: multiple vocabularies require a codebook axis")
        return self


class ContinuousRepresentation(Record):
    kind: Literal["continuous"] = "continuous"
    space: Identifier  # e.g. codec latent, image VAE latent, velocity; no implied decoder.
    encoder_id: Identifier | None = None


Representation = Annotated[
    DiscreteRepresentation | ContinuousRepresentation, Field(discriminator="kind")
]


class ReductionMeasure(Record):
    unit: MeasureUnit
    # Each True mask cell is one unit. Remaining value axes are feature dimensions.
    feature_reduction: Literal["sum", "mean"] = "sum"
    # Optional explicit nonnegative per-unit weights, e.g. event duration in seconds.
    weights: TensorRef | None = None


class SupervisionTarget(Record):
    id: Identifier
    modality: Modality
    representation: Representation
    values: TensorRef
    mask: TensorRef
    measure: ReductionMeasure

    @model_validator(mode="after")
    def layout(self):
        if self.mask.dtype != "bool":
            raise ValueError("E_SUPERVISION_MASK: mask must be boolean")
        n = len(self.mask.axes)
        if self.values.axes[:n] != self.mask.axes or self.values.shape[:n] != self.mask.shape:
            raise ValueError("E_SUPERVISION_MASK: mask axes must be an exact prefix of value axes")
        if self.representation.kind == "discrete":
            if self.values.dtype not in ("int32", "int64") or self.mask.shape != self.values.shape:
                raise ValueError("E_DISCRETE_TARGET: integer values require a mask per symbol")
            if self.measure.unit != "symbols":
                raise ValueError("E_DISCRETE_TARGET: discrete supervision is measured in symbols")
            axis = self.representation.codebook_axis
            if axis is not None:
                if axis not in self.values.axes:
                    raise ValueError("E_CODEBOOK_AXIS: unknown axis")
                sizes = self.representation.vocabulary_sizes
                if len(sizes) != self.values.shape[self.values.axes.index(axis)]:
                    raise ValueError("E_CODEBOOK_AXIS: vocabularies must match codebook count")
        elif self.values.dtype not in ("float16", "bfloat16", "float32", "float64"):
            raise ValueError("E_CONTINUOUS_TARGET: continuous values must be floating point")
        if self.measure.unit == "seconds" and self.measure.weights is None:
            raise ValueError("E_MEASURE: seconds require explicit per-unit durations")
        if self.representation.kind == "continuous":
            if self.measure.unit == "symbols":
                raise ValueError("E_MEASURE: continuous supervision cannot count discrete symbols")
            if self.measure.unit == "elements" and self.values.shape != self.mask.shape:
                raise ValueError("E_MEASURE: element reduction requires an element-level mask")
            if any(size == 0 for size in self.values.shape[n:]):
                raise ValueError("E_MEASURE: feature axes cannot be empty")
        weights = self.measure.weights
        if weights is not None and (
            weights.axes != self.mask.axes
            or weights.shape != self.mask.shape
            or weights.dtype not in ("float16", "bfloat16", "float32", "float64")
        ):
            raise ValueError("E_MEASURE: weights must match mask axes and shape")
        return self


class Normalizer(Record):
    id: Identifier
    unit: MeasureUnit


class LossObjective(Record):
    id: Identifier
    target_id: Identifier
    kind: Literal["cross_entropy", "mse", "flow_matching", "diffusion", "custom"]
    normalizer: Identifier
    weight: Nonnegative = 1.0


class Timeline(Record):
    id: Identifier
    unit: Literal["microseconds"] = "microseconds"
    origin: Literal["sample"] = "sample"


class AlignedStream(Record):
    id: Identifier
    timeline_id: Identifier
    modality: Modality
    direction: Literal["input", "output", "condition"]
    representation: Representation
    values: TensorRef
    event_axis: Identifier
    intervals_us: TensorRef  # [event, boundary], half-open [start,end).
    available_at_us: TensorRef  # [event], no earlier than event end.
    valid: TensorRef  # [event], excludes padding from alignment/attention.
    target_id: str | None = None

    @model_validator(mode="after")
    def event_layout(self):
        if self.event_axis not in self.values.axes:
            raise ValueError("E_STREAM_AXIS: event axis absent from values")
        n = self.values.shape[self.values.axes.index(self.event_axis)]
        event = (self.event_axis,)
        if (
            self.intervals_us.shape != (n, 2)
            or self.intervals_us.axes != (*event, "boundary")
            or self.intervals_us.dtype != "int64"
        ):
            raise ValueError("E_STREAM_TIME: intervals require int64 [event,boundary] shape [N,2]")
        for ref, dtype in ((self.available_at_us, "int64"), (self.valid, "bool")):
            if ref.axes != event or ref.shape != (n,) or ref.dtype != dtype:
                raise ValueError("E_STREAM_TIME: availability/validity must match event axis")
        if self.direction != "output" and self.target_id is not None:
            raise ValueError("E_STREAM_TARGET: only output streams can bind supervision targets")
        # Reuse representation and range layout validation without a tensor dependency.
        kind = self.representation.kind
        if kind == "discrete" and self.values.dtype not in ("int32", "int64"):
            raise ValueError("E_STREAM_VALUES: discrete stream requires integers")
        if kind == "continuous" and self.values.dtype not in (
            "float16",
            "bfloat16",
            "float32",
            "float64",
        ):
            raise ValueError("E_STREAM_VALUES: continuous stream requires floating point")
        if kind == "discrete" and self.representation.codebook_axis is not None:
            axis = self.representation.codebook_axis
            if (
                axis not in self.values.axes
                or len(self.representation.vocabulary_sizes)
                != self.values.shape[self.values.axes.index(axis)]
            ):
                raise ValueError("E_CODEBOOK_AXIS: stream codebooks differ from declared axis")
        return self


class CausalAlignment(Record):
    query_stream: Identifier
    context_stream: Identifier
    mode: Literal["available_before_start"] = "available_before_start"
    # No implicit lookahead. A model-specific noncausal objective needs a separate contract.


class SupervisionPlan(Record):
    version: Literal["1"] = "1"
    targets: tuple[SupervisionTarget, ...] = ()
    objectives: tuple[LossObjective, ...] = ()
    normalizers: tuple[Normalizer, ...] = ()
    timelines: tuple[Timeline, ...] = ()
    streams: tuple[AlignedStream, ...] = ()
    alignments: tuple[CausalAlignment, ...] = ()

    @model_validator(mode="after")
    def references(self):
        for items in (
            self.targets,
            self.objectives,
            self.normalizers,
            self.timelines,
            self.streams,
        ):
            if len({item.id for item in items}) != len(items):
                raise ValueError("E_SUPERVISION_ID: duplicate IDs")
        targets = {t.id: t for t in self.targets}
        normalizers = {n.id: n for n in self.normalizers}
        used = set()
        for objective in self.objectives:
            target = targets.get(objective.target_id)
            group = normalizers.get(objective.normalizer)
            if target is None or group is None:
                raise ValueError("E_LOSS_REFERENCE: unknown target/normalizer")
            if target.measure.unit != group.unit:
                raise ValueError("E_LOSS_UNIT: unlike units cannot share a normalizer")
            if objective.kind == "cross_entropy" and target.representation.kind != "discrete":
                raise ValueError("E_LOSS_KIND: cross entropy requires discrete targets")
            if (
                objective.kind in ("mse", "flow_matching", "diffusion")
                and target.representation.kind != "continuous"
            ):
                raise ValueError("E_LOSS_KIND: continuous objective requires continuous targets")
            used.add(target.id)
        if used != set(targets) or {o.normalizer for o in self.objectives} != set(normalizers):
            raise ValueError("E_LOSS_REFERENCE: unused targets/normalizers")
        timelines = {t.id for t in self.timelines}
        streams = {s.id: s for s in self.streams}
        for stream in self.streams:
            if stream.timeline_id not in timelines:
                raise ValueError("E_STREAM_TIMELINE: unknown timeline")
            if stream.target_id is not None:
                target = targets.get(stream.target_id)
                if (
                    target is None
                    or target.values != stream.values
                    or target.representation != stream.representation
                    or target.modality != stream.modality
                ):
                    raise ValueError(
                        "E_STREAM_TARGET: stream and target representations must agree"
                    )
        pairs = set()
        for edge in self.alignments:
            q, c = streams.get(edge.query_stream), streams.get(edge.context_stream)
            if q is None or c is None or q.timeline_id != c.timeline_id:
                raise ValueError("E_ALIGNMENT: streams must exist on the same timeline")
            pair = (edge.query_stream, edge.context_stream)
            if pair in pairs:
                raise ValueError("E_ALIGNMENT: duplicate edge")
            pairs.add(pair)
        return self
