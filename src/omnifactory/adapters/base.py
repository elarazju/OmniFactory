"""Safe tensor-tree storage for flat and nested native processor outputs."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from omnifactory.data.supervision import SupervisionPlan
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import SampleCost


def pack_tree(payload):
    tensors = {}

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            tensors[path] = value
            return {"tensor": path}
        if isinstance(value, dict):
            if any(not isinstance(k, str) or not k or "/" in k for k in value):
                raise OmniFactoryError(
                    "E_TENSOR_TREE", "keys must be nonempty strings without slashes"
                )
            return {"dict": {k: visit(v, path + "/" + k if path else k) for k, v in value.items()}}
        if isinstance(value, (list, tuple)):
            return {"list": [visit(v, path + "/" + str(i)) for i, v in enumerate(value)]}
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"value": value}
        raise OmniFactoryError("E_TENSOR_TREE", f"unsupported payload: {type(value)}")

    layout = visit(payload, "")
    return tensors, layout


def unpack_tree(tensors, layout=None):
    if layout is None:
        return tensors
    if "tensor" in layout:
        return tensors[layout["tensor"]]
    if "dict" in layout:
        return {k: unpack_tree(tensors, v) for k, v in layout["dict"].items()}
    if "list" in layout:
        return [unpack_tree(tensors, v) for v in layout["list"]]
    return layout["value"]


def move_tree(value, *, device, dtype):
    """Move nested native tensors, retaining integer/bool index and mask dtypes."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
    if isinstance(value, dict):
        return {k: move_tree(v, device=device, dtype=dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [move_tree(v, device=device, dtype=dtype) for v in value]
    return value


@dataclass
class EncodedExample:
    """Native inputs plus a typed supervision contract; neither labels nor token IDs required."""

    tensors: dict[str, torch.Tensor]
    cost: SampleCost
    prompt_length: int | None = None
    answer: str | None = None
    layout: dict | None = None
    metadata: dict | None = None
    supervision: SupervisionPlan = field(default_factory=SupervisionPlan)

    def validate(self, *, inference=False):
        from omnifactory.adapters.supervision import normalization_counts

        counts = normalization_counts(self.supervision, self.tensors)
        if inference and (
            self.supervision.targets
            or any(s.direction == "output" for s in self.supervision.streams)
        ):
            raise OmniFactoryError("E_TARGET_LEAK", "inference encoding must exclude supervision")
        if not inference and (not counts or not any(v > 0 for v in counts.values())):
            raise OmniFactoryError(
                "E_EMPTY_SUPERVISION", "training requires active measured targets"
            )
        latent_refs = {
            t.values.key: t.values
            for t in self.supervision.targets
            if t.representation.kind == "continuous"
        }
        latent_refs.update(
            {
                s.values.key: s.values
                for s in self.supervision.streams
                if s.representation.kind == "continuous"
            }
        )
        latent_elements = sum(self.tensors[key].numel() for key in latent_refs)
        stream_events = sum(s.valid.shape[0] for s in self.supervision.streams)
        if self.cost.latent_elements < latent_elements or self.cost.stream_events < stream_events:
            raise OmniFactoryError(
                "E_SUPERVISION_COST", "latent/stream allocation is missing from SampleCost"
            )
        return self

    def with_supervision(self, plan, tensors):
        if set(tensors) & set(self.tensors):
            raise OmniFactoryError(
                "E_TENSOR_REFERENCE", "supervision would overwrite native tensors"
            )
        # Freeze the native payload layout before adding target-only tensors to storage.
        if self.layout is None:
            self.tensors, self.layout = pack_tree(self.tensors)
        self.tensors.update(tensors)
        self.supervision = plan
        return self.validate()

    @property
    def normalization_counts(self):
        from omnifactory.adapters.supervision import normalization_counts

        return normalization_counts(self.supervision, self.tensors)

    @property
    def target_tokens(self):
        """Compatibility accessor for discrete adapters only; never inferred from `labels`."""
        if (
            len(self.supervision.normalizers) > 1
            or any(n.unit != "symbols" for n in self.supervision.normalizers)
            or any(t.measure.weights is not None for t in self.supervision.targets)
        ):
            raise OmniFactoryError(
                "E_MEASURE_UNIT", "continuous/stream measures are not token counts"
            )
        return int(sum(self.normalization_counts.values()))

    @property
    def payload(self):
        return unpack_tree(self.tensors, self.layout)


class TrainingAdapter(Protocol):
    """Family-owned native batching/generation; forward returns an explicit LossBundle via .losses."""

    processor: Any

    def encode(self, sample, prediction_id, *, contract_id, inference=False) -> EncodedExample: ...
    def collate(self, rows: list[dict]) -> dict: ...
    def forward(self, model, batch): ...
    def generate(self, model, batch, *, max_new_tokens: int) -> list[str]: ...
    def save_processor(self, path): ...
    def generate_image(self, model, batch, *, num_steps: int): ...
    def generate_speech(
        self, model, batch, *, max_new_tokens: int, max_codec_tokens: int
    ) -> dict: ...
    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None): ...
    @staticmethod
    def to_device(batch, model, config) -> dict: ...
