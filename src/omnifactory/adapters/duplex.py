"""Event-time attention wired into native MiniCPM/Qwen3 eager or SDPA attention.

Bindings refer to processor-expanded token slots, including projected media slots.
They are not codec tokens or inferred word-level alignment. Cached/live generation
needs a separate scheduler; this module trains bounded teacher-forced sequences.
"""

import hashlib
import json
from typing import Annotated, Literal

import torch
from pydantic import Field, model_validator

from omnifactory.adapters.base import pack_tree
from omnifactory.adapters.supervision import tensor_ref
from omnifactory.data.schema import Identifier, Record
from omnifactory.data.supervision import (
    AlignedStream,
    CausalAlignment,
    DiscreteRepresentation,
    SupervisionPlan,
    Timeline,
)
from omnifactory.errors import OmniFactoryError


class TokenEvent(Record):
    id: Identifier
    start_token: Annotated[int, Field(ge=0)]
    end_token: Annotated[int, Field(gt=0)]
    start_us: Annotated[int, Field(ge=0)]
    end_us: Annotated[int, Field(gt=0)]
    available_us: Annotated[int, Field(ge=0)]
    direction: Literal["input", "output", "condition"]
    modality: Literal["text", "audio", "image", "video"]

    @model_validator(mode="after")
    def ordered(self):
        if (
            self.end_token <= self.start_token
            or self.end_us <= self.start_us
            or self.available_us < self.end_us
        ):
            raise ValueError("E_DUPLEX_EVENT: invalid token/time interval or availability")
        return self


class TokenSchedule(Record):
    sample_id: Identifier
    prediction_id: Identifier
    input_ids_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    events: tuple[TokenEvent, ...]


def token_fingerprint(ids):
    return hashlib.sha256(ids.detach().long().cpu().contiguous().numpy().tobytes()).hexdigest()


def bind_duplex(encoded, schedule: TokenSchedule, *, max_pairs=1_000_000):
    """Bind explicit event timing to exact native expanded inputs; no timestamp guessing."""
    payload = encoded.payload
    ids = payload["input_ids"]
    if ids.shape[0] != 1 or token_fingerprint(ids) != schedule.input_ids_sha256:
        raise OmniFactoryError(
            "E_DUPLEX_FINGERPRINT", "timing must match exact processor-expanded input IDs"
        )
    if (encoded.cost.sample_id, encoded.cost.prediction_id) != (
        schedule.sample_id,
        schedule.prediction_id,
    ):
        raise OmniFactoryError("E_DUPLEX_ID", "schedule belongs to another prediction")
    n = ids.shape[1]
    if n * n > max_pairs:
        raise OmniFactoryError("E_ALIGNMENT_BUDGET", "native dense attention exceeds max_pairs")
    events = sorted(schedule.events, key=lambda e: e.start_token)
    if (
        not events
        or len({e.id for e in events}) != len(events)
        or events[0].start_token != 0
        or events[-1].end_token != n
        or any(a.end_token != b.start_token for a, b in zip(events, events[1:]))
    ):
        raise OmniFactoryError(
            "E_DUPLEX_COVERAGE", "unique events must partition every native token slot"
        )
    query, available, groups, output = (torch.empty(n, dtype=torch.long) for _ in range(4))
    streams, extra = [], {}
    for i, event in enumerate(events):
        span = slice(event.start_token, event.end_token)
        query[span] = event.start_us if event.direction == "output" else event.available_us
        available[span] = event.available_us
        groups[span] = i
        output[span] = int(event.direction == "output")
        key = f"duplex_stream/{i}"
        tensors = {
            key + "/slots": ids[:, span].clone(),
            key + "/interval": torch.tensor([[event.start_us, event.end_us]]),
            key + "/available": torch.tensor([event.available_us]),
            key + "/valid": torch.tensor([True]),
        }
        extra.update(tensors)
        streams.append(
            AlignedStream(
                id=event.id,
                timeline_id="duplex",
                modality=event.modality,
                direction=event.direction,
                representation=DiscreteRepresentation(),
                event_axis="event",
                values=tensor_ref(
                    key + "/slots", tensors[key + "/slots"], ("event", "native_slot")
                ),
                intervals_us=tensor_ref(
                    key + "/interval", tensors[key + "/interval"], ("event", "boundary")
                ),
                available_at_us=tensor_ref(
                    key + "/available", tensors[key + "/available"], ("event",)
                ),
                valid=tensor_ref(key + "/valid", tensors[key + "/valid"], ("event",)),
            )
        )
    if any(len(row) for row in payload.get("image_bound", [])):
        raise OmniFactoryError(
            "E_DUPLEX_MODALITY",
            "timed attention currently supports audio/text; image/video events need encoder-boundary validation",
        )
    bounds = payload.get("audio_bounds", [[]])[0]
    durations = (encoded.metadata or {}).get("audio_durations_us", [])
    if len(bounds) != len(durations):
        raise OmniFactoryError("E_DUPLEX_AUDIO", "missing measured audio durations")
    for (start, end), duration in zip(bounds, durations):
        start, end = int(start), int(end)
        event = events[int(groups[start])]
        if (
            (groups[start:end] != groups[start]).any()
            or event.modality != "audio"
            or event.direction != "input"
            or event.end_us - event.start_us < duration
        ):
            raise OmniFactoryError(
                "E_DUPLEX_AUDIO",
                "each complete encoded audio clip must belong to one available input event spanning its measured duration",
            )
    labels = payload.get("labels")
    if labels is not None:
        active = labels[0, 1:] != -100
        if ((groups[:-1] != groups[1:]) & active).any() or ((output[1:] == 0) & active).any():
            raise OmniFactoryError(
                "E_DUPLEX_SHIFT",
                "each prediction query and shifted target must share an output event; include its preceding prompt token",
            )
    payload.update(
        duplex_query_us=query[None],
        duplex_available_us=available[None],
        duplex_group=groups[None],
        duplex_output=output.bool()[None],
    )
    native, layout = pack_tree(payload)
    # Keep existing target-only arrays out of native forward arguments.
    for key, value in encoded.tensors.items():
        if key not in native:
            native[key] = value
    native.update(extra)
    raw = encoded.supervision.model_dump()
    raw["timelines"] = [Timeline(id="duplex").model_dump()]
    raw["streams"] = [s.model_dump() for s in streams]
    raw["alignments"] = [
        CausalAlignment(query_stream=q.id, context_stream=c.id).model_dump()
        for q in streams
        if q.direction == "output"
        for c in streams
        if c.id != q.id
    ]
    encoded.tensors, encoded.layout = native, layout
    encoded.supervision = SupervisionPlan.model_validate(raw)
    encoded.cost = encoded.cost.model_copy(update={"stream_events": len(events)})
    encoded.metadata = {
        **(encoded.metadata or {}),
        "duplex": schedule.model_dump(mode="json"),
        "teacher_forcing": "same_output_event_serial_prefix",
        "stream_values": "native expanded tokenizer slots; audio slots receive projected continuous features at forward",
    }
    return encoded.validate()


def bind_from_manifest(encoded, path, *, max_pairs):
    schedules = [TokenSchedule.model_validate(row) for row in json.loads(path.read_text())]
    keys = [(s.sample_id, s.prediction_id) for s in schedules]
    if len(set(keys)) != len(keys):
        raise OmniFactoryError("E_DUPLEX_ID", "duplicate schedule predictions")
    key = (encoded.cost.sample_id, encoded.cost.prediction_id)
    if key not in keys:
        raise OmniFactoryError("E_DUPLEX_ID", f"missing timing for {key}")
    return bind_duplex(encoded, schedules[keys.index(key)], max_pairs=max_pairs)


def native_attention(batch, *, dtype, max_pairs=1_000_000):
    """Return additive native attention and visibility-derived rotary positions.

    Available context is causal in serialization order. Within an output event,
    previous teacher-forced tokens are visible; later tokens never are. Computing
    every intermediate state with this mask prevents indirect future leakage.
    """
    valid = batch["attention_mask"].bool()
    query, available = batch["duplex_query_us"], batch["duplex_available_us"]
    group, output = batch["duplex_group"], batch["duplex_output"]
    b, n = valid.shape
    if b * n * n > max_pairs:
        raise OmniFactoryError("E_ALIGNMENT_BUDGET", "native dense attention exceeds max_pairs")
    serial = torch.ones(n, n, dtype=torch.bool, device=valid.device).tril()
    teacher = (group[:, :, None] == group[:, None, :]) & output[:, :, None] & output[:, None, :]
    visible = ((available[:, None, :] <= query[:, :, None]) | teacher) & serial
    visible &= valid[:, :, None] & valid[:, None, :]
    if (valid & ~visible.any(-1)).any():
        raise OmniFactoryError("E_DUPLEX_CONTEXT", "a valid query has no available context")
    # Padding self-edges avoid undefined all-masked rows; no valid query sees them.
    visible |= torch.eye(n, dtype=torch.bool, device=valid.device)[None] & ~valid[:, :, None]
    bias = torch.zeros(b, 1, n, n, device=valid.device, dtype=dtype).masked_fill(
        ~visible[:, None], torch.finfo(dtype).min
    )
    positions = (visible.sum(-1) - 1).clamp_min(0).masked_fill(~valid, 0)
    return bias, positions
