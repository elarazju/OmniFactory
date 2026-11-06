"""Bounded-window, reproducible multimodal batch planning.

Costs must be supplied by a model encoder/profiler. Raw media lengths are not
silently converted into pretend model token counts. No tensor packing is performed.
"""

import random
from collections.abc import Iterable, Iterator
from itertools import islice
from math import ceil
from typing import Annotated, Literal

from pydantic import Field, model_validator

from omnifactory.data.schema import Identifier, Record
from omnifactory.errors import OmniFactoryError

Count = Annotated[int, Field(ge=0, strict=True)]
Positive = Annotated[int, Field(gt=0, strict=True)]


class SampleCost(Record):
    sample_id: Identifier
    prediction_id: Identifier
    # Must identify model revision, processor, recipe and trainable policy.
    contract_id: Identifier
    modalities: Annotated[
        tuple[Literal["text", "image", "audio", "video"], ...], Field(min_length=1)
    ]
    llm_tokens: Count = 0
    latent_elements: Count = 0
    stream_events: Count = 0
    vision_patches: Count = 0
    audio_frames: Count = 0
    audio_sequences: Count = 0
    audio_sequence_length: Count = 0
    codec_frames: Count = 0
    decoded_media_bytes: Count = 0

    @model_validator(mode="after")
    def unique_modalities(self):
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be unique")
        if self.audio_sequences or self.audio_sequence_length:
            if (
                not self.audio_sequences
                or not self.audio_sequence_length
                or self.audio_frames != self.audio_sequences * self.audio_sequence_length
            ):
                raise ValueError("audio_frames must equal audio_sequences * audio_sequence_length")
        return self


class ResourceBudget(Record):
    max_examples: Positive = 8
    max_llm_tokens: Positive
    max_vision_patches: Positive
    max_audio_frames: Positive
    max_codec_frames: Positive
    max_decoded_media_bytes: Positive
    max_gpu_bytes: Positive | None = None
    max_latent_elements: Positive | None = None
    max_stream_events: Positive | None = None


class MemoryProfile(Record):
    profile_id: Identifier
    contract_id: Identifier
    hardware: Identifier
    provenance: Literal["synthetic", "calibrated"]
    fixed_bytes: Count
    workspace_bytes: Count = 0
    bytes_per_llm_token: Count
    bytes_per_vision_patch: Count
    bytes_per_audio_frame: Count
    bytes_per_codec_frame: Count
    bytes_per_attention_pair: Count = 0
    bytes_per_latent_element: Count | None = None
    bytes_per_stream_event: Count | None = None
    safety_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.15
    max_calibrated_costs: dict[str, Count] = {}
    calibrated_modalities: tuple[tuple[str, ...], ...] = ()

    @model_validator(mode="after")
    def valid_calibration_caps(self):
        allowed = {
            "examples",
            "allocated_llm_tokens",
            "allocated_vision_patches",
            "allocated_audio_frames",
            "allocated_codec_frames",
            "allocated_attention_pairs",
            "allocated_latent_elements",
            "allocated_stream_events",
        }
        if not self.max_calibrated_costs.keys() <= allowed:
            raise ValueError("unknown calibrated cost dimension")
        return self


class BatchCost(Record):
    examples: int
    useful_llm_tokens: int
    useful_vision_patches: int = 0
    useful_audio_frames: int = 0
    useful_codec_frames: int = 0
    useful_latent_elements: int = 0
    useful_stream_events: int = 0
    allocated_llm_tokens: int
    allocated_vision_patches: int
    allocated_audio_frames: int
    allocated_codec_frames: int
    decoded_media_bytes: int
    allocated_latent_elements: int = 0
    allocated_stream_events: int = 0
    allocated_attention_pairs: int = 0
    estimated_gpu_bytes: int | None


class BatchPlan(Record):
    contract_id: str
    example_ids: tuple[tuple[str, str], ...]
    execution: Literal["padded", "varlen", "sequential"] = "padded"
    fallback_reason: str | None = None
    media_execution: str | None = None
    cost: BatchCost
    memory_profile_id: str | None
    memory_provenance: str | None
    # Estimates are not an OOM guarantee, including when calibrated.
    gpu_fit_verified: Literal[False] = False


def estimate_batch_cost(
    rows: list[SampleCost], *, memory_profile=None, packing=False, capability=None, timed=False
) -> BatchCost:
    """Pure conservative layout estimate; planning gates remain in ModalBatchPlanner."""
    if not rows:
        raise OmniFactoryError("E_EMPTY_BATCH", "cannot cost an empty batch")
    contracts = {r.contract_id for r in rows}
    if len(contracts) != 1:
        raise OmniFactoryError("E_CONTRACT", "cannot mix different model/recipe contracts")
    if memory_profile is not None and contracts != {memory_profile.contract_id}:
        raise OmniFactoryError("E_MEMORY_PROFILE", "profile does not match sample contract")
    count = len(rows)
    allocated = {
        name: count * max(getattr(r, name) for r in rows)
        for name in (
            "llm_tokens",
            "vision_patches",
            "audio_frames",
            "codec_frames",
            "latent_elements",
            "stream_events",
        )
    }
    if all(r.audio_frames == 0 or r.audio_sequences > 0 for r in rows):
        allocated["audio_frames"] = sum(r.audio_sequences for r in rows) * max(
            r.audio_sequence_length for r in rows
        )
    packed = (
        packing
        and not any("video" in r.modalities for r in rows)
        and (
            capability is None
            or capability.decision(rows, packing=True, timed=timed)[0] == "varlen"
        )
    )
    attention_pairs = count * max(r.llm_tokens for r in rows) ** 2
    if packed:
        allocated["llm_tokens"] = sum(r.llm_tokens for r in rows)
        attention_pairs = sum(r.llm_tokens**2 for r in rows)
    gpu = None
    if memory_profile is not None:
        p = memory_profile
        signature = tuple(sorted({m for r in rows for m in r.modalities}))
        if p.calibrated_modalities and signature not in p.calibrated_modalities:
            raise OmniFactoryError("E_MEMORY_PROFILE", f"uncalibrated modalities: {signature}")
        for dimension, coefficient in (
            ("latent_elements", p.bytes_per_latent_element),
            ("stream_events", p.bytes_per_stream_event),
        ):
            if allocated[dimension] and coefficient is None:
                raise OmniFactoryError("E_MEMORY_PROFILE", f"missing {dimension} calibration")
        estimate = (
            allocated["latent_elements"] * (p.bytes_per_latent_element or 0)
            + allocated["stream_events"] * (p.bytes_per_stream_event or 0)
            + p.fixed_bytes
            + p.workspace_bytes
            + allocated["llm_tokens"] * p.bytes_per_llm_token
            + allocated["vision_patches"] * p.bytes_per_vision_patch
            + allocated["audio_frames"] * p.bytes_per_audio_frame
            + allocated["codec_frames"] * p.bytes_per_codec_frame
            + attention_pairs * p.bytes_per_attention_pair
        )
        gpu = ceil(estimate * (1 + p.safety_fraction))
    return BatchCost(
        examples=count,
        useful_llm_tokens=sum(r.llm_tokens for r in rows),
        **{
            "useful_" + k: sum(getattr(r, k) for r in rows)
            for k in (
                "vision_patches",
                "audio_frames",
                "codec_frames",
                "latent_elements",
                "stream_events",
            )
        },
        allocated_llm_tokens=allocated["llm_tokens"],
        allocated_vision_patches=allocated["vision_patches"],
        allocated_audio_frames=allocated["audio_frames"],
        allocated_codec_frames=allocated["codec_frames"],
        allocated_latent_elements=allocated["latent_elements"],
        allocated_stream_events=allocated["stream_events"],
        allocated_attention_pairs=attention_pairs,
        decoded_media_bytes=sum(r.decoded_media_bytes for r in rows),
        estimated_gpu_bytes=gpu,
    )


class ModalBatchPlanner:
    def __init__(
        self,
        budget: ResourceBudget,
        *,
        memory_profile: MemoryProfile | None = None,
        window_size: int = 128,
        seed: int = 0,
        packing: bool = False,
        capability=None,
        timed: bool = False,
    ):
        if window_size < 1:
            raise OmniFactoryError("E_WINDOW", "window_size must be positive")
        if budget.max_gpu_bytes is not None and memory_profile is None:
            raise OmniFactoryError(
                "E_MEMORY_PROFILE", "GPU budgeting requires an explicit memory profile"
            )
        self.budget, self.profile = budget, memory_profile
        self.window_size, self.seed = window_size, seed
        self.packing = packing
        self.capability, self.timed = capability, timed

    def cost(self, rows: list[SampleCost]) -> BatchCost:
        return estimate_batch_cost(
            rows,
            memory_profile=self.profile,
            packing=self.packing,
            capability=self.capability,
            timed=self.timed,
        )

    def exceeded(self, cost: BatchCost) -> list[str]:
        checks = {
            "examples": self.budget.max_examples,
            "allocated_llm_tokens": self.budget.max_llm_tokens,
            "allocated_vision_patches": self.budget.max_vision_patches,
            "allocated_audio_frames": self.budget.max_audio_frames,
            "allocated_codec_frames": self.budget.max_codec_frames,
            "decoded_media_bytes": self.budget.max_decoded_media_bytes,
        }
        failed = [name for name, cap in checks.items() if getattr(cost, name) > cap]
        for name, cap in (
            ("allocated_latent_elements", self.budget.max_latent_elements),
            ("allocated_stream_events", self.budget.max_stream_events),
        ):
            if getattr(cost, name) and (cap is None or getattr(cost, name) > cap):
                failed.append(name + ("_budget_required" if cap is None else ""))
        if (
            self.budget.max_gpu_bytes is not None
            and cost.estimated_gpu_bytes > self.budget.max_gpu_bytes
        ):
            failed.append("estimated_gpu_bytes")
        if self.profile is not None:
            failed.extend(
                "uncalibrated_" + name
                for name, cap in self.profile.max_calibrated_costs.items()
                if getattr(cost, name) > cap
            )
        return failed

    def plan(self, samples: Iterable[SampleCost]) -> Iterator[BatchPlan]:
        rng = random.Random(self.seed)
        iterator = iter(samples)
        while window := list(islice(iterator, self.window_size)):
            rng.shuffle(window)
            window.sort(key=lambda s: (s.contract_id, tuple(sorted(s.modalities)), -s.llm_tokens))
            batches: list[list[SampleCost]] = []
            for row in window:
                failures = self.exceeded(self.cost([row]))
                if failures:
                    raise OmniFactoryError(
                        "E_SAMPLE_OVER_BUDGET",
                        f"{row.sample_id}/{row.prediction_id}: {', '.join(failures)}; adjust budget or explicitly preprocess this sample",
                    )
                for batch in batches:
                    if batch[0].contract_id != row.contract_id or (
                        self.capability.grouping_key(batch[0]) != self.capability.grouping_key(row)
                        if self.capability is not None
                        else set(batch[0].modalities) != set(row.modalities)
                    ):
                        continue
                    if not self.exceeded(self.cost([*batch, row])):
                        batch.append(row)
                        break
                else:
                    batches.append([row])
            rng.shuffle(batches)
            for batch in batches:
                execution, reason = (
                    self.capability.decision(batch, packing=self.packing, timed=self.timed)
                    if self.capability is not None
                    else (
                        "varlen"
                        if self.packing and not any("video" in r.modalities for r in batch)
                        else "padded",
                        None,
                    )
                )
                yield BatchPlan(
                    contract_id=batch[0].contract_id,
                    fallback_reason=reason,
                    media_execution=self.capability.media_execution if self.capability else None,
                    example_ids=tuple((r.sample_id, r.prediction_id) for r in batch),
                    execution=execution,
                    cost=self.cost(batch),
                    memory_profile_id=self.profile.profile_id if self.profile else None,
                    memory_provenance=self.profile.provenance if self.profile else None,
                )
