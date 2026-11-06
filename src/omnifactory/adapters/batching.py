"""Dependency-light task declarations used by plugins, planning and execution audits.

A native decoder batch does not imply batched media encoders, packing or generation.
"""

from dataclasses import dataclass

from omnifactory.errors import OmniFactoryError


@dataclass(frozen=True)
class TaskBatchCapability:
    mode: str = "sequential"
    modalities: tuple[str, ...] = ("text", "image", "audio", "video")
    heterogeneous: bool = False
    supervision: tuple[str, ...] = ("discrete",)
    timed: bool = False
    fallback_reason: str = "native_batch_not_verified"
    media_execution: str = "per_example"
    input_layout: str = "native"
    packing: bool = False
    version: str = "1"

    def validate(self):
        if (
            self.mode not in ("padded", "sequential")
            or not self.modalities
            or len(set(self.modalities)) != len(self.modalities)
            or not set(self.modalities) <= {"text", "image", "audio", "video"}
            or not self.supervision
            or not set(self.supervision) <= {"discrete", "continuous"}
            or self.media_execution not in ("native", "per_example")
            or self.input_layout
            not in ("native", "token_sequence", "mimo_codebooks", "qwen3_tts_pairs")
            or self.packing
            and self.mode != "padded"
            or not self.version
            or (self.mode == "sequential" and (self.heterogeneous or not self.fallback_reason))
        ):
            raise OmniFactoryError("E_BATCH_CAPABILITY", "invalid task batch declaration")
        return self

    def decision(self, costs, *, packing=False, timed=False):
        signature = {m for row in costs for m in row.modalities}
        reason = None
        if packing and not self.packing:
            reason = "packing_not_verified_for_task"
        elif timed:
            reason = "timed_attention_requires_sample_local_execution"
        elif not signature <= set(self.modalities):
            reason = "unsupported_native_modalities:" + ",".join(
                sorted(signature - set(self.modalities))
            )
        elif self.mode == "sequential":
            reason = self.fallback_reason
        elif not self.heterogeneous and len({tuple(sorted(r.modalities)) for r in costs}) > 1:
            reason = "heterogeneous_batch_not_verified"
        return ("sequential", reason) if reason else ("varlen" if packing else "padded", None)

    def grouping_key(self, row):
        mode, reason = self.decision([row])
        return (
            mode,
            reason,
            () if self.heterogeneous and reason is None else tuple(sorted(row.modalities)),
        )
