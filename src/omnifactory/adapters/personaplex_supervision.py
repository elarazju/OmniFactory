"""Physical-frame supervision for native PersonaPlex's temporal/depth schedule.

The three aligned streams contain 1 + 8 + 8 codebooks on an 80 ms timeline.
Their timestamps describe media, not an invented generic attention mask. Native
codebook delays and depth ordering determine visibility in the model itself.
"""

import torch

from omnifactory.adapters.personaplex_model import DELAYS, FRAME_US
from omnifactory.adapters.supervision import tensor_ref, validate_supervision
from omnifactory.data.supervision import (
    AlignedStream,
    DiscreteRepresentation,
    LossObjective,
    Normalizer,
    ReductionMeasure,
    SupervisionPlan,
    SupervisionTarget,
    Timeline,
)
from omnifactory.errors import OmniFactoryError


def aligned_duplex_supervision(
    codes,
    supervised,
    *,
    text_card=32000,
    audio_card=2048,
    text_weight=1.0,
    audio_weight=1.0,
    user_weight=0.0,
    offset_us=0,
):
    """Bind one complete recording to a serializable prepared-data contract.

    Only actual vocabulary tokens are accepted here. Prefix frames are real tokens
    with False supervision masks; model-internal initial/zero tokens are not media.
    """
    if (
        codes.ndim != 3
        or codes.shape[:2] != (1, 17)
        or codes.shape[-1] == 0
        or codes.dtype != torch.long
        or supervised.shape != codes.shape
        or supervised.dtype != torch.bool
        or offset_us < 0
        or (codes < 0).any()
        or (codes[:, :1] >= text_card).any()
        or (codes[:, 1:] >= audio_card).any()
    ):
        raise OmniFactoryError("E_DUPLEX_LAYOUT", "one complete 17-codebook recording required")
    frames = codes.shape[-1]
    # Native forward_train excludes the tail that cannot be predicted after delay.
    mask = supervised.detach().cpu().clone()
    for cb, delay in enumerate(DELAYS):
        if delay:
            mask[:, cb, -delay:] = False
    starts = torch.arange(frames, dtype=torch.long) * FRAME_US + offset_us
    tensors = {
        "duplex/intervals": torch.stack([starts, starts + FRAME_US], -1),
        "duplex/available": starts + FRAME_US,
        "duplex/valid": torch.ones(frames, dtype=torch.bool),
    }
    intervals = tensor_ref("duplex/intervals", tensors["duplex/intervals"], ("frame", "boundary"))
    available = tensor_ref("duplex/available", tensors["duplex/available"], ("frame",))
    valid = tensor_ref("duplex/valid", tensors["duplex/valid"], ("frame",))
    targets, streams, objectives, normalizers = [], [], [], []
    for name, selection, weight, card, count, modality, direction in (
        ("text", slice(0, 1), text_weight, text_card, 1, "text", "output"),
        ("agent", slice(1, 9), audio_weight, audio_card, 8, "audio", "output"),
        ("user", slice(9, 17), user_weight, audio_card, 8, "audio", "input"),
    ):
        key = "duplex/" + name
        tensors[key] = codes[:, selection].detach().cpu().clone()
        tensors[key + "/mask"] = mask[:, selection].clone()
        axes = ("batch", "codebook", "frame")
        values = tensor_ref(key, tensors[key], axes)
        representation = DiscreteRepresentation(
            vocabulary_sizes=(card,) * count, codebook_axis="codebook"
        )
        targets.append(
            SupervisionTarget(
                id=name,
                modality=modality,
                representation=representation,
                values=values,
                mask=tensor_ref(key + "/mask", tensors[key + "/mask"], axes),
                measure=ReductionMeasure(unit="symbols"),
            )
        )
        streams.append(
            AlignedStream(
                id=name,
                timeline_id="conversation",
                modality=modality,
                direction=direction,
                representation=representation,
                values=values,
                event_axis="frame",
                intervals_us=intervals,
                available_at_us=available,
                valid=valid,
                # User prediction is a separate auxiliary objective. It does not
                # turn incoming audio into a generated output stream.
                target_id=name if direction == "output" else None,
            )
        )
        normalizers.append(Normalizer(id=name + "_symbols", unit="symbols"))
        objectives.append(
            LossObjective(
                id=name + "_ce",
                target_id=name,
                kind="cross_entropy",
                normalizer=name + "_symbols",
                weight=weight,
            )
        )
    plan = SupervisionPlan(
        targets=tuple(targets),
        streams=tuple(streams),
        objectives=tuple(objectives),
        normalizers=tuple(normalizers),
        timelines=(Timeline(id="conversation"),),
    )
    validate_supervision(plan, tensors)
    return plan, tensors
