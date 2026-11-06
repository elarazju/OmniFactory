"""Canonical conversation data v0.1. No vLLM, Transformers or torch dependency."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1)]
Modality = Literal["text", "image", "audio", "video"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MediaAsset(Record):
    uri: Identifier
    modality: Literal["image", "audio", "video"]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    # Import-time metadata is advisory; MediaResolver verifies it before encoding.
    sample_rate_hz: Annotated[int, Field(gt=0)] | None = None
    duration_us: Annotated[int, Field(gt=0)] | None = None
    derived_from: Identifier | None = None


class MediaSlice(Record):
    asset_id: Identifier
    start_us: Annotated[int, Field(ge=0)] = 0
    end_us: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.end_us is not None and self.end_us <= self.start_us:
            raise ValueError("E_TIME_RANGE: end_us must exceed start_us")
        return self


class Placement(Record):
    """The selected slice starts at offset_us on a shared sample-local timeline."""

    timeline_id: Identifier
    offset_us: Annotated[int, Field(ge=0)]


class TextBlock(Record):
    id: Identifier
    type: Literal["text"]
    text: Annotated[str, Field(min_length=1)]


class ImageBlock(Record):
    id: Identifier
    type: Literal["image"]
    asset_id: Identifier


class AudioBlock(Record):
    id: Identifier
    type: Literal["audio"]
    media: MediaSlice
    placement: Placement | None = None


class VideoBlock(Record):
    id: Identifier
    type: Literal["video"]
    media: MediaSlice
    # Required: never silently discard or double-count an embedded soundtrack.
    audio_mode: Literal["include", "exclude"]
    placement: Placement | None = None


ContentBlock = Annotated[
    TextBlock | ImageBlock | AudioBlock | VideoBlock, Field(discriminator="type")
]


class Message(Record):
    id: Identifier
    role: Literal["system", "user", "assistant"]
    content: Annotated[tuple[ContentBlock, ...], Field(min_length=1)]


class SpeakerCondition(Record):
    kind: Literal["speaker"]
    value: Identifier


class LanguageCondition(Record):
    kind: Literal["language"]
    value: Identifier


class ReferenceAudioCondition(Record):
    kind: Literal["reference_audio"]
    media: MediaSlice
    transcript: str | None = None


Condition = Annotated[
    SpeakerCondition | LanguageCondition | ReferenceAudioCondition,
    Field(discriminator="kind"),
]


class Prediction(Record):
    """One prediction boundary; only earlier messages are visible at evaluation."""

    id: Identifier
    target_message_id: Identifier
    target_block_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    conditions: tuple[Condition, ...] = ()


class SpeechPair(Record):
    text_block_id: Identifier
    audio_block_id: Identifier
    # This association makes no claim about word/token/frame timing.
    alignment: Literal["utterance"] = "utterance"


class Provenance(Record):
    dataset: Identifier
    source_id: Identifier
    group_id: Identifier
    split: Literal["train", "validation", "test"]


class CanonicalSample(Record):
    schema_version: Literal["0.1"]
    id: Identifier
    provenance: Provenance
    assets: dict[str, MediaAsset]
    messages: Annotated[tuple[Message, ...], Field(min_length=2)]
    predictions: Annotated[tuple[Prediction, ...], Field(min_length=1)]
    speech_pairs: tuple[SpeechPair, ...] = ()

    @model_validator(mode="after")
    def references_and_boundaries(self):
        message_map = {m.id: m for m in self.messages}
        blocks = {b.id: b for m in self.messages for b in m.content}
        block_count = sum(len(m.content) for m in self.messages)
        if len(message_map) != len(self.messages) or len(blocks) != block_count:
            raise ValueError("E_DUPLICATE_ID: message and block IDs must be unique")
        if len({p.id for p in self.predictions}) != len(self.predictions):
            raise ValueError("E_DUPLICATE_ID: prediction IDs must be unique")
        for key, asset in self.assets.items():
            if not key:
                raise ValueError("E_ASSET: empty asset ID")
            seen = {key}
            parent = asset.derived_from
            while parent is not None:
                if parent not in self.assets or parent in seen:
                    raise ValueError("E_ASSET_LINEAGE: dangling or cyclic derived_from")
                seen.add(parent)
                parent = self.assets[parent].derived_from
        for block in blocks.values():
            if isinstance(block, ImageBlock):
                self._check_asset(block.asset_id, "image")
            elif isinstance(block, (AudioBlock, VideoBlock)):
                self._check_slice(block.media, block.type)
        for pair in self.speech_pairs:
            if not isinstance(blocks.get(pair.text_block_id), TextBlock) or not isinstance(
                blocks.get(pair.audio_block_id), AudioBlock
            ):
                raise ValueError("E_SPEECH_PAIR: expected existing text and audio blocks")
            owners = [
                m.id
                for m in self.messages
                if any(b.id in (pair.text_block_id, pair.audio_block_id) for b in m.content)
            ]
            if len(owners) != 1:
                raise ValueError("E_SPEECH_PAIR: paired blocks must belong to one message")
        for prediction in self.predictions:
            target = message_map.get(prediction.target_message_id)
            if target is None or target.role != "assistant":
                raise ValueError("E_TARGET: target must be an existing assistant message")
            if self.messages.index(target) == 0:
                raise ValueError("E_TARGET: prediction needs preceding context")
            local = {b.id for b in target.content}
            ids = prediction.target_block_ids
            if len(set(ids)) != len(ids) or not set(ids) <= local:
                raise ValueError("E_TARGET: unique target blocks must belong to target message")
            kinds = [c.kind for c in prediction.conditions]
            if len(set(kinds)) != len(kinds):
                raise ValueError("E_CONDITION: duplicate condition kinds in v0.1")
            for condition in prediction.conditions:
                if isinstance(condition, ReferenceAudioCondition):
                    self._check_slice(condition.media, "audio")
                    for block in target.content:
                        if isinstance(block, AudioBlock) and self._overlap(
                            condition.media, block.media
                        ):
                            raise ValueError("E_TARGET_LEAK: reference audio overlaps target audio")
        return self

    def _check_asset(self, asset_id: str, modality: str) -> MediaAsset:
        asset = self.assets.get(asset_id)
        if asset is None or asset.modality != modality:
            raise ValueError("E_ASSET: missing asset or mismatched modality")
        return asset

    def _check_slice(self, media: MediaSlice, modality: str) -> None:
        asset = self._check_asset(media.asset_id, modality)
        if asset.duration_us is not None and (
            media.start_us >= asset.duration_us
            or (media.end_us is not None and media.end_us > asset.duration_us)
        ):
            raise ValueError("E_TIME_RANGE: slice exceeds declared duration")

    def _overlap(self, left: MediaSlice, right: MediaSlice) -> bool:
        a, b = self.assets[left.asset_id], self.assets[right.asset_id]
        same = (
            left.asset_id == right.asset_id
            or a.uri == b.uri
            or (a.sha256 is not None and a.sha256 == b.sha256)
        )
        return same and max(left.start_us, right.start_us) < min(
            left.end_us if left.end_us is not None else float("inf"),
            right.end_us if right.end_us is not None else float("inf"),
        )


class InferenceView(Record):
    """Target-free, backend-neutral input. Not an OpenAI HTTP request body."""

    sample_id: str
    prediction_id: str
    assets: dict[str, MediaAsset]
    messages: tuple[Message, ...]
    conditions: tuple[Condition, ...]
    requested_outputs: Annotated[tuple[Literal["text", "audio", "image"], ...], Field(min_length=1)]


def inference_view(
    sample: CanonicalSample,
    prediction_id: str,
    requested_outputs: tuple[Literal["text", "audio", "image"], ...],
) -> InferenceView:
    """Whitelist earlier context and reachable assets; never serialize the sample wholesale."""
    prediction = next((p for p in sample.predictions if p.id == prediction_id), None)
    if prediction is None:
        raise ValueError("E_PREDICTION: unknown prediction ID")
    stop = next(i for i, m in enumerate(sample.messages) if m.id == prediction.target_message_id)
    history = sample.messages[:stop]
    asset_ids = set()
    for message in history:
        for block in message.content:
            if isinstance(block, ImageBlock):
                asset_ids.add(block.asset_id)
            elif isinstance(block, (AudioBlock, VideoBlock)):
                asset_ids.add(block.media.asset_id)
    for condition in prediction.conditions:
        if isinstance(condition, ReferenceAudioCondition):
            asset_ids.add(condition.media.asset_id)
    # Provenance/lineage are not serving inputs. A parent may be a target-only asset.
    assets = {
        key: sample.assets[key].model_copy(update={"derived_from": None})
        for key in sorted(asset_ids)
    }
    return InferenceView(
        sample_id=sample.id,
        prediction_id=prediction.id,
        assets=assets,
        messages=history,
        conditions=prediction.conditions,
        requested_outputs=requested_outputs,
    )
