"""Concrete semantic processing facade; no placeholder tensors or pretend training."""

from typing import Literal

from omnifactory.data.media import LocalMediaResolver, MediaManifest
from omnifactory.data.schema import (
    CanonicalSample,
    ContentBlock,
    InferenceView,
    Prediction,
    Record,
    VideoBlock,
    inference_view,
)
from omnifactory.errors import OmniFactoryError
from omnifactory.models.registry import ModelRegistry, builtin_registry


class Diagnostic(Record):
    code: str
    prediction_id: str
    message: str
    severity: Literal["error", "warning"] = "error"


class PredictionPlan(Record):
    prediction_id: str
    context_message_ids: tuple[str, ...]
    target_block_ids: tuple[str, ...]
    ignored_target_block_ids: tuple[str, ...]
    input_modalities: tuple[str, ...]


class PreparationPlan(Record):
    model: str
    task: str
    sample_id: str
    status: Literal["semantic_plan_only"] = "semantic_plan_only"
    training_ready: Literal[False] = False
    predictions: tuple[PredictionPlan, ...]
    stages: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]
    pending: tuple[str, ...] = (
        "model-specific media encoding and exact token-cost measurement",
        "runtime adapter selection, model loading and training execution",
        "native weight export and vLLM-Omni deployment verification",
    )


class OmniDataProcessor:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or builtin_registry()

    def encode(self, sample, prediction_id, *, adapter, contract_id, inference=False):
        """Execute a supplied family adapter and validate its typed encoded contract.

        Training dependencies are loaded only here. Semantic planning remains lightweight.
        Native tensors and supervision targets are separate views of the encoded artifact.
        """
        from omnifactory.adapters.base import EncodedExample

        encoded = adapter.encode(
            sample, prediction_id, contract_id=contract_id, inference=inference
        )
        if not isinstance(encoded, EncodedExample):
            raise OmniFactoryError("E_ENCODED_TYPE", "adapter must return EncodedExample")
        return encoded.validate(inference=inference)

    def normalize(self, record: dict) -> CanonicalSample:
        return CanonicalSample.model_validate(record)

    @staticmethod
    def _targets(sample: CanonicalSample, prediction: Prediction) -> tuple[ContentBlock, ...]:
        message = next(m for m in sample.messages if m.id == prediction.target_message_id)
        # Message order is authoritative, even when target IDs were supplied in another order.
        return tuple(b for b in message.content if b.id in prediction.target_block_ids)

    def validate(
        self, sample: CanonicalSample, *, model: str, task: str = "understanding_sft"
    ) -> tuple[Diagnostic, ...]:
        spec = self.registry.get(model)
        if task not in spec.planned_tasks:
            raise OmniFactoryError("E_TASK", f"{spec.key} has no contract for {task}")
        findings = []
        allowed = {frozenset(combo) for combo in spec.input_combinations}
        for prediction in sample.predictions:
            view = self.to_inference(sample, prediction.id)
            modalities = self._modalities(view)
            if frozenset(modalities) not in allowed:
                findings.append(
                    Diagnostic(
                        code="E_INPUT_COMBINATION",
                        prediction_id=prediction.id,
                        message=f"unsupported input combination: {modalities}",
                    )
                )
            for condition in prediction.conditions:
                if condition.kind not in spec.condition_kinds:
                    findings.append(
                        Diagnostic(
                            code="E_CONDITION",
                            prediction_id=prediction.id,
                            message=f"unsupported condition: {condition.kind}",
                        )
                    )
                elif task == "understanding_sft" and condition.kind == "reference_audio":
                    findings.append(
                        Diagnostic(
                            code="W_UNUSED_CONDITION",
                            prediction_id=prediction.id,
                            message="reference voice is for speech generation; understanding recipe does not train it",
                            severity="warning",
                        )
                    )
            targets = self._targets(sample, prediction)
            text_ids = {b.id for b in targets if b.type == "text"}
            audio_ids = {b.id for b in targets if b.type == "audio"}
            if task == "image_sft":
                if len(targets) != 1 or targets[0].type != "image":
                    findings.append(
                        Diagnostic(
                            code="E_IMAGE_TARGET",
                            prediction_id=prediction.id,
                            message="image SFT requires exactly one image target",
                        )
                    )
                continue
            if any(b.type in ("image", "video") for b in targets):
                findings.append(
                    Diagnostic(
                        code="E_TARGET_MODALITY",
                        prediction_id=prediction.id,
                        message="image/video generation training is outside the current contracts",
                    )
                )
            if not text_ids:
                findings.append(
                    Diagnostic(
                        code="E_MISSING_TRANSCRIPT",
                        prediction_id=prediction.id,
                        message="selected answer text is required; no automatic ASR target generation",
                    )
                )
            if task == "speech_sft":
                if not audio_ids:
                    findings.append(
                        Diagnostic(
                            code="E_AUDIO_TARGET",
                            prediction_id=prediction.id,
                            message="speech SFT requires target audio",
                        )
                    )
                for audio_id in audio_ids:
                    pairs = [
                        p
                        for p in sample.speech_pairs
                        if p.audio_block_id == audio_id and p.text_block_id in text_ids
                    ]
                    if len(pairs) != 1:
                        findings.append(
                            Diagnostic(
                                code="E_SPEECH_PAIR",
                                prediction_id=prediction.id,
                                message=f"{audio_id} needs exactly one selected paired transcript",
                            )
                        )
            elif audio_ids:
                findings.append(
                    Diagnostic(
                        code="W_UNUSED_TARGET",
                        prediction_id=prediction.id,
                        message="audio targets are not supervised by understanding_sft",
                        severity="warning",
                    )
                )
        return tuple(findings)

    @staticmethod
    def _modalities(view: InferenceView) -> tuple[str, ...]:
        kinds = {block.type for message in view.messages for block in message.content}
        if any(
            isinstance(b, VideoBlock) and b.audio_mode == "include"
            for m in view.messages
            for b in m.content
        ):
            kinds.add("audio")
        return tuple(sorted(kinds))

    def plan(
        self, sample: CanonicalSample, *, model: str, task: str = "understanding_sft"
    ) -> PreparationPlan:
        diagnostics = self.validate(sample, model=model, task=task)
        errors = [d for d in diagnostics if d.severity == "error"]
        if errors:
            raise OmniFactoryError(
                "E_PREPARATION",
                "; ".join(f"{d.prediction_id}: {d.code}: {d.message}" for d in errors),
            )
        spec = self.registry.get(model)
        predictions = []
        for prediction in sample.predictions:
            targets = self._targets(sample, prediction)
            selected = tuple(
                b.id for b in targets if b.type == "text" or task in ("speech_sft", "image_sft")
            )
            view = self.to_inference(sample, prediction.id)
            predictions.append(
                PredictionPlan(
                    prediction_id=prediction.id,
                    context_message_ids=tuple(m.id for m in view.messages),
                    target_block_ids=selected,
                    ignored_target_block_ids=tuple(b.id for b in targets if b.id not in selected),
                    input_modalities=self._modalities(view),
                )
            )
        pending = PreparationPlan.model_fields["pending"].default
        if task == "speech_sft":
            pending += ("target codec compatibility and causal speech supervision validation",)
        return PreparationPlan(
            model=spec.key,
            task=task,
            sample_id=sample.id,
            predictions=tuple(predictions),
            stages=tuple(s.id for s in spec.stages),
            diagnostics=diagnostics,
            pending=pending,
        )

    def manifest(self, sample: CanonicalSample, *, resolver: LocalMediaResolver) -> MediaManifest:
        return resolver.manifest(sample)

    def to_inference(
        self,
        sample: CanonicalSample,
        prediction_id: str,
        requested_outputs: tuple[Literal["text", "audio", "image"], ...] = ("text",),
    ) -> InferenceView:
        return inference_view(sample, prediction_id, requested_outputs)
