"""Explicit single-device training configuration; paths are relative to the YAML."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from omnifactory.data.schema import Record
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.adaptation import ComponentPolicy
from omnifactory.planning.batching import ResourceBudget


class TrainConfig(Record):
    model: str
    revision: str | None = None
    dataset: Path
    dataset_root: Path | None = None
    output_dir: Path
    adapter: str = "qwen2_5_omni"
    task: Literal["understanding_sft", "speech_sft", "flow_sft", "image_sft"] = "understanding_sft"
    image_height: Annotated[int, Field(gt=0)] = 256
    image_width: Annotated[int, Field(gt=0)] = 256
    duplex_schedule: Path | None = None
    duplex_text_alignment: Path | None = None
    personaplex_voice: str = "NATF2.pt"
    personaplex_user_loss_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    personaplex_agent_only_depth: bool = False
    max_attention_pairs: Annotated[int, Field(gt=0)] = 1_000_000
    flow_sigma_min: Annotated[float, Field(ge=0, lt=1)] = 1e-6
    flow_cfg_dropout: Annotated[float, Field(ge=0, lt=1)] = 0.2
    speech_tokenizer: Path | None = None
    speech_assets: Path | None = None
    speech_tokenizer_sha256: str | None = None
    speech_reference: Path | None = None
    speech_reference_sha256: str | None = None
    speech_reference_text: str | None = None
    speech_codes: Path | None = None
    speech_speaker: str = "Chelsie"
    cosyvoice_sequence_mode: Literal["mixed", "unistream", "bistream"] = "mixed"
    text_loss_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0
    speech_loss_weight: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    max_codec_tokens: Annotated[int, Field(gt=0)] = 750
    video_fps: Annotated[float, Field(gt=0, le=10)] = 1.0
    max_video_frames: Annotated[int, Field(gt=0)] = 32
    max_video_seconds: Annotated[float, Field(gt=0, le=30)] = 30.0
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    dtype: Literal["float32", "bfloat16"] = "float32"
    quantization: Literal["none", "nf4"] = "none"
    attention: Literal["eager", "sdpa"] = "sdpa"
    minicpm_precision: Literal["native", "fp32_decoder_compute"] = "native"
    feature_cache: Path | None = None
    feature_cache_components: tuple[Literal["vision_encoder", "audio_encoder"], ...] = (
        "vision_encoder",
        "audio_encoder",
    )
    packing: Literal["none", "flash_attention_2"] = "none"
    gradient_checkpointing: bool = True
    deterministic: bool = False
    max_steps: Annotated[int, Field(gt=0)] = 20
    save_steps: Annotated[int, Field(gt=0)] = 10
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    max_length: Annotated[int, Field(gt=1)] = 2048
    max_audio_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 30
    max_image_pixels: Annotated[int, Field(gt=0)] = 1048576
    seed: int = 42
    max_grad_norm: Annotated[float, Field(gt=0)] = 1.0
    components: dict[str, ComponentPolicy]
    budget: ResourceBudget
    # Opt in after checking the native recipe/dtype; no packing or speed claim.
    heterogeneous_batches: bool = False
    window_size: Annotated[int, Field(gt=0)] = 64
    resume_from: Path | None = None
    memory_profile: Path | None = None
    prepared_dataset: Path | None = None
    prepared_shard_records: Annotated[int, Field(gt=0)] = 256
    prepared_shard_bytes: Annotated[int, Field(gt=0)] = 256 * 1024**2
    prepared_prefetch_records: Annotated[int, Field(ge=0)] = 2
    prepared_prefetch_bytes: Annotated[int, Field(ge=0)] = 64 * 1024**2
    prepared_cache_bytes: Annotated[int, Field(ge=0)] = 0
    validation_dataset: Path | None = None
    validation_root: Path | None = None
    eval_steps: Annotated[int, Field(ge=0)] = 0
    save_total_limit: Annotated[int, Field(ge=2)] | None = None
    weight_decay: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    optimizer_master_weights: bool = False
    lr_scheduler: Literal["constant", "linear", "cosine"] = "constant"
    warmup_steps: Annotated[int, Field(ge=0)] = 0
    scheduler_steps: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def valid_runtime(self):
        from omnifactory.adapters.runtime import adapter_spec

        spec = adapter_spec(self.adapter)
        if self.task not in spec.tasks:
            raise ValueError(f"{self.adapter} does not support {self.task}")
        if spec.max_audio_seconds is not None and self.max_audio_seconds > spec.max_audio_seconds:
            raise ValueError(
                f"{self.adapter} max_audio_seconds must be <= {spec.max_audio_seconds:g}"
            )
        spec.validate(self)
        if self.lr_scheduler != "constant" and self.scheduler_steps is None:
            raise ValueError(
                "linear/cosine schedules require fixed scheduler_steps for exact resume"
            )
        if self.scheduler_steps and self.warmup_steps >= self.scheduler_steps:
            raise ValueError("warmup_steps must be smaller than scheduler_steps")
        if self.eval_steps and self.validation_dataset is None:
            raise ValueError("eval_steps requires validation_dataset")
        if self.quantization != "none" and (self.device != "cuda" or self.dtype != "bfloat16"):
            raise ValueError("NF4 requires CUDA and bfloat16 compute")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU reference execution requires float32")
        if self.minicpm_precision != "native" and (
            self.adapter != "minicpm_o_4_5"
            or self.task != "understanding_sft"
            or self.device != "cuda"
            or self.dtype != "bfloat16"
            or self.attention != "sdpa"
            or self.duplex_schedule is not None
            or set(self.components) != {"language_model"}
            or self.components["language_model"].mode != "lora"
        ):
            raise ValueError(
                "minicpm_precision requires untimed MiniCPM understanding CUDA/BF16/SDPA "
                "with language_model LoRA only"
            )
        if self.budget.max_gpu_bytes is not None and self.memory_profile is None:
            raise ValueError(
                "GPU budgeting requires an accepted memory_profile calibration artifact"
            )
        if self.memory_profile is not None and self.device != "cuda":
            raise ValueError("measured GPU memory profiles require CUDA")
        if self.speech_tokenizer_sha256 is not None:
            import re

            if not re.fullmatch(r"[0-9a-f]{64}", self.speech_tokenizer_sha256):
                raise ValueError("speech_tokenizer_sha256 must be 64 lowercase hex characters")
        if not self.components:
            raise ValueError("configure at least one trainable component")
        return self

    def execution_contract(self):
        contract = self.model_dump(
            mode="json",
            exclude={
                "output_dir",
                "resume_from",
                "max_steps",
                "save_steps",
                "prepared_dataset",
                "save_total_limit",
                "prepared_cache_bytes",
                "prepared_shard_records",
                "prepared_shard_bytes",
                "prepared_prefetch_records",
                "prepared_prefetch_bytes",
            },
        )
        # Preserve existing default contracts; deterministic execution is opt-in.
        if not self.deterministic:
            contract.pop("deterministic")
        if not self.personaplex_agent_only_depth:
            contract.pop("personaplex_agent_only_depth")
        if not self.optimizer_master_weights:
            contract.pop("optimizer_master_weights")
        if self.minicpm_precision == "native":
            contract.pop("minicpm_precision")
        from dataclasses import asdict

        from omnifactory.adapters.runtime import task_batch_capability

        contract["batch_execution_contract"] = {
            "version": "capability-batching-v1",
            "task": asdict(task_batch_capability(self)),
        }
        return contract

    def encoding_contract(self):
        # Optimizer choices do not change native labels or preprocessed inputs.
        excluded = {
            "output_dir",
            "resume_from",
            "max_steps",
            "save_steps",
            "prepared_dataset",
            "components",
            "budget",
            "window_size",
            "heterogeneous_batches",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "weight_decay",
            "optimizer_master_weights",
            "minicpm_precision",
            "lr_scheduler",
            "warmup_steps",
            "scheduler_steps",
            "validation_dataset",
            "validation_root",
            "eval_steps",
            "save_total_limit",
            "memory_profile",
            "feature_cache",
            "feature_cache_components",
            "packing",
            "gradient_checkpointing",
            "deterministic",
            "personaplex_agent_only_depth",
            "prepared_cache_bytes",
            "prepared_shard_records",
            "prepared_shard_bytes",
            "prepared_prefetch_records",
            "prepared_prefetch_bytes",
        }
        return self.model_dump(mode="json", exclude=excluded)


def read_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OmniFactoryError("E_CONFIG", "expected a YAML mapping")
    root = path.resolve().parent
    for key in (
        "prepared_dataset",
        "validation_dataset",
        "validation_root",
        "memory_profile",
        "feature_cache",
        "duplex_schedule",
        "duplex_text_alignment",
        "dataset",
        "dataset_root",
        "output_dir",
        "resume_from",
        "speech_tokenizer",
        "speech_assets",
        "speech_codes",
        "speech_reference",
    ):
        if raw.get(key) is not None:
            raw[key] = str((root / Path(raw[key]).expanduser()).resolve())
    model = raw.get("model", "")
    if model.startswith((".", "/", "~")) or (root / model).is_dir():
        raw["model"] = str((root / Path(model).expanduser()).resolve())
    return TrainConfig.model_validate(raw)
