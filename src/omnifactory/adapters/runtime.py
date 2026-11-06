"""Lazy adapter registry. Third-party adapters register once without patching the engine."""

from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points

from omnifactory.adapters.batching import TaskBatchCapability
from omnifactory.errors import OmniFactoryError


def resolve(reference):
    if callable(reference):
        return reference
    module, name = reference.split(":")
    return getattr(import_module(module), name)


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    implementation: str
    loaders: dict[str, str]
    policy_model: str
    outputs: tuple[str, ...] = ("text",)
    policy_prefixes: dict[str, str] = field(default_factory=dict)
    validator: str | None = None
    exporter: str = "omnifactory.training.export:export_builtin"
    artifact_validator: str = "omnifactory.artifacts:validate_native"
    model_spec: object | None = None
    loss_components: dict[str, tuple[str, ...]] = field(default_factory=dict)
    audio_unit: str | None = None
    speech_request_builder: str | None = None
    image_edit_request_builder: str | None = None
    realtime_probe: str | None = None
    named_speakers: bool = False
    stage_only: bool = False
    encoding_version: str = "1"
    batch_capabilities: dict[str, TaskBatchCapability] = field(default_factory=dict)
    # None delegates duration to the adapter's sequence/media budget checks.
    max_audio_seconds: float | None = 30.0

    @property
    def tasks(self):
        return tuple(self.loaders)

    def batch_capability(self, task):
        try:
            return self.batch_capabilities[task]
        except KeyError:
            raise OmniFactoryError(
                "E_BATCH_CAPABILITY", f"missing declaration: {self.name}/{task}"
            ) from None

    def validate(self, config):
        if self.validator:
            resolve(self.validator)(config)


_REGISTRY = {}
_PLUGINS_LOADED = False


def discover_adapters():
    """Installed entry points return AdapterSpec; duplicate names are rejected."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    existing = set(_REGISTRY)
    try:
        for point in entry_points(group="omnifactory.adapters"):
            spec = point.load()()
            if not isinstance(spec, AdapterSpec) or spec.name != point.name:
                raise OmniFactoryError("E_ADAPTER_PLUGIN", f"invalid plugin: {point.name}")
            register_adapter(spec)
    except Exception:
        for name in set(_REGISTRY) - existing:
            del _REGISTRY[name]
        _PLUGINS_LOADED = False
        raise


def register_adapter(spec: AdapterSpec):
    if spec.name in _REGISTRY:
        raise OmniFactoryError("E_ADAPTER_DUPLICATE", spec.name)
    if not spec.loaders or not spec.outputs:
        raise OmniFactoryError("E_ADAPTER", "tasks and output capabilities are required")
    if set(spec.batch_capabilities) != set(spec.loaders):
        raise OmniFactoryError("E_BATCH_CAPABILITY", "every task needs a batch capability")
    for capability in spec.batch_capabilities.values():
        if not isinstance(capability, TaskBatchCapability):
            raise OmniFactoryError("E_BATCH_CAPABILITY", "expected TaskBatchCapability")
        capability.validate()
    if set(spec.loss_components) != set(spec.loaders):
        raise OmniFactoryError("E_ADAPTER", "every task needs loss components")
    _REGISTRY[spec.name] = spec


def adapter_spec(name):
    discover_adapters()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise OmniFactoryError("E_ADAPTER", name) from None


def adapter_specs():
    discover_adapters()
    return tuple(_REGISTRY.values())


def adapter_class(name):
    return resolve(adapter_spec(name).implementation)


def load_model(config):
    spec = adapter_spec(config.adapter)
    if config.task not in spec.loaders:
        raise OmniFactoryError("E_ADAPTER_TASK", f"{config.adapter}/{config.task}")
    return resolve(spec.loaders[config.task])(config)


def policy_namespace(adapter, task="understanding_sft"):
    spec = adapter_spec(adapter)
    return spec.policy_model, spec.policy_prefixes.get(task, "")


def _builtin(name, module, cls, loaders, policy, **kwargs):
    heads = {
        "understanding_sft": ("language_model",),
        "speech_sft": ("speech_generator",),
        "flow_sft": ("audio_decoder",),
        "image_sft": ("image_head",),
    }
    if name in ("qwen3_omni", "qwen3_tts"):
        heads["speech_sft"] = ("speech_generator", "code_predictor")
    if name == "voxcpm2":
        heads["speech_sft"] = ("audio_decoder", "stop_predictor")
    if name == "personaplex":
        heads["speech_sft"] = ("speech_generator", "text_head")
    capabilities = {
        task: TaskBatchCapability(
            supervision=("continuous",)
            if task in ("flow_sft", "image_sft")
            else ("discrete", "continuous")
            if name == "voxcpm2"
            else ("discrete",),
            timed=name in ("minicpm_o_4_5", "personaplex"),
            input_layout="mimo_codebooks"
            if name == "mimo_audio"
            else "qwen3_tts_pairs"
            if name == "qwen3_tts"
            else "native",
            fallback_reason="task_native_batch_not_verified",
        )
        for task in loaders
    }
    if name in ("qwen2_5_omni", "minicpm_o_4_5"):
        capabilities["understanding_sft"] = TaskBatchCapability(
            mode="padded",
            input_layout="token_sequence",
            packing=name == "qwen2_5_omni",
            modalities=("text", "image", "audio"),
            heterogeneous=True,
            media_execution="native" if name == "qwen2_5_omni" else "per_example",
            timed=name == "minicpm_o_4_5",
        )
    kwargs["batch_capabilities"] = capabilities
    kwargs["loss_components"] = {task: heads[task] for task in loaders}
    register_adapter(
        AdapterSpec(
            name,
            f"omnifactory.adapters.{module}:{cls}",
            {task: f"omnifactory.adapters.{path}" for task, path in loaders.items()},
            policy,
            validator="omnifactory.adapters.configuration:validate_builtin",
            **kwargs,
        )
    )


_builtin(
    "qwen2_5_omni",
    "qwen2_5",
    "Qwen25Adapter",
    {"understanding_sft": "qwen2_5:load_thinker", "speech_sft": "qwen_speech:load_qwen_speech"},
    "qwen2_5_omni_3b",
    outputs=("text", "audio"),
    audio_unit="codec_frames",
    named_speakers=True,
    policy_prefixes={"understanding_sft": "thinker."},
)
_builtin(
    "qwen3_omni",
    "qwen3",
    "Qwen3Adapter",
    {
        "understanding_sft": "qwen3:load_qwen3_thinker",
        "speech_sft": "qwen3_speech:load_qwen3_speech",
    },
    "qwen3_omni",
    outputs=("text", "audio"),
    audio_unit="codec_frames",
    named_speakers=True,
    policy_prefixes={"understanding_sft": "thinker."},
)
_builtin(
    "minicpm_o_4_5",
    "minicpm",
    "MiniCPMAdapter",
    {"understanding_sft": "minicpm:load_minicpm", "speech_sft": "minicpm:load_minicpm"},
    "minicpm_o_4_5",
    outputs=("text", "audio"),
    audio_unit="codec_frames",
)
_builtin(
    "mimo_audio",
    "mimo",
    "MiMoAdapter",
    {"understanding_sft": "mimo_model:load_mimo", "speech_sft": "mimo_model:load_mimo"},
    "mimo_audio_7b",
    outputs=("text", "audio"),
    audio_unit="codec_frames",
)
_builtin(
    "minicpm_token2wav",
    "minicpm_flow",
    "MiniCPMFlowAdapter",
    {"flow_sft": "minicpm_flow:load_flow_model"},
    "minicpm_o_4_5",
    outputs=("audio",),
    stage_only=True,
)
_builtin(
    "voxcpm2",
    "voxcpm2",
    "VoxCPM2Adapter",
    {"speech_sft": "voxcpm2_model:load_voxcpm2"},
    "voxcpm2",
    outputs=("audio",),
    audio_unit="latent_patches",
)
_builtin(
    "bagel_dit",
    "bagel",
    "BagelAdapter",
    {"image_sft": "bagel_model:load_bagel"},
    "bagel",
    outputs=("image",),
)

_builtin(
    "qwen3_tts",
    "qwen3_tts",
    "Qwen3TTSAdapter",
    {"speech_sft": "qwen3_tts:load_qwen3_tts"},
    "qwen3_tts",
    outputs=("audio",),
    audio_unit="codec_frames",
    named_speakers=True,
    speech_request_builder="omnifactory.adapters.qwen3_tts:build_speech_request",
    exporter="omnifactory.adapters.qwen3_tts:export_qwen3_tts",
    artifact_validator="omnifactory.adapters.qwen3_tts:validate_export",
)

_builtin(
    "cosyvoice3",
    "cosyvoice3",
    "CosyVoice3Adapter",
    {
        "speech_sft": "cosyvoice3_model:load_cosyvoice3",
        "flow_sft": "cosyvoice3_model:load_cosyvoice3",
    },
    "cosyvoice3",
    outputs=("audio",),
    audio_unit="codec_frames",
    speech_request_builder="omnifactory.adapters.cosyvoice3:build_speech_request",
    exporter="omnifactory.adapters.cosyvoice3:export_cosyvoice3",
    artifact_validator="omnifactory.adapters.cosyvoice3:validate_export",
)

_builtin(
    "qwen_image_edit",
    "qwen_image_edit",
    "QwenImageEditAdapter",
    {"image_sft": "qwen_image_edit_model:load_qwen_image_edit"},
    "qwen_image_edit",
    outputs=("image",),
    exporter="omnifactory.adapters.qwen_image_edit:export_qwen_image_edit",
    artifact_validator="omnifactory.adapters.qwen_image_edit:validate_export",
    image_edit_request_builder="omnifactory.adapters.qwen_image_edit:build_edit_request",
)

_builtin(
    "personaplex",
    "personaplex",
    "PersonaPlexAdapter",
    {"speech_sft": "personaplex:load_model"},
    "personaplex",
    outputs=("text", "audio"),
    audio_unit="session_frames",
    exporter="omnifactory.adapters.personaplex:export_personaplex",
    artifact_validator="omnifactory.adapters.personaplex:validate_export",
    realtime_probe="omnifactory.adapters.personaplex_serving:probe_realtime",
    max_audio_seconds=None,
)


def task_batch_capability(config):
    """Recipe controls opt-in heterogeneous execution independently of support."""
    from dataclasses import replace

    capability = adapter_spec(config.adapter).batch_capability(config.task)
    return capability if config.heterogeneous_batches else replace(capability, heterogeneous=False)
