"""Different Omni families share contracts, not hard-coded thinker/talker stage enums."""

from typing import Literal

from pydantic import model_validator

from omnifactory.data.schema import Identifier, Record
from omnifactory.errors import OmniFactoryError


class ComponentSpec(Record):
    name: Identifier
    prefixes: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()


class StageSpec(Record):
    id: Identifier
    components: tuple[str, ...]
    input_stages: tuple[str, ...] = ()
    outputs: tuple[str, ...]


class ModelSpec(Record):
    key: Identifier
    model_id: Identifier
    family: Identifier
    contract_version: Identifier = "1"
    status: Literal["contract_only", "experimental_adapter"] = "contract_only"
    runtime_adapter: str | None = None
    runtime_verification: Literal["none", "tiny_random_cpu"] = "none"
    verification_records: tuple[dict, ...] = ()
    components: tuple[ComponentSpec, ...]
    stages: tuple[StageSpec, ...]
    input_combinations: tuple[tuple[str, ...], ...]
    condition_kinds: tuple[str, ...] = ()
    planned_tasks: tuple[str, ...] = ("understanding_sft",)
    packing_verified: bool = False
    # Components interleaved inside repeated blocks cannot be detached separately
    # when any member lies on a trainable gradient path.
    gradient_couplings: tuple[tuple[str, ...], ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_graphs(self):
        names = [c.name for c in self.components]
        stages = [s.id for s in self.stages]
        if len(set(names)) != len(names) or len(set(stages)) != len(stages):
            raise ValueError("component and stage IDs must be unique")
        self._acyclic({c.name: c.inputs for c in self.components})
        self._acyclic({s.id: s.input_stages for s in self.stages})
        for stage in self.stages:
            if not set(stage.components) <= set(names):
                raise ValueError("stage references unknown components")
        for group in self.gradient_couplings:
            if len(group) < 2 or len(set(group)) != len(group) or not set(group) <= set(names):
                raise ValueError("gradient coupling requires distinct known components")
        return self

    @staticmethod
    def _acyclic(graph: dict[str, tuple[str, ...]]) -> None:
        done, active = set(), set()

        def visit(name):
            if name not in graph:
                raise ValueError(f"unknown graph dependency: {name}")
            if name in active:
                raise ValueError(f"cyclic dependency: {name}")
            if name in done:
                return
            active.add(name)
            for parent in graph[name]:
                visit(parent)
            active.remove(name)
            done.add(name)

        for name in graph:
            visit(name)


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        if spec.key in self._models or any(
            m.model_id == spec.model_id for m in self._models.values()
        ):
            raise OmniFactoryError("E_DUPLICATE_MODEL", spec.key)
        self._models[spec.key] = spec

    def get(self, key: str) -> ModelSpec:
        for model in self._models.values():
            if key in (model.key, model.model_id):
                return model.model_copy(deep=True)
        raise OmniFactoryError("E_MODEL", f"unknown model {key!r}; run 'omnifactory models list'")

    def list(self) -> tuple[ModelSpec, ...]:
        return tuple(m.model_copy(deep=True) for m in self._models.values())


_INPUTS = (
    ("text",),
    ("image",),
    ("audio",),
    ("video",),
    ("image", "text"),
    ("audio", "text"),
    ("audio", "image"),
    ("audio", "image", "text"),
    ("text", "video"),
    ("audio", "video"),
    ("audio", "text", "video"),
)


def builtin_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            key="qwen2_5_omni_3b",
            status="experimental_adapter",
            runtime_adapter="qwen2_5_omni",
            runtime_verification="tiny_random_cpu",
            model_id="Qwen/Qwen2.5-Omni-3B",
            family="qwen_omni",
            planned_tasks=("understanding_sft", "speech_sft"),
            components=(
                ComponentSpec(name="vision_encoder", prefixes=("thinker.visual",)),
                ComponentSpec(
                    name="vision_projector",
                    prefixes=("thinker.visual.merger",),
                    inputs=("vision_encoder",),
                ),
                ComponentSpec(name="audio_encoder", prefixes=("thinker.audio_tower",)),
                ComponentSpec(
                    name="language_model",
                    prefixes=("thinker.model", "thinker.lm_head"),
                    inputs=("vision_projector", "audio_encoder"),
                ),
                ComponentSpec(
                    name="speech_generator", prefixes=("talker",), inputs=("language_model",)
                ),
                ComponentSpec(
                    name="audio_decoder", prefixes=("token2wav",), inputs=("speech_generator",)
                ),
            ),
            stages=(
                StageSpec(
                    id="thinker",
                    components=(
                        "vision_encoder",
                        "vision_projector",
                        "audio_encoder",
                        "language_model",
                    ),
                    outputs=("text", "features"),
                ),
                StageSpec(
                    id="talker",
                    components=("speech_generator",),
                    input_stages=("thinker",),
                    outputs=("codec",),
                ),
                StageSpec(
                    id="code2wav",
                    components=("audio_decoder",),
                    input_stages=("talker",),
                    outputs=("audio",),
                ),
            ),
            input_combinations=_INPUTS,
            notes=(
                "Native Thinker text/image/audio/video and Talker codec SFT; real inventory checked at load.",
                "Understanding and native-label speech recipes have different precision and asset requirements; see docs/models.md#qwen25-omni.",
                "The audio tower is treated as one component until its projection boundary is verified.",
                "Speech supervision requires native codec labels (teacher synthesis supported); arbitrary waveform encoding is unavailable.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="minicpm_o_4_5",
            model_id="openbmb/MiniCPM-o-4_5",
            family="minicpm_o",
            planned_tasks=("understanding_sft", "speech_sft"),
            status="experimental_adapter",
            runtime_adapter="minicpm_o_4_5",
            runtime_verification="tiny_random_cpu",
            components=(
                ComponentSpec(name="vision_encoder", prefixes=("vpm",)),
                ComponentSpec(
                    name="vision_projector", prefixes=("resampler",), inputs=("vision_encoder",)
                ),
                ComponentSpec(name="audio_encoder", prefixes=("apm",)),
                ComponentSpec(
                    name="audio_projector",
                    prefixes=("audio_projection_layer",),
                    inputs=("audio_encoder",),
                ),
                ComponentSpec(
                    name="language_model",
                    prefixes=("llm",),
                    inputs=("vision_projector", "audio_projector"),
                ),
                ComponentSpec(
                    name="speech_generator", prefixes=("tts",), inputs=("language_model",)
                ),
                ComponentSpec(
                    name="audio_decoder",
                    prefixes=("flow.decoder.estimator",),
                    inputs=("speech_generator",),
                ),
            ),
            stages=(
                StageSpec(
                    id="llm",
                    components=(
                        "vision_encoder",
                        "vision_projector",
                        "audio_encoder",
                        "audio_projector",
                        "language_model",
                    ),
                    outputs=("text", "features"),
                ),
                StageSpec(
                    id="tts",
                    components=("speech_generator",),
                    input_stages=("llm",),
                    outputs=("codec",),
                ),
                StageSpec(
                    id="code2wav",
                    components=("audio_decoder",),
                    input_stages=("tts",),
                    outputs=("audio",),
                ),
            ),
            input_combinations=_INPUTS,
            condition_kinds=("reference_audio", "language"),
            notes=(
                "Pinned native 4.5 implementation; see docs/models.md#minicpm-o-45 for required speech assets and supported tasks.",
                "Decoder is an external stage without an assumed HF parameter prefix.",
                "Offline video/audio and native S3 speech SFT implemented; NF4 and streaming training remain unsupported.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="qwen3_omni",
            model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            family="qwen_omni",
            status="experimental_adapter",
            runtime_adapter="qwen3_omni",
            runtime_verification="tiny_random_cpu",
            gradient_couplings=(
                ("language_model", "language_experts", "language_router"),
                ("speech_generator", "speech_experts", "speech_router", "code_predictor"),
            ),
            components=(
                ComponentSpec(name="vision_encoder", prefixes=("thinker.visual",)),
                ComponentSpec(name="audio_encoder", prefixes=("thinker.audio_tower",)),
                ComponentSpec(
                    name="vision_projector",
                    prefixes=("thinker.visual.merger", "thinker.visual.merger_list"),
                    inputs=("vision_encoder",),
                ),
                ComponentSpec(
                    name="audio_projector",
                    prefixes=("thinker.audio_tower.proj1", "thinker.audio_tower.proj2"),
                    inputs=("audio_encoder",),
                ),
                ComponentSpec(
                    name="language_experts",
                    prefixes=tuple(f"thinker.model.layers.{i}.mlp.experts" for i in range(48)),
                    inputs=("vision_projector", "audio_projector"),
                ),
                ComponentSpec(
                    name="language_router",
                    prefixes=tuple(f"thinker.model.layers.{i}.mlp.gate" for i in range(48)),
                    inputs=("vision_projector", "audio_projector"),
                ),
                ComponentSpec(
                    name="language_model",
                    prefixes=("thinker.model", "thinker.lm_head"),
                    inputs=(
                        "vision_projector",
                        "audio_projector",
                        "language_experts",
                        "language_router",
                    ),
                ),
                ComponentSpec(
                    name="speech_projection",
                    prefixes=("talker.text_projection", "talker.hidden_projection"),
                    inputs=("language_model",),
                ),
                ComponentSpec(
                    name="speech_experts",
                    prefixes=tuple(
                        f"talker.model.layers.{i}.mlp.{kind}"
                        for i in range(48)
                        for kind in ("experts", "shared_expert")
                    ),
                    inputs=("speech_projection",),
                ),
                ComponentSpec(
                    name="speech_router",
                    prefixes=tuple(
                        f"talker.model.layers.{i}.mlp.{kind}"
                        for i in range(48)
                        for kind in ("gate", "shared_expert_gate")
                    ),
                    inputs=("speech_projection",),
                ),
                ComponentSpec(
                    name="speech_generator",
                    prefixes=("talker",),
                    inputs=("speech_projection", "speech_experts", "speech_router"),
                ),
                ComponentSpec(
                    name="code_predictor",
                    prefixes=("talker.code_predictor",),
                    inputs=("speech_generator",),
                ),
                ComponentSpec(
                    name="audio_decoder", prefixes=("code2wav",), inputs=("code_predictor",)
                ),
            ),
            stages=(
                StageSpec(
                    id="thinker",
                    components=(
                        "vision_encoder",
                        "audio_encoder",
                        "vision_projector",
                        "audio_projector",
                        "language_experts",
                        "language_router",
                        "language_model",
                    ),
                    outputs=("text", "features"),
                ),
                StageSpec(
                    id="talker",
                    components=(
                        "speech_projection",
                        "speech_generator",
                        "speech_experts",
                        "speech_router",
                        "code_predictor",
                    ),
                    input_stages=("thinker",),
                    outputs=("codec",),
                ),
                StageSpec(
                    id="code2wav",
                    components=("audio_decoder",),
                    input_stages=("talker",),
                    outputs=("audio",),
                ),
            ),
            input_combinations=_INPUTS,
            condition_kinds=("speaker", "language"),
            planned_tasks=("understanding_sft", "speech_sft"),
            notes=(
                "Native 16-codebook speech SFT requires matching native teacher labels; no arbitrary-waveform encoder. See docs/models.md#qwen-speech-labels.",
                "Native understanding adapter; attention/expert LoRA and projector tuning, with frozen routers. Expert/projector tuning has tiny-model tests only. See docs/models.md#qwen3-omni.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="mimo_audio_7b",
            model_id="XiaomiMiMo/MiMo-Audio-7B-Instruct",
            family="mimo_audio",
            status="experimental_adapter",
            runtime_adapter="mimo_audio",
            runtime_verification="tiny_random_cpu",
            planned_tasks=("understanding_sft", "speech_sft"),
            input_combinations=(("text",), ("audio",), ("audio", "text")),
            components=(
                ComponentSpec(
                    name="audio_encoder", prefixes=("speech_embeddings", "input_local_transformer")
                ),
                ComponentSpec(
                    name="audio_projector",
                    prefixes=("speech_group_downcast",),
                    inputs=("audio_encoder",),
                ),
                ComponentSpec(
                    name="language_model",
                    prefixes=("model", "lm_head"),
                    inputs=("audio_projector",),
                ),
                ComponentSpec(
                    name="speech_generator",
                    prefixes=(
                        "hidden_states_downcast",
                        "local_transformer",
                        "local_transformer_lm_heads",
                        "speech_embeddings_to_local",
                    ),
                    inputs=("language_model", "audio_encoder"),
                ),
                ComponentSpec(
                    name="audio_decoder",
                    prefixes=("flow.decoder.estimator",),
                    inputs=("speech_generator",),
                ),
            ),
            stages=(
                StageSpec(
                    id="mimo_audio",
                    components=(
                        "audio_encoder",
                        "audio_projector",
                        "language_model",
                        "speech_generator",
                    ),
                    outputs=("text", "codec"),
                ),
                StageSpec(
                    id="code2wav",
                    components=("audio_decoder",),
                    input_stages=("mimo_audio",),
                    outputs=("audio",),
                ),
            ),
            notes=(
                "Audio-native family, no image/video inputs. Frozen external codec is separate from patch encoder.",
                "Multi-codebook patch SFT requires a matching frozen codec; see docs/models.md#mimo-audio for preparation and serving prerequisites.",
                "Offline utterance pairs; voice conditioning, assistant audio history and duplex training not implemented.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="voxcpm2",
            model_id="openbmb/VoxCPM2",
            family="voxcpm",
            status="experimental_adapter",
            runtime_adapter="voxcpm2",
            runtime_verification="tiny_random_cpu",
            planned_tasks=("speech_sft",),
            input_combinations=(("text",),),
            condition_kinds=("reference_audio",),
            components=(
                ComponentSpec(name="audio_vae", prefixes=("audio_vae",)),
                ComponentSpec(
                    name="audio_encoder", prefixes=("feat_encoder",), inputs=("audio_vae",)
                ),
                ComponentSpec(
                    name="audio_projector", prefixes=("enc_to_lm_proj",), inputs=("audio_encoder",)
                ),
                ComponentSpec(
                    name="language_model", prefixes=("base_lm",), inputs=("audio_projector",)
                ),
                ComponentSpec(
                    name="semantic_quantizer", prefixes=("fsq_layer",), inputs=("language_model",)
                ),
                ComponentSpec(
                    name="residual_model",
                    prefixes=("residual_lm", "fusion_concat_proj"),
                    inputs=("semantic_quantizer", "audio_projector"),
                ),
                ComponentSpec(
                    name="speech_projection",
                    prefixes=("lm_to_dit_proj", "res_to_dit_proj"),
                    inputs=("semantic_quantizer", "residual_model"),
                ),
                ComponentSpec(
                    name="audio_decoder", prefixes=("feat_decoder",), inputs=("speech_projection",)
                ),
                ComponentSpec(
                    name="stop_predictor",
                    prefixes=("stop_proj", "stop_head"),
                    inputs=("semantic_quantizer",),
                ),
            ),
            stages=(
                StageSpec(
                    id="voxcpm2",
                    components=(
                        "audio_vae",
                        "audio_encoder",
                        "audio_projector",
                        "language_model",
                        "semantic_quantizer",
                        "residual_model",
                        "speech_projection",
                        "audio_decoder",
                        "stop_predictor",
                    ),
                    outputs=("audio",),
                ),
            ),
            notes=(
                "Native continuous AudioVAE latent CFM plus binary stop objective; frozen AudioVAE.",
                "Text-to-speech and reference-only conditioning; no text answer, image/video or continuation training.",
                "Literal text-to-speech with continuous flow supervision; see docs/models.md#voxcpm2 for reference conditioning and generation limits.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="bagel",
            model_id="ByteDance-Seed/BAGEL-7B-MoT",
            family="bagel",
            status="experimental_adapter",
            runtime_adapter="bagel_dit",
            runtime_verification="tiny_random_cpu",
            planned_tasks=("image_sft",),
            input_combinations=(("text",),),
            components=(
                ComponentSpec(name="language_model", prefixes=("language_model",)),
                ComponentSpec(name="latent_projector", prefixes=("vae2llm",)),
                ComponentSpec(name="timestep_embedder", prefixes=("time_embedder",)),
                ComponentSpec(name="image_position", prefixes=("latent_pos_embed",)),
                ComponentSpec(
                    name="image_generator",
                    inputs=(
                        "language_model",
                        "latent_projector",
                        "timestep_embedder",
                        "image_position",
                    ),
                    prefixes=("language_model.model.norm_moe_gen",)
                    + tuple(
                        f"language_model.model.layers.{i}.{part}"
                        for i in range(128)
                        for part in (
                            "self_attn.q_proj_moe_gen",
                            "self_attn.k_proj_moe_gen",
                            "self_attn.v_proj_moe_gen",
                            "self_attn.o_proj_moe_gen",
                            "self_attn.q_norm_moe_gen",
                            "self_attn.k_norm_moe_gen",
                            "mlp_moe_gen",
                            "input_layernorm_moe_gen",
                            "post_attention_layernorm_moe_gen",
                        )
                    ),
                ),
                ComponentSpec(
                    name="image_head", prefixes=("llm2vae",), inputs=("image_generator",)
                ),
            ),
            stages=(
                StageSpec(
                    id="dit",
                    components=(
                        "language_model",
                        "latent_projector",
                        "timestep_embedder",
                        "image_position",
                        "image_generator",
                        "image_head",
                    ),
                    outputs=("image",),
                ),
            ),
            notes=(
                "Text-to-image training only; image understanding/editing are unsupported training tasks. See docs/models.md#bagel-dit.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="qwen3_tts",
            model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            family="qwen3_tts",
            status="experimental_adapter",
            runtime_adapter="qwen3_tts",
            runtime_verification="tiny_random_cpu",
            planned_tasks=("speech_sft",),
            input_combinations=(("text",),),
            condition_kinds=("reference_audio",),
            components=(
                ComponentSpec(name="speaker_encoder", prefixes=("speaker_encoder",)),
                ComponentSpec(name="text_embedding", prefixes=("talker.model.text_embedding",)),
                ComponentSpec(
                    name="text_projector",
                    prefixes=("talker.text_projection",),
                    inputs=("text_embedding",),
                ),
                ComponentSpec(name="audio_embedding", prefixes=("talker.model.codec_embedding",)),
                ComponentSpec(
                    name="speech_generator",
                    prefixes=("talker.model.layers", "talker.model.norm", "talker.codec_head"),
                    inputs=("text_projector", "audio_embedding"),
                ),
                ComponentSpec(
                    name="code_predictor",
                    prefixes=("talker.code_predictor",),
                    inputs=("speech_generator",),
                ),
            ),
            gradient_couplings=(("speech_generator", "code_predictor"),),
            stages=(
                StageSpec(
                    id="tts",
                    components=(
                        "speaker_encoder",
                        "text_embedding",
                        "text_projector",
                        "audio_embedding",
                        "speech_generator",
                        "code_predictor",
                    ),
                    outputs=("audio",),
                ),
            ),
            notes=(
                "Experimental 12 Hz Base single-speaker SFT; native codec and speaker encoder frozen.",
                "Exports a fixed-speaker CustomVoice checkpoint; bounded 1.7B BF16 LoRA/native/HTTP evidence is attached, not a quality claim.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="cosyvoice3",
            model_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            family="cosyvoice3",
            status="experimental_adapter",
            runtime_adapter="cosyvoice3",
            runtime_verification="tiny_random_cpu",
            planned_tasks=("speech_sft", "flow_sft"),
            input_combinations=(("text",),),
            condition_kinds=("reference_audio",),
            components=(
                ComponentSpec(
                    name="text_embedding",
                    prefixes=("talker.llm.model.model.embed_tokens", "talker.llm.model.lm_head"),
                ),
                ComponentSpec(name="audio_embedding", prefixes=("talker.speech_embedding",)),
                ComponentSpec(
                    name="speech_generator",
                    prefixes=(
                        "talker.llm.model.model.layers",
                        "talker.llm.model.model.norm",
                        "talker.llm_decoder",
                    ),
                    inputs=("text_embedding", "audio_embedding"),
                ),
                ComponentSpec(
                    name="audio_projector",
                    prefixes=(
                        "flow.input_embedding",
                        "flow.pre_lookahead_layer",
                        "flow.spk_embed_affine_layer",
                    ),
                ),
                ComponentSpec(
                    name="audio_decoder",
                    prefixes=("flow.decoder.estimator",),
                    inputs=("audio_projector",),
                ),
            ),
            stages=(
                StageSpec(
                    id="tts",
                    components=(
                        "text_embedding",
                        "audio_embedding",
                        "speech_generator",
                        "audio_projector",
                        "audio_decoder",
                    ),
                    outputs=("audio",),
                ),
            ),
            notes=(
                "Native LLM CE and mel flow training; frozen tokenizer, speaker encoder and vocoder.",
                "Bounded pretrained LLM/flow BF16 LoRA, native reload and HTTP evidence attached; not a speech-quality claim.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="qwen_image_edit",
            model_id="Qwen/Qwen-Image-Edit",
            family="qwen_image",
            status="experimental_adapter",
            runtime_adapter="qwen_image_edit",
            runtime_verification="tiny_random_cpu",
            planned_tasks=("image_sft",),
            input_combinations=(("text", "image"),),
            components=(
                ComponentSpec(name="latent_projector", prefixes=("img_in",)),
                ComponentSpec(name="text_projector", prefixes=("txt_in", "txt_norm")),
                ComponentSpec(name="timestep_embedder", prefixes=("time_text_embed",)),
                ComponentSpec(
                    name="image_generator",
                    prefixes=("transformer_blocks",),
                    inputs=("latent_projector", "text_projector", "timestep_embedder"),
                ),
                ComponentSpec(
                    name="image_head",
                    prefixes=("norm_out", "proj_out"),
                    inputs=("image_generator",),
                ),
            ),
            stages=(
                StageSpec(
                    id="dit",
                    components=(
                        "latent_projector",
                        "text_projector",
                        "timestep_embedder",
                        "image_generator",
                        "image_head",
                    ),
                    outputs=("image",),
                ),
            ),
            notes=(
                "Original single-image editing: bounded BF16 DiT LoRA, native export and HTTP evidence attached; not quality.",
                "Frozen Qwen2.5-VL/VAE prepared features; EditPlus/2509/2511 are separate variants.",
            ),
        )
    )
    registry.register(
        ModelSpec(
            key="personaplex",
            model_id="nvidia/personaplex-7b-v1",
            family="personaplex",
            status="experimental_adapter",
            runtime_adapter="personaplex",
            runtime_verification="tiny_random_cpu",
            planned_tasks=("speech_sft",),
            input_combinations=(("text", "audio"), ("audio",)),
            components=(
                ComponentSpec(name="input_embeddings", prefixes=("emb", "text_emb")),
                ComponentSpec(
                    name="temporal_model",
                    prefixes=("transformer", "out_norm"),
                    inputs=("input_embeddings",),
                ),
                ComponentSpec(
                    name="text_head", prefixes=("text_linear",), inputs=("temporal_model",)
                ),
                ComponentSpec(
                    name="audio_projector",
                    prefixes=("depformer_in", "depformer_emb", "depformer_text_emb"),
                    inputs=("temporal_model",),
                ),
                ComponentSpec(
                    name="code_predictor", prefixes=("depformer",), inputs=("audio_projector",)
                ),
                ComponentSpec(
                    name="speech_generator", prefixes=("linears",), inputs=("code_predictor",)
                ),
            ),
            stages=(
                StageSpec(
                    id="duplex",
                    components=(
                        "input_embeddings",
                        "temporal_model",
                        "text_head",
                        "audio_projector",
                        "code_predictor",
                        "speech_generator",
                    ),
                    outputs=("text", "audio"),
                ),
            ),
            notes=(
                "Native 17-stream SFT; explicit token/frame timing and bundled voice required.",
                "Native streaming and WebSocket duplex export; see per-artifact execution evidence, not a quality claim.",
            ),
        )
    )
    from omnifactory.adapters.runtime import adapter_specs
    from omnifactory.capabilities import verified_scopes

    for spec in adapter_specs():
        if spec.model_spec is not None and spec.model_spec.key not in registry._models:
            registry.register(spec.model_spec)
    for key, model in registry._models.items():
        registry._models[key] = model.model_copy(
            update={"verification_records": tuple(verified_scopes(key))}
        )
    return registry
