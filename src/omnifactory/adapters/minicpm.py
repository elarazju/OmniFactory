"""MiniCPM-o 4.5 understanding SFT against pinned OpenBMB native code.

Preserves native image slicing and audio placeholders. Media encoders run per
sample/clip to avoid cross-sample padding and the upstream audio scatter bug.
Offline video and native speech SFT are supported; streaming remains separate.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from omnifactory._vendor.minicpm_o45.configuration_minicpmo import MiniCPMOConfig
from omnifactory._vendor.minicpm_o45.modeling_minicpmo import MiniCPMO
from omnifactory._vendor.minicpm_o45.processing_minicpmo import (
    MiniCPMAAudioProcessor,
    MiniCPMOProcessor,
    MiniCPMVImageProcessor,
)
from omnifactory._vendor.minicpm_o45.tokenization_minicpmo_fast import MiniCPMOTokenizerFast
from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
from omnifactory.adapters.media import MediaInputs
from omnifactory.adapters.supervision import discrete_supervision
from omnifactory.data.media import LocalMediaResolver
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import SampleCost
from omnifactory.training.artifacts import write_json
from omnifactory.training.losses import LossBundle, attach_losses, symbol_losses

REVISION = "eb4141de24550a3a1a4082f718be58f46b761728"
NATIVE_PREFIXES = ("llm.", "vpm.", "resampler.", "apm.", "audio_projection_layer.")


def load_processor(model, revision=None):
    kwargs = {"revision": revision}
    image = MiniCPMVImageProcessor.from_pretrained(model, **kwargs)
    audio = MiniCPMAAudioProcessor.from_pretrained(model, **kwargs)
    tokenizer = MiniCPMOTokenizerFast.from_pretrained(model, **kwargs)
    # Native processor defaults to five; retain explicit metadata on saved processors.
    settings, _ = MiniCPMVImageProcessor.get_image_processor_dict(model, **kwargs)
    return MiniCPMOProcessor(
        image, audio, tokenizer, audio_pool_step=settings.get("audio_pool_step", 5)
    )


def save_processor(processor, path):
    path.mkdir(parents=True, exist_ok=True)
    processor.tokenizer.save_pretrained(path)
    settings = {
        **json.loads(processor.image_processor.to_json_string()),
        **json.loads(processor.audio_processor.to_json_string()),
    }
    settings["norm_mean"] = processor.image_processor.mean.tolist()
    settings["norm_std"] = processor.image_processor.std.tolist()
    settings.update(processor_class="MiniCPMOProcessor", audio_pool_step=processor.pool_step)
    write_json(path / "preprocessor_config.json", settings)
    # Self-contained native sources, with licenses and exact revision provenance.
    source = Path(__file__).parents[1] / "_vendor/minicpm_o45"
    for file in source.iterdir():
        if file.is_file() and file.name != "__init__.py":
            shutil.copy2(file, path / file.name)


class MiniCPMAdapter(MediaInputs):
    reserved_content_tokens = ("<audio>", "</audio>")

    def __init__(
        self,
        processor,
        *,
        root,
        max_length=2048,
        max_audio_seconds=30,
        max_image_pixels=1048576,
        max_decoded_bytes=67108864,
    ):
        self.task = "understanding_sft"
        self.training_config = None
        self.codec = None
        self.processor = processor
        self.resolver = LocalMediaResolver(root)
        self.max_length = max_length
        self.max_audio_seconds = min(max_audio_seconds, processor.audio_processor.chunk_length)
        self.max_image_pixels = max_image_pixels
        self.max_decoded_bytes = max_decoded_bytes

    @classmethod
    def from_config(cls, config):
        adapter = cls(
            load_processor(config.model, config.revision),
            root=config.dataset_root or config.dataset.parent,
            max_length=config.max_length,
            max_audio_seconds=config.max_audio_seconds,
            max_image_pixels=config.max_image_pixels,
            max_decoded_bytes=config.budget.max_decoded_media_bytes,
        )
        adapter.task = config.task
        adapter.training_config = config
        adapter.video_options = {
            "fps": config.video_fps,
            "max_frames": config.max_video_frames,
            "max_seconds": config.max_video_seconds,
        }
        return adapter

    @staticmethod
    def to_device(batch, model, config):
        return {
            k: move_tree(
                v,
                device=config.device,
                dtype=model.apm.conv1.weight.dtype if k == "audio_features" else model.vpm.dtype,
            )
            for k, v in batch.items()
        }

    def save_processor(self, path):
        save_processor(self.processor, path)

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        messages, images, audio, decoded, modalities = self._messages(sample, prediction_id)
        # MiniCPM's native tokenizer expects string content with ordered placeholders.
        messages = [
            {
                "role": m["role"],
                "content": ("" if "video" in modalities else "\n").join(
                    b["text"]
                    if b["type"] == "text"
                    else "<image>./</image>"
                    if b["type"] == "image"
                    else "<audio>./</audio>"
                    for b in m["content"]
                ),
            }
            for m in messages
        ]
        prediction = next(p for p in sample.predictions if p.id == prediction_id)
        target = next(m for m in sample.messages if m.id == prediction.target_message_id)
        selected = [b for b in target.content if b.id in prediction.target_block_ids]
        speech = self.task == "speech_sft"
        if any(b.type not in (("text", "audio") if speech else ("text",)) for b in selected):
            raise OmniFactoryError(
                "E_RUNTIME_TARGET", "MiniCPM understanding requires text targets"
            )
        answer = "\n".join(b.text for b in selected if b.type == "text")
        if any(
            t in answer
            for t in self.processor.tokenizer.all_special_tokens
            + list(self.reserved_content_tokens)
        ):
            raise OmniFactoryError("E_CONTROL_TOKEN", "literal special tokens in targets")

        # Disable thinking and speech-specific templates consistently in prompt and target.
        def render(msgs, prompt):
            return self.processor.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=prompt,
                enable_thinking=False,
                use_tts_template=speech,
            )

        def process(text):
            result = dict(
                self.processor(
                    text=[text],
                    images=[images] if images else None,
                    audios=[audio],
                    sampling_rate=16000,
                    stream_input=False,
                    use_image_id=self.processor.image_processor.use_image_id,
                    return_tensors="pt",
                )
            )
            result.pop("image_sizes", None)
            result.pop("token_type_ids", None)
            result["input_ids"] = result["input_ids"].long()
            return result

        prompt = process(render(messages, True))
        prompt_length = prompt["input_ids"].shape[1]
        inputs = (
            prompt
            if inference
            else process(
                render(
                    messages
                    + [
                        {
                            "role": "assistant",
                            "content": ("<|tts_bos|>" + answer + "<|tts_eos|>")
                            if speech
                            else answer,
                        }
                    ],
                    False,
                )
            )
        )
        if not inference:
            if not torch.equal(inputs["input_ids"][:, :prompt_length], prompt["input_ids"]):
                raise OmniFactoryError(
                    "E_TEMPLATE_BOUNDARY", "prompt must be an exact token prefix"
                )
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_length] = -100
            if not (labels[:, 1:] != -100).any():
                raise OmniFactoryError("E_EMPTY_LABELS", "no causal target tokens")
            inputs["labels"] = labels
        length = inputs["input_ids"].shape[1]
        if length > self.max_length:
            raise OmniFactoryError(
                "E_SEQUENCE_BUDGET", f"{length} > {self.max_length}; no truncation"
            )
        frames = inputs["audio_features"]
        sizes = inputs["tgt_sizes"][0]
        # Native vision pads slices within one sample; different samples encode separately.
        patches = len(sizes) * int((sizes[:, 0] * sizes[:, 1]).max()) if len(sizes) else 0
        sequences = len(frames)
        frame_length = frames.shape[-1] if sequences else 0
        if any(
            int(n) < 2 * self.processor.pool_step - 1
            for ns in inputs["audio_feature_lens"]
            for n in ns
        ):
            raise OmniFactoryError("E_AUDIO", "audio clip is too short for native pooling")
        codec_count = 0
        metadata = {"video": self.last_media_metadata} if self.last_media_metadata else {}
        if audio:
            metadata["audio_durations_us"] = [round(len(clip) / 16000 * 1e6) for clip in audio]
        if speech and not inference:
            from omnifactory.adapters.speech import S3Tokenizer, load_target_audio

            texts = [b for b in selected if b.type == "text"]
            targets = [b for b in selected if b.type == "audio"]
            if (
                len(texts) != 1
                or len(targets) != 1
                or not any(
                    p.text_block_id == texts[0].id and p.audio_block_id == targets[0].id
                    for p in sample.speech_pairs
                )
            ):
                raise OmniFactoryError(
                    "E_SPEECH_PAIR", "speech prediction requires one selected transcript/audio pair"
                )
            if targets[0].placement is not None:
                raise OmniFactoryError(
                    "E_SPEECH_TIMELINE", "target speech uses utterance supervision"
                )
            waveform, target_bytes = load_target_audio(
                self.resolver,
                sample.assets[targets[0].media.asset_id],
                targets[0].media,
                max_seconds=self.max_audio_seconds,
                max_bytes=self.max_decoded_bytes - decoded,
            )
            decoded += target_bytes
            cfg = self.training_config
            if self.codec is None:
                self.codec = S3Tokenizer(cfg.speech_tokenizer, cfg.speech_tokenizer_sha256)
            codes = self.codec.encode(waveform)
            if len(codes) > cfg.max_codec_tokens:
                raise OmniFactoryError("E_CODEC_BUDGET", "target speech exceeds codec token budget")
            # S3 vocabulary 0..6560, native TTS EOS = 6561.
            inputs["codec_labels"] = [torch.cat([codes, torch.tensor([6561])])]
            eos = self.processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>")
            end = (inputs["input_ids"][0, prompt_length:] == eos).nonzero().flatten()
            if not len(end) or int(end[0]) == 0:
                raise OmniFactoryError("E_TTS_SPAN", "empty or missing spoken-text span")
            inputs["speech_spans"] = [torch.tensor([prompt_length, prompt_length + int(end[0])])]
            codec_count = len(codes) + 1
            metadata["codec"] = {
                "sha256": cfg.speech_tokenizer_sha256,
                "tokens": len(codes),
                "eos": 6561,
            }
        cost = SampleCost(
            sample_id=sample.id,
            prediction_id=prediction_id,
            contract_id=contract_id,
            modalities=modalities,
            llm_tokens=length,
            vision_patches=patches,
            audio_frames=sequences * frame_length,
            audio_sequences=sequences,
            audio_sequence_length=frame_length,
            decoded_media_bytes=decoded,
            codec_frames=codec_count,
        )
        tensors, layout = pack_tree(inputs)
        encoded = EncodedExample(tensors, cost, prompt_length, answer, layout, metadata or None)
        if not inference:
            cfg = self.training_config if speech else None
            encoded.with_supervision(
                *discrete_supervision(
                    inputs["labels"][:, 1:],
                    codec_labels=torch.cat(inputs["codec_labels"]) if speech else None,
                    codec_vocabularies=(6562,),
                    text_weight=cfg.text_loss_weight if speech else 1.0,
                    codec_weight=cfg.speech_loss_weight if speech else 1.0,
                )
            )
        if self.training_config and self.training_config.duplex_schedule is not None:
            if inference:
                raise OmniFactoryError(
                    "E_DUPLEX_GENERATION",
                    "timed teacher-forced attention requires a live event scheduler for generation",
                )
            from omnifactory.adapters.duplex import bind_from_manifest

            encoded = bind_from_manifest(
                encoded,
                self.training_config.duplex_schedule,
                max_pairs=self.training_config.max_attention_pairs,
            )
        return encoded

    def collate(self, rows):
        if any("duplex_query_us" in r for r in rows):
            if not all("duplex_query_us" in r for r in rows):
                raise OmniFactoryError("E_BATCH_FIELDS", "cannot mix timed and untimed rows")
            return {"examples": rows}
        length = max(r["input_ids"].shape[1] for r in rows)
        result = {}
        for key, fill in (
            ("input_ids", self.processor.tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("labels", -100),
        ):
            if key not in rows[0]:
                if any(key in r for r in rows):
                    raise OmniFactoryError("E_BATCH_FIELDS", f"inconsistent {key}")
                continue
            if any(key not in r for r in rows):
                raise OmniFactoryError("E_BATCH_FIELDS", f"inconsistent {key}")
            result[key] = torch.cat(
                [
                    torch.nn.functional.pad(r[key], (length - r[key].shape[1], 0), value=fill)
                    for r in rows
                ]
            )
        for key in ("pixel_values", "tgt_sizes", "audio_feature_lens"):
            result[key] = [r[key][0] for r in rows]
        for key in ("image_bound", "audio_bounds", "spk_bounds"):
            result[key] = [r[key][0] + length - r["input_ids"].shape[1] for r in rows]
        features = [r["audio_features"] for r in rows if len(r["audio_features"])]
        if features:
            longest = max(f.shape[-1] for f in features)
            result["audio_features"] = torch.cat(
                [torch.nn.functional.pad(f, (0, longest - f.shape[-1])) for f in features]
            )
        else:
            result["audio_features"] = []
        if "codec_labels" in rows[0]:
            result["codec_labels"] = [r["codec_labels"][0] for r in rows]
            result["speech_spans"] = [
                r["speech_spans"][0] + length - r["input_ids"].shape[1] for r in rows
            ]
        return result

    def embeddings(self, model, batch):
        # Keep native image scatter, but bound vision padding to one sample.
        embeddings = []
        for i in range(len(batch["input_ids"])):
            row = {
                "input_ids": batch["input_ids"][i : i + 1],
                **{k: [batch[k][i]] for k in ("pixel_values", "tgt_sizes", "image_bound")},
            }
            if not row["pixel_values"][0]:
                row["vision_hidden_states"] = [[]]  # Avoid native dummy-image training work.
            embeddings.append(model.get_vllm_embedding(row)[0])
        embeddings = torch.cat(embeddings)
        clip_index = 0
        # Encode each clip without other clips' padding: exact native singleton semantics.
        for i, lengths in enumerate(batch["audio_feature_lens"]):
            bounds = batch["audio_bounds"][i]
            if len(bounds) != len(lengths):
                raise OmniFactoryError("E_AUDIO_BOUNDS", "audio clips and token spans differ")
            for n, bound in zip(lengths, bounds):
                features = batch["audio_features"][clip_index : clip_index + 1, :, : int(n)]
                encoded = model.get_audio_embedding(
                    {"audio_features": features, "audio_feature_lens": [n.reshape(1)]}, dummy=False
                )[0][0]
                start, end = map(int, bound)
                if end - start != len(encoded):
                    raise OmniFactoryError(
                        "E_AUDIO_BOUNDS", "pooled features do not match token span"
                    )
                # CopySlices preserves projector gradients through the frozen language model.
                embeddings[i, start:end] = encoded.to(embeddings.dtype)
                clip_index += 1
        if clip_index != len(batch["audio_features"]):
            raise OmniFactoryError("E_AUDIO_BOUNDS", "unconsumed audio features")
        return embeddings

    def forward(self, model, batch):
        if "examples" in batch:
            outputs = [self.forward(model, row) for row in batch["examples"]]
            return attach_losses(SimpleNamespace(), LossBundle.combine([o.losses for o in outputs]))
        mask = batch["attention_mask"]
        positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask.bool(), 0)
        if "duplex_query_us" in batch:
            from omnifactory.adapters.duplex import native_attention

            mask, positions = native_attention(
                batch,
                dtype=next(model.llm.parameters()).dtype,
                max_pairs=self.training_config.max_attention_pairs
                if self.training_config
                else 1_000_000,
            )
        output = model.llm(
            inputs_embeds=self.embeddings(model, batch),
            attention_mask=mask,
            position_ids=positions,
            labels=batch.get("labels"),
            output_hidden_states="codec_labels" in batch,
            use_cache=False,
        )
        if "codec_labels" in batch:
            from omnifactory.adapters.speech import speech_loss

            output["text_loss"] = output.loss
            output.loss, codec_loss = speech_loss(
                model,
                batch,
                output.hidden_states[-1],
                text_loss=output.loss,
                text_count=int((batch["labels"][:, 1:] != -100).sum()),
                text_weight=self.training_config.text_loss_weight,
                speech_weight=self.training_config.speech_loss_weight,
            )
            output["codec_loss"] = codec_loss
        count = int((batch["labels"][:, 1:] != -100).sum())
        speech = "codec_labels" in batch
        codec_count = sum(len(c) for c in batch.get("codec_labels", []))
        return attach_losses(
            output,
            symbol_losses(
                (output.text_loss if speech else output.loss) * count,
                count,
                codec_sum=output.codec_loss * codec_count if speech else None,
                codec_count=codec_count,
                text_weight=self.training_config.text_loss_weight if speech else 1.0,
                codec_weight=self.training_config.speech_loss_weight if speech else 1.0,
            ),
        )

    def generate_ids(self, model, batch, *, max_new_tokens):
        if "duplex_query_us" in batch:
            raise OmniFactoryError(
                "E_DUPLEX_GENERATION", "timed generation requires a live event/KV scheduler"
            )
        # Transformers' native Qwen3 generation supports modern Cache APIs. OpenBMB's
        # instance override uses removed seen_tokens; replace only this instance method.
        from transformers import Qwen3ForCausalLM

        model.llm.prepare_inputs_for_generation = (
            Qwen3ForCausalLM.prepare_inputs_for_generation.__get__(model.llm)
        )
        result = model.llm.generate(
            inputs_embeds=self.embeddings(model, batch),
            attention_mask=batch["attention_mask"],
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=[
                self.processor.tokenizer.eos_token_id,
                self.processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>"),
            ]
            if self.task == "speech_sft"
            else self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        return result

    def generate(self, model, batch, *, max_new_tokens):
        result = self.generate_ids(model, batch, max_new_tokens=max_new_tokens)
        return self.processor.tokenizer.batch_decode(result, skip_special_tokens=True)

    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None):
        from omnifactory.adapters.token2wav import NativeToken2Wav

        if reference_audio is None:
            raise OmniFactoryError(
                "E_SPEECH_INFERENCE", "MiniCPM requires explicit reference_audio"
            )
        assets = speech_assets or Path(self.training_config.model) / "assets/token2wav"
        decoder = NativeToken2Wav(assets)
        return lambda codes: decoder.decode(codes, reference_audio)

    def generate_speech(self, model, batch, *, max_new_tokens, max_codec_tokens):
        from omnifactory.adapters.speech import generate_codes, speech_condition

        if self.task != "speech_sft" or not hasattr(model, "tts") or len(batch["input_ids"]) != 1:
            raise OmniFactoryError(
                "E_SPEECH_MODEL", "speech inference requires one prompt and a speech-trained export"
            )
        generated = self.generate_ids(model, batch, max_new_tokens=max_new_tokens)[0]
        stops = {
            self.processor.tokenizer.eos_token_id,
            self.processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>"),
        }
        finished = len(generated) > 0 and int(generated[-1]) in stops
        if finished:
            generated = generated[:-1]
        if not len(generated):
            raise OmniFactoryError("E_EMPTY_SPEECH", "model generated no spoken text")
        # Recompute complete generated-text hidden states, including the final spoken token.
        prompt_embeddings = self.embeddings(model, batch)
        full = torch.cat(
            [prompt_embeddings, model.llm.get_input_embeddings()(generated).unsqueeze(0)], dim=1
        )
        mask = torch.cat(
            [
                batch["attention_mask"],
                torch.ones(
                    (1, len(generated)), device=full.device, dtype=batch["attention_mask"].dtype
                ),
            ],
            dim=1,
        )
        hidden = model.llm(
            inputs_embeds=full, attention_mask=mask, output_hidden_states=True, use_cache=False
        ).hidden_states[-1][0, -len(generated) :]
        condition = speech_condition(model.tts, generated, hidden)
        codes, speech_finished = generate_codes(model.tts, condition, max_codec_tokens)
        return {
            "text": self.processor.tokenizer.decode(generated, skip_special_tokens=True),
            "codec_tokens": codes,
            "text_finished": finished,
            "speech_finished": speech_finished,
        }


def _whisper_attention_compat(self, *args, **kwargs):
    """Bridge the pinned native encoder's legacy tuple to Transformers 4.57."""
    from transformers.models.whisper.modeling_whisper import WhisperAttention

    past = kwargs.pop("past_key_value", None)
    result = WhisperAttention.forward(self, *args, past_key_values=past, **kwargs)
    return (*result, past)


def configure_native_runtime(model):
    for layer in model.apm.layers:
        layer.self_attn.forward = _whisper_attention_compat.__get__(layer.self_attn)
    return model


def load_minicpm(config):
    if config.quantization != "none":
        raise OmniFactoryError(
            "E_QUANTIZATION", "MiniCPM NF4 is not validated; use unquantized LoRA/projector SFT"
        )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise OmniFactoryError("E_DEVICE", "CUDA was requested but is unavailable")
    from transformers import PretrainedConfig

    raw, _ = PretrainedConfig.get_config_dict(config.model, revision=config.revision)
    if raw.get("model_type") != "minicpmo" or str(raw.get("version", "")) != "4.5":
        raise OmniFactoryError("E_MODEL_TYPE", "expected MiniCPM-o 4.5 config")
    native = MiniCPMOConfig.from_dict(raw)
    native.init_tts = config.task == "speech_sft"
    if native.init_tts and (
        native.tts_config.num_audio_tokens != 6562 or native.tts_config.num_vq != 1
    ):
        raise OmniFactoryError("E_TTS_CONFIG", "speech SFT requires native 6562-token S3 TTS")
    native.use_cache = False
    model, info = MiniCPMO.from_pretrained(
        config.model,
        revision=config.revision,
        config=native,
        dtype=getattr(torch, config.dtype),
        attn_implementation=config.attention,
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise OmniFactoryError("E_CHECKPOINT_INCOMPLETE", json.dumps(info))
    model.to(config.device)
    from omnifactory.adapters.minicpm_precision import configure_minicpm_precision

    return configure_minicpm_precision(configure_native_runtime(model), config)
