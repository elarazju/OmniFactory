"""Native Qwen2.5-Omni video understanding and codec-supervised Talker SFT."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
from omnifactory.adapters.media import MediaInputs
from omnifactory.adapters.supervision import discrete_supervision
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import CanonicalSample
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import SampleCost
from omnifactory.training.config import TrainConfig
from omnifactory.training.losses import LossBundle, attach_losses, symbol_losses


class Qwen25Adapter(MediaInputs):
    processor_class = Qwen2_5OmniProcessor
    codec_vocabularies = (8448,)
    codec_axes = None

    def load_code_labels(self):
        from omnifactory.adapters.qwen_speech import NativeCodeLabels

        return NativeCodeLabels(self.training_config.speech_codes)

    def __init__(
        self,
        processor,
        *,
        root: Path,
        max_length: int = 2048,
        max_audio_seconds: float = 30,
        max_image_pixels: int = 1048576,
        max_decoded_bytes: int = 67108864,
    ):
        self.processor = processor
        self.resolver = LocalMediaResolver(root)
        self.max_length = max_length
        self.max_audio_seconds = min(max_audio_seconds, processor.feature_extractor.chunk_length)
        self.max_image_pixels = max_image_pixels
        self.max_decoded_bytes = max_decoded_bytes

    @classmethod
    def from_config(cls, config: TrainConfig):
        processor = cls.processor_class.from_pretrained(
            config.model, revision=config.revision, use_fast=False
        )
        adapter = cls(
            processor,
            root=config.dataset_root or config.dataset.parent,
            max_length=config.max_length,
            max_audio_seconds=config.max_audio_seconds,
            max_image_pixels=config.max_image_pixels,
            max_decoded_bytes=config.budget.max_decoded_media_bytes,
        )

        adapter.task = config.task
        adapter.training_config = config
        adapter.native_video = True
        adapter.video_options = {
            "fps": config.video_fps,
            "max_frames": config.max_video_frames,
            "max_seconds": config.max_video_seconds,
        }
        adapter.code_labels = None
        return adapter

    def encode(
        self,
        sample: CanonicalSample,
        prediction_id: str,
        *,
        contract_id: str,
        inference: bool = False,
    ) -> EncodedExample:
        messages, images, audio, decoded, modalities = self._messages(sample, prediction_id)
        if messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                        }
                    ],
                },
            )
        prediction = next(p for p in sample.predictions if p.id == prediction_id)
        target = next(m for m in sample.messages if m.id == prediction.target_message_id)
        selected = [b for b in target.content if b.id in prediction.target_block_ids]
        speech = getattr(self, "task", "understanding_sft") == "speech_sft"
        if not speech and any(b.type != "text" for b in selected):
            raise OmniFactoryError(
                "E_RUNTIME_TARGET", "understanding SFT requires text-only selected targets"
            )
        answer = "\n".join(b.text for b in selected if b.type == "text")
        if not inference and any(
            token in answer for token in self.processor.tokenizer.all_special_tokens
        ):
            raise OmniFactoryError(
                "E_CONTROL_TOKEN", "literal special tokens in targets are unsupported"
            )
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        media = {"images": images or None, "audio": audio or None}
        if self.last_videos:
            if len(set(self.last_video_audio_modes)) != 1:
                raise OmniFactoryError(
                    "E_VIDEO_AUDIO_MODE",
                    "Qwen requires a common audio_mode for videos within one prediction",
                )
            media["videos"] = self.last_videos
            media["videos_kwargs"] = {
                "fps": self.video_options["fps"],
                "do_sample_frames": False,
                "use_audio_in_video": self.last_video_audio_modes[0],
                "size": {"shortest_edge": 56 * 56, "longest_edge": self.max_image_pixels},
            }
        if audio:
            hop = self.processor.feature_extractor.hop_length
            samples = ((max(len(a) for a in audio) + hop - 1) // hop) * hop
            # STFT uses centered windows. Retain a zero tail so valid frames match
            # the official long-padded extractor instead of reflecting the waveform.
            tail = (self.processor.feature_extractor.n_fft // 2 + hop - 1) // hop
            samples += tail * hop
            media["audio_kwargs"] = {"max_length": samples, "truncation": False}

        def encode_text(text):
            # ProcessorMixin mutates nested kwargs; do not share them between calls.
            kwargs = {k: deepcopy(v) if k.endswith("_kwargs") else v for k, v in media.items()}
            return self.processor(text=text, **kwargs, return_tensors="pt", padding=False)

        prompt_inputs = encode_text(prompt)
        prompt_length = prompt_inputs["input_ids"].shape[1]
        if inference:
            inputs = dict(prompt_inputs)
        else:
            full_messages = messages + [
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            full = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            inputs = dict(encode_text(full))
            if not torch.equal(inputs["input_ids"][:, :prompt_length], prompt_inputs["input_ids"]):
                raise OmniFactoryError(
                    "E_TEMPLATE_BOUNDARY",
                    "prompt is not an exact token prefix of the supervised sequence",
                )
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_length] = -100
            if not (labels[:, 1:] != -100).any():
                raise OmniFactoryError("E_EMPTY_LABELS", "no causal target tokens")
            inputs["labels"] = labels
        inputs.pop("token_type_ids", None)
        if self.last_videos:
            inputs["use_audio_in_video"] = self.last_video_audio_modes[0]
        metadata = {"video": self.last_media_metadata}
        codec_frames = 0
        if speech and not inference:
            if self.code_labels is None:
                self.code_labels = self.load_code_labels()
            codes, provenance = self.code_labels.get(
                sample,
                prediction_id,
                self.resolver,
                self.training_config.speech_speaker,
                self.training_config.max_codec_tokens,
            )
            stop = (
                inputs["input_ids"][0, prompt_length:] == self.processor.tokenizer.eos_token_id
            ).nonzero()
            if not len(stop) or int(stop[0]) == 0:
                raise OmniFactoryError(
                    "E_SPEECH_SPAN", "transcript must end at the native assistant boundary"
                )
            inputs["codec_labels"] = [codes]
            inputs["speech_spans"] = [torch.tensor([prompt_length, prompt_length + int(stop[0])])]
            codec_frames = len(codes)
            metadata["speech"] = provenance
        length = inputs["input_ids"].shape[1]
        if length > self.max_length:
            raise OmniFactoryError(
                "E_SEQUENCE_BUDGET",
                f"{sample.id}/{prediction_id}: {length} > {self.max_length}; no truncation applied",
            )
        frames = inputs.get("input_features")
        patches = inputs.get("pixel_values")
        video_patches = inputs.get("pixel_values_videos")
        cost = SampleCost(
            sample_id=sample.id,
            prediction_id=prediction_id,
            contract_id=contract_id,
            modalities=modalities,
            llm_tokens=length,
            vision_patches=(0 if patches is None else patches.shape[0])
            + (0 if video_patches is None else video_patches.shape[0]),
            codec_frames=codec_frames,
            audio_frames=0 if frames is None else frames.shape[0] * frames.shape[-1],
            audio_sequences=0 if frames is None else frames.shape[0],
            audio_sequence_length=0 if frames is None else frames.shape[-1],
            decoded_media_bytes=decoded,
        )
        tensors, layout = pack_tree(inputs)
        encoded = EncodedExample(tensors, cost, prompt_length, answer, layout, metadata)
        if not inference:
            cfg = self.training_config if speech else None
            encoded.with_supervision(
                *discrete_supervision(
                    inputs["labels"][:, 1:],
                    codec_labels=torch.cat(inputs["codec_labels"]) if speech else None,
                    codec_vocabularies=self.codec_vocabularies,
                    codec_axes=self.codec_axes,
                    text_weight=cfg.text_loss_weight if speech else 1.0,
                    codec_weight=cfg.speech_loss_weight if speech else 1.0,
                )
            )
        return encoded

    def collate(self, rows: list[dict[str, torch.Tensor]]):
        if getattr(getattr(self, "training_config", None), "packing", "none") != "none" and not any(
            "codec_labels" in row or "pixel_values_videos" in row for row in rows
        ):
            return {
                "packing_rows": rows,
                "packing_lengths": [r["input_ids"].shape[1] for r in rows],
            }
        return self.collate_padded(rows)

    def collate_padded(self, rows):
        if any("codec_labels" in row or "pixel_values_videos" in row for row in rows):
            # Native video flags and dual-track sequences are per-example. Sequential
            # execution retains exact gradients without falsely claiming packed attention.
            result = {"examples": rows}
            if "codec_labels" in rows[0]:
                result["codec_labels"] = [row["codec_labels"][0] for row in rows]
            return result
        max_length = max(row["input_ids"].shape[1] for row in rows)
        result = {}
        for key in set().union(*(row.keys() for row in rows)):
            values = [row[key] for row in rows if key in row]
            if key in ("input_ids", "attention_mask", "labels"):
                if len(values) != len(rows):
                    raise OmniFactoryError("E_BATCH_FIELDS", f"inconsistent {key}")
                fill = {
                    "input_ids": self.processor.tokenizer.pad_token_id,
                    "attention_mask": 0,
                    "labels": -100,
                }[key]
                values = [
                    torch.nn.functional.pad(v, (max_length - v.shape[1], 0), value=fill)
                    for v in values
                ]
            elif key in ("input_features", "feature_attention_mask"):
                longest = max(v.shape[-1] for v in values)
                values = [torch.nn.functional.pad(v, (0, longest - v.shape[-1])) for v in values]
            elif key not in (
                "input_features",
                "feature_attention_mask",
                "pixel_values",
                "image_grid_thw",
            ):
                raise OmniFactoryError("E_BATCH_FIELDS", f"unsupported processor output: {key}")
            result[key] = torch.cat(values, dim=0)
        return result

    @staticmethod
    def to_device(batch, model, config):
        thinker = getattr(model, "thinker", model)
        if "packing_rows" in batch:
            return {
                **batch,
                "packing_rows": [
                    Qwen25Adapter.to_device(r, model, config) for r in batch["packing_rows"]
                ],
            }
        if "examples" in batch:
            return {
                **batch,
                "examples": [
                    Qwen25Adapter.to_device(row, model, config) for row in batch["examples"]
                ],
            }
        return {
            k: move_tree(
                v,
                device=config.device,
                dtype=thinker.audio_tower.dtype if k == "input_features" else thinker.visual.dtype,
            )
            for k, v in batch.items()
        }

    def save_processor(self, path):
        from omnifactory.training.artifacts import save_processor

        save_processor(self.processor, path)

    def forward(self, model, batch):
        if "packing_rows" in batch:
            from omnifactory.adapters.packing import packed_batch

            batch = packed_batch(getattr(model, "thinker", model), batch["packing_rows"])
        if "examples" in batch:
            results = [self.forward(model, row) for row in batch["examples"]]
            codec_counts = [
                sum(len(c) for c in row.get("codec_labels", [])) for row in batch["examples"]
            ]
            codec_loss = (
                sum(result.codec_loss * n for result, n in zip(results, codec_counts))
                / sum(codec_counts)
                if sum(codec_counts)
                else None
            )
            return attach_losses(
                SimpleNamespace(codec_loss=codec_loss),
                LossBundle.combine([r.losses for r in results]),
            )
        thinker = getattr(model, "thinker", model)
        kwargs = {k: v for k, v in batch.items() if k not in ("codec_labels", "speech_spans")}
        output = thinker(**kwargs, use_cache=False, output_hidden_states="codec_labels" in batch)
        if "codec_labels" in batch:
            from omnifactory.adapters.qwen_speech import talker_condition, talker_logits

            start, end = map(int, batch["speech_spans"][0])
            condition = talker_condition(model, batch, output, start, end)
            codes = batch["codec_labels"][0]
            logits = talker_logits(model, condition, codes)
            codec_sum = torch.nn.functional.cross_entropy(logits[0].float(), codes, reduction="sum")
            output.text_loss = output.loss
            output.codec_loss = codec_sum / len(codes)
        text_count = int((batch["labels"][:, 1:] != -100).sum())
        speech = "codec_labels" in batch
        return attach_losses(
            output,
            symbol_losses(
                (output.text_loss if speech else output.loss) * text_count,
                text_count,
                codec_sum=codec_sum if speech else None,
                codec_count=len(codes) if speech else 0,
                text_weight=self.training_config.text_loss_weight if speech else 1.0,
                codec_weight=self.training_config.speech_loss_weight if speech else 1.0,
            ),
        )

    def generate_ids(self, model, batch, *, max_new_tokens):
        thinker = getattr(model, "thinker", model)
        return thinker.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

    def generate(self, model, batch, *, max_new_tokens):
        generated = self.generate_ids(model, batch, max_new_tokens=max_new_tokens)
        return self.processor.tokenizer.batch_decode(
            generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def generate_speech(self, model, batch, *, max_new_tokens, max_codec_tokens):
        ids = self.generate_ids(model, batch, max_new_tokens=max_new_tokens)
        start = batch["input_ids"].shape[1]
        new = ids[0, start:]
        eos = (new == self.processor.tokenizer.eos_token_id).nonzero()
        end = start + (int(eos[0]) if len(eos) else len(new))
        full = {**batch, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
        output = model.thinker(**full, use_cache=False, output_hidden_states=True)
        from omnifactory.adapters.qwen_speech import generate_talker, talker_condition

        codes, end_token = generate_talker(
            model,
            talker_condition(model, full, output, start, end),
            max_codec_tokens,
            do_sample=True,
        )
        return {
            "text": self.processor.tokenizer.decode(ids[0, start:end], skip_special_tokens=True),
            "codec_tokens": codes,
            "text_finished": bool(len(eos)),
            "speech_finished": end_token is not None,
            "codec_eos_token_id": end_token,
            "speaker": model.speech_speaker,
        }

    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None):
        if speech_assets is not None or reference_audio is not None:
            raise OmniFactoryError(
                "E_SPEECH_INFERENCE",
                "Qwen uses checkpoint speakers; arbitrary reference voice cloning is unsupported",
            )
        from omnifactory.adapters.qwen_speech import make_decoder

        return make_decoder(self.training_config)


def load_thinker(config: TrainConfig):
    if config.packing != "none":
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except (ImportError, OSError) as exc:
            raise OmniFactoryError(
                "E_PACKING_INSTALL", "packing requires a FlashAttention build matching PyTorch/CUDA"
            ) from exc
    if config.device == "cuda" and not torch.cuda.is_available():
        raise OmniFactoryError("E_DEVICE", "CUDA was requested but is unavailable")
    if (
        config.device == "cuda"
        and config.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise OmniFactoryError("E_DEVICE", "this CUDA device does not support bfloat16")
    if config.quantization == "nf4":
        import importlib.util

        if importlib.util.find_spec("bitsandbytes") is None:
            raise OmniFactoryError(
                "E_QUANTIZATION_INSTALL",
                "install NF4 dependencies with: uv sync --extra train --extra cuda",
            )
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise OmniFactoryError("E_DEVICE", "MPS was requested but is unavailable")
    kwargs = {
        "revision": config.revision,
        "dtype": getattr(torch, config.dtype),
        "attn_implementation": config.attention,
    }
    if config.quantization == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": "cuda:0"},
        )
    from transformers import PretrainedConfig, Qwen2_5OmniThinkerConfig

    source_config, _ = PretrainedConfig.get_config_dict(config.model, revision=config.revision)
    if source_config.get("model_type") not in ("qwen2_5_omni", "qwen2_5_omni_thinker"):
        raise OmniFactoryError("E_MODEL_TYPE", "this runtime requires Qwen2.5-Omni weights")
    thinker_config = Qwen2_5OmniThinkerConfig.from_dict(
        source_config.get("thinker_config", source_config)
    )
    model, loading = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        config.model, config=thinker_config, output_loading_info=True, **kwargs
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise OmniFactoryError("E_CHECKPOINT_INCOMPLETE", str(loading))
    if config.quantization == "none":
        model.to(config.device)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model
