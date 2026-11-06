"""MiMo-Audio canonical conversations → native [batch, codebooks+text, frames]."""

from pathlib import Path

import torch
from transformers import AutoTokenizer

from omnifactory._vendor.mimo_audio.modeling_mimo_audio import MiMoAudioConfig, MiMoSampler
from omnifactory._vendor.mimo_audio.process_speechdata import InputSegment, StreamingInputSegment
from omnifactory.adapters.base import EncodedExample, move_tree
from omnifactory.adapters.speech import load_target_audio
from omnifactory.adapters.supervision import discrete_supervision
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import SampleCost
from omnifactory.training.losses import attach_losses, symbol_losses


class MiMoAdapter:
    def __init__(self, tokenizer, native, config):
        self.processor = tokenizer
        self.native = native
        self.config = config
        self.resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        self.codec = None
        self.codec_path = Path(config.model) / "assets/mimo_codec"
        if not self.codec_path.is_dir():
            self.codec_path = config.speech_tokenizer
        self.empty = tokenizer.convert_tokens_to_ids("<|empty|>")
        self.g = native.group_size
        self.channels = native.audio_channels
        self.zero = native.parsed_speech_empty_ids()
        self.specials = specials = (
            "<|sosp|>",
            "<|eosp|>",
            "<|sostm|>",
            "<|eostm|>",
            "<|eot|>",
            "<|empty|>",
            "<|im_start|>",
            "<|im_end|>",
        )
        if any(t not in tokenizer.get_vocab() for t in specials):
            raise OmniFactoryError("E_MIMO_TOKENIZER", "missing native special tokens")
        if len(self.zero) != self.channels or self.g < 1:
            raise OmniFactoryError("E_MIMO_CONFIG", "inconsistent native codec dimensions")

    @classmethod
    def from_config(cls, config):
        return cls(
            AutoTokenizer.from_pretrained(config.model, revision=config.revision),
            MiMoAudioConfig.from_pretrained(config.model, revision=config.revision),
            config,
        )

    def _codec(self):
        if self.codec is None:
            if self.codec_path is None:
                raise OmniFactoryError(
                    "E_MIMO_CODEC", "audio requires a local MiMo codec checkpoint"
                )
            from omnifactory.adapters.mimo_codec import MiMoCodec
            from omnifactory.training.prepare import source_fingerprint

            if (
                source_fingerprint(str(self.codec_path), None)
                != self.config.speech_tokenizer_sha256
            ):
                raise OmniFactoryError("E_CODEC_HASH", "MiMo codec directory fingerprint differs")
            self.codec = MiMoCodec(self.codec_path, self.channels)
            sizes = self.codec.config.codebook_size
            if len(sizes) == 1:
                sizes = sizes * self.codec.config.num_quantizers
            if sizes[: self.channels] != self.zero:
                raise OmniFactoryError("E_CODEC_TYPE", "codec codebooks differ from MiMo model")
        return self.codec

    def _text(self, text):
        return (
            InputSegment(text=text, speech_zeroemb_idx=self.zero, text_zeroemb_idx=self.empty)
            .to_input_id(self.processor, self.g, self.channels)
            .long()
        )

    def _check_text(self, text):
        if any(t in text for t in (*self.processor.all_special_tokens, *self.specials)):
            raise OmniFactoryError("E_CONTROL_TOKEN", "literal control tokens in content")

    def _audio(self, sample, block, remaining):
        if block.placement is not None:
            raise OmniFactoryError("E_RUNTIME_MODALITY", "aligned/duplex audio not implemented")
        codec = self._codec()
        audio, size = load_target_audio(
            self.resolver,
            sample.assets[block.media.asset_id],
            block.media,
            max_seconds=self.config.max_audio_seconds,
            max_bytes=remaining,
            target_rate=codec.config.sampling_rate,
        )
        codes = codec.encode(audio, codec.config.sampling_rate)
        if codes.ndim != 2 or codes.shape[1] != self.channels or not len(codes):
            raise OmniFactoryError("E_CODEC_SHAPE", "expected [frames, codebooks]")
        for i, limit in enumerate(self.zero):
            if (codes[:, i] < 0).any() or (codes[:, i] >= limit).any():
                raise OmniFactoryError("E_CODEC_RANGE", "codec index outside native vocabulary")
        original = len(codes)
        if len(codes) % self.g:
            codes = torch.cat([codes, codes[-1:].repeat(self.g - len(codes) % self.g, 1)])
        return codes, size, original

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        view = inference_view(sample, prediction_id, requested_outputs=("text", "audio"))
        if view.conditions:
            raise OmniFactoryError("E_RUNTIME_CONDITION", "MiMo condition recipes not implemented")
        segments, decoded, audio_frames = [], 0, 0
        modalities = {"text"}
        audio_metadata = []
        for message in view.messages:
            segments.append(self._text(f"<|im_start|>{message.role}\n"))
            for block in message.content:
                if block.type == "text":
                    self._check_text(block.text)
                    segments.append(self._text(block.text))
                elif block.type == "audio" and message.role != "assistant":
                    codes, size, frames = self._audio(
                        sample, block, self.config.budget.max_decoded_media_bytes - decoded
                    )
                    decoded += size
                    audio_frames += len(codes)
                    modalities.add("audio")
                    segments.append(
                        InputSegment(
                            audio=codes.flatten(),
                            speech_zeroemb_idx=self.zero,
                            text_zeroemb_idx=self.empty,
                        )
                        .to_input_id(self.processor, self.g, self.channels)
                        .long()
                    )
                    audio_metadata.append(
                        {"block_id": block.id, "frames": frames, "padded_frames": len(codes)}
                    )
                else:
                    raise OmniFactoryError(
                        "E_RUNTIME_MODALITY",
                        "MiMo v1 supports text/audio inputs; assistant audio history requires an explicit interleaving recipe",
                    )
            segments.append(self._text("<|im_end|>\n"))
        # Native spoken dialogue starts in the interleaved speech stream. The
        # understanding-only thinking prefix otherwise yields text without codes
        # on the pretrained model, even when audio sampling is enabled.
        speech_mode = self.config.task == "speech_sft"
        assistant_prefix = "<|sostm|>" if speech_mode else "<think>\n\n</think>\n"
        segments.append(self._text("<|im_start|>assistant\n" + assistant_prefix))
        prompt_frames = sum(s.shape[1] for s in segments)
        answer, codec_frames = "", 0
        if not inference:
            prediction = next(p for p in sample.predictions if p.id == prediction_id)
            target = next(m for m in sample.messages if m.id == prediction.target_message_id)
            blocks = [b for b in target.content if b.id in prediction.target_block_ids]
            texts, audios = (
                [b for b in blocks if b.type == "text"],
                [b for b in blocks if b.type == "audio"],
            )
            if any(b.type not in ("text", "audio") for b in blocks) or len(texts) != 1:
                raise OmniFactoryError("E_TARGET", "MiMo requires exactly one text target")
            answer = texts[0].text
            self._check_text(answer)
            if self.config.task == "speech_sft":
                if len(audios) != 1 or not any(
                    p.text_block_id == texts[0].id and p.audio_block_id == audios[0].id
                    for p in sample.speech_pairs
                ):
                    raise OmniFactoryError(
                        "E_SPEECH_PAIR", "MiMo speech SFT needs one paired text/audio target"
                    )
                codes, size, frames = self._audio(
                    sample, audios[0], self.config.budget.max_decoded_media_bytes - decoded
                )
                if len(codes) > self.config.max_codec_tokens:
                    raise OmniFactoryError(
                        "E_CODEC_LENGTH", "target exceeds max_codec_tokens (frames)"
                    )
                decoded += size
                codec_frames = len(codes)
                modalities.add("audio")
                stream = StreamingInputSegment(
                    text=answer,
                    audio=codes.flatten(),
                    speech_zeroemb_idx=self.zero,
                    text_zeroemb_idx=self.empty,
                    tokenizer=self.processor,
                    group_size=self.g,
                    audio_channels=self.channels,
                ).to_input_id(self.processor, self.g, self.channels)
                # StreamingInputSegment includes sostm; it is already part of
                # the shared, target-free training/inference prompt above.
                segments.append(stream[:, self.g :])
                audio_metadata.append(
                    {"block_id": audios[0].id, "frames": frames, "padded_frames": len(codes)}
                )
            else:
                segments.append(self._text(answer))
            segments.append(self._text("<|im_end|>\n"))
        ids = torch.cat(segments, dim=1).long().unsqueeze(0)
        length = ids.shape[-1] // self.g
        if length > self.config.max_length:
            raise OmniFactoryError("E_LENGTH", "MiMo global patch sequence exceeds max_length")
        payload = {"input_ids": ids, "attention_mask": torch.ones(1, length, dtype=torch.long)}
        if not inference:
            labels = ids.clone()
            labels[:, :, :prompt_frames] = -100
            payload["labels"] = labels
        encoded = EncodedExample(
            tensors=payload,
            prompt_length=prompt_frames,
            answer=answer,
            cost=SampleCost(
                sample_id=sample.id,
                prediction_id=prediction_id,
                contract_id=contract_id,
                modalities=tuple(sorted(modalities)),
                llm_tokens=length,
                audio_frames=audio_frames,
                codec_frames=codec_frames,
                decoded_media_bytes=decoded,
            ),
            metadata={
                "representation": "discrete_multi_codebook",
                "axes": ["batch", "text_and_codebooks", "frames"],
                "codebooks": self.channels,
                "patch_frames": self.g,
                "codec_supervision_tokens": codec_frames * self.channels,
                "audio": audio_metadata,
            },
        )
        if not inference:
            text_labels = labels[:, 0, self.g :: self.g]
            codec_labels = None
            if self.config.task == "speech_sft":
                speech = text_labels == self.empty
                patches = labels[:, 1:].reshape(1, self.channels, -1, self.g).transpose(1, 2)
                codec_labels = patches[:, 1:][speech]
            encoded.with_supervision(
                *discrete_supervision(
                    text_labels,
                    codec_labels=codec_labels,
                    codec_axes=("patch", "codebook", "frame"),
                    codec_vocabularies=self.zero,
                    text_weight=self.config.text_loss_weight,
                    codec_weight=self.config.speech_loss_weight,
                )
            )
        return encoded

    def collate(self, rows):
        length = max(r["input_ids"].shape[-1] for r in rows)
        ids = self._text("<|im_end|>").repeat(1, length // self.g)[None].repeat(len(rows), 1, 1)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros(len(rows), length // self.g, dtype=torch.long)
        for i, row in enumerate(rows):
            n = row["input_ids"].shape[-1]
            ids[i, :, :n] = row["input_ids"][0]
            if "labels" in row:
                labels[i, :, :n] = row["labels"][0]
            mask[i, : n // self.g] = 1
        result = {"input_ids": ids, "attention_mask": mask}
        if all("labels" in row for row in rows):
            result["labels"] = labels
        return result

    def forward(self, model, batch):
        result = model(**batch)
        count = int((batch["labels"][:, 0, self.g :: self.g] != -100).sum())
        speech = self.config.task == "speech_sft"
        return attach_losses(
            result,
            symbol_losses(
                result.text_loss_sum,
                count,
                codec_sum=result.codec_loss_sum if speech else None,
                codec_count=result.codec_tokens,
                text_weight=self.config.text_loss_weight,
                codec_weight=self.config.speech_loss_weight,
            ),
        )

    @staticmethod
    def to_device(batch, model, config):
        return move_tree(
            batch, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
        )

    @torch.no_grad()
    def _generate(self, model, batch, *, max_new_tokens, max_codec_tokens, audio):
        ids = batch["input_ids"]
        if ids.size(0) != 1:
            raise OmniFactoryError(
                "E_GENERATION_BATCH", "MiMo generation currently needs one prompt"
            )
        g = self.g
        mask = batch["attention_mask"]
        cache, texts, codes = None, [], []
        token = self.processor.convert_tokens_to_ids
        stops = {token("<|im_end|>"), token("<|eostm|>")}
        controls = {token(t) for t in ("<|sostm|>", "<|eot|>", "<|sosp|>", "<|eosp|>")}
        finished = False
        global_sampler = MiMoSampler(do_sample=True, temperature=0.6, top_p=0.95)
        local_sampler = MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
        for _ in range(max_new_tokens):
            if mask.size(-1) >= self.config.max_length:
                break
            result = model(input_ids=ids, attention_mask=mask, past_key_values=cache)
            cache = result.past_key_values
            scores = result.text_logits[:, -1].clone()
            if not audio:
                scores[:, list(controls | {self.empty})] = -torch.inf
            next_id = int(global_sampler.sample(scores) if audio else scores.argmax(-1))
            if next_id in stops:
                finished = True
                break
            patch = self._text("<|im_end|>").to(ids.device)[None]
            patch[:, 0, 0] = next_id
            if next_id == self.empty:
                if len(codes) + g > max_codec_tokens:
                    break
                local = model.local_forward(
                    result.local_hidden_states,
                    torch.long,
                    ids.device,
                    local_sampler=local_sampler,
                )[0]
                codes.extend(local.cpu().tolist())
                patch[:, 1:] = local.T
            elif next_id not in controls:
                texts.append(next_id)
            ids = patch
            mask = torch.cat((mask, mask.new_ones(1, 1)), dim=1)
        return {
            "text": self.processor.decode(texts, skip_special_tokens=True),
            "codec_tokens": codes,
            "codec_layout": "frames_codebooks",
            "codebooks": self.channels,
            "finished": finished,
        }

    def generate(self, model, batch, *, max_new_tokens):
        return [
            self._generate(
                model, batch, max_new_tokens=max_new_tokens, max_codec_tokens=0, audio=False
            )["text"]
        ]

    def generate_speech(self, model, batch, *, max_new_tokens, max_codec_tokens):
        return self._generate(
            model,
            batch,
            max_new_tokens=max_new_tokens,
            max_codec_tokens=max_codec_tokens,
            audio=True,
        )

    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None):
        if speech_assets is not None or reference_audio is not None:
            raise OmniFactoryError(
                "E_RUNTIME_CONDITION",
                "MiMo uses its bundled codec; voice conditions not yet supported",
            )
        return self._codec()

    def save_processor(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(path)
        self.native.save_pretrained(path)
