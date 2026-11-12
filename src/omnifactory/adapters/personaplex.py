"""PersonaPlex: parallel user/agent recordings, timed text and native voice prefix."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import sentencepiece
import torch
from safetensors.torch import save_file

from omnifactory._vendor.personaplex.models.lm import SILENCE_TOKENS, SINE_TOKENS, LMGen
from omnifactory._vendor.personaplex.models.loaders import MIMI_NAME, TEXT_TOKENIZER_NAME
from omnifactory._vendor.personaplex.utils.compile import no_compile
from omnifactory.adapters.base import EncodedExample, move_tree, pack_tree
from omnifactory.adapters.personaplex_features import (
    MimiFeatures,
    aligned_text,
    load_alignments,
    read_voice,
)
from omnifactory.adapters.personaplex_model import (
    duplex_loss,
    inspect_native_weights,
    load_personaplex,
)
from omnifactory.adapters.personaplex_supervision import aligned_duplex_supervision
from omnifactory.data.media import LocalMediaResolver
from omnifactory.data.schema import inference_view
from omnifactory.data.supervision import SupervisionPlan
from omnifactory.errors import OmniFactoryError
from omnifactory.planning.batching import SampleCost
from omnifactory.training.artifacts import new_directory, write_json
from omnifactory.training.losses import LossBundle, attach_losses


def model_root(config):
    root = Path(config.model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(config.model, revision=config.revision))
    pointer = root / "personaplex_processor.json"
    if pointer.exists():
        root = Path(json.loads(pointer.read_text())["source"])
    return root


def load_model(config):
    return load_personaplex(
        model_root(config), device=config.device, dtype=getattr(torch, config.dtype)
    )


def input_context(view):
    if (
        len(view.messages) not in (1, 2)
        or view.messages[-1].role != "user"
        or len(view.messages[-1].content) != 1
        or view.messages[-1].content[0].type != "audio"
        or view.conditions
    ):
        raise OmniFactoryError(
            "E_DUPLEX_INPUT", "one complete user audio track and optional system persona required"
        )
    role = "You are a helpful assistant."
    if len(view.messages) == 2:
        system = view.messages[0]
        if system.role != "system" or any(b.type != "text" for b in system.content):
            raise OmniFactoryError("E_DUPLEX_INPUT", "persona must be literal system text")
        role = "".join(b.text for b in system.content).strip()
    if not role or "<system>" in role or "<|" in role:
        raise OmniFactoryError("E_DUPLEX_INPUT", "provide literal persona without control tags")
    block = view.messages[-1].content[0]
    validate_track(block)
    return block, f"<system> {role} <system>"


def validate_track(block):
    if (
        block.media.start_us != 0
        or block.media.end_us is not None
        or block.placement is None
        or block.placement.offset_us != 0
    ):
        raise OmniFactoryError(
            "E_DUPLEX_TRACK", "complete tracks require explicit common timeline at offset zero"
        )


def persona_prefix(tokens):
    # The native offline pipeline uses six spacer frames on each side of persona.
    text = [3] * 6 + tokens + [3] * 6
    frames = len(text)
    return torch.cat(
        [
            torch.tensor(text, dtype=torch.long).reshape(1, 1, frames),
            torch.tensor(SILENCE_TOKENS).reshape(1, 8, 1).expand(1, 8, frames),
            torch.tensor(SINE_TOKENS).reshape(1, 8, 1).expand(1, 8, frames),
        ],
        1,
    )


class PersonaPlexAdapter:
    def __init__(self, config):
        self.config, self.root = config, model_root(config)
        self.processor = sentencepiece.SentencePieceProcessor(str(self.root / TEXT_TOKENIZER_NAME))
        self.resolver = LocalMediaResolver(config.dataset_root or config.dataset.parent)
        self.features = MimiFeatures(self.root, config)
        self.voice = read_voice(self.root, config.personaplex_voice)
        self._alignments = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def save_processor(self, path):
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "personaplex_processor.json", {"source": str(self.root.resolve())})

    def encode(self, sample, prediction_id, *, contract_id, inference=False):
        view = inference_view(sample, prediction_id, ("text", "audio"))
        user, role = input_context(view)
        role_ids = list(self.processor.encode(role))
        prefix = persona_prefix(role_ids)
        wave, decoded = self.features.audio(self.resolver, view.assets[user.media.asset_id], user)
        frames = (len(wave) + 1919) // 1920
        total = frames + prefix.shape[-1] + len(self.voice["embeddings"])
        if total > self.config.max_length or total * total > self.config.max_attention_pairs:
            raise OmniFactoryError(
                "E_SEQUENCE_LENGTH", "voice, persona and session exceed attention budget"
            )
        target_wave = text = None
        if not inference:
            prediction = next(p for p in sample.predictions if p.id == prediction_id)
            target = next(m for m in sample.messages if m.id == prediction.target_message_id)
            blocks = [b for b in target.content if b.id in prediction.target_block_ids]
            audios = [b for b in blocks if b.type == "audio"]
            texts = [b for b in blocks if b.type == "text"]
            if len(blocks) != 2 or len(audios) != 1 or len(texts) != 1:
                raise OmniFactoryError("E_DUPLEX_TARGET", "select agent track and exact transcript")
            agent, text = audios[0], texts[0].text
            validate_track(agent)
            if agent.placement.timeline_id != user.placement.timeline_id or not any(
                p.text_block_id == texts[0].id and p.audio_block_id == agent.id
                for p in sample.speech_pairs
            ):
                raise OmniFactoryError(
                    "E_DUPLEX_TARGET", "paired transcript and common audio timeline required"
                )
            target_path = self.resolver.resolve(sample.assets[agent.media.asset_id]).path
            target_wave, size = self.features.audio(
                self.resolver, sample.assets[agent.media.asset_id], agent
            )
            decoded += size
            if len(target_wave) != len(wave):
                raise OmniFactoryError(
                    "E_DUPLEX_TRACK", "aligned full-session tracks must have equal sample counts"
                )
            if self._alignments is None:
                self._alignments = load_alignments(self.config.duplex_text_alignment)
            row = self._alignments.get((sample.id, prediction_id))
            if row is None:
                raise OmniFactoryError("E_TEXT_ALIGNMENT", "missing sample/prediction timing")
            text_codes = aligned_text(
                row,
                text,
                self.processor,
                self.root / TEXT_TOKENIZER_NAME,
                target_path,
                frames=frames,
            )
        if decoded > self.config.budget.max_decoded_media_bytes:
            raise OmniFactoryError("E_DECODE_BUDGET", "combined duplex tracks exceed byte budget")
        user_codes = self.features.encode(wave)
        payload = dict(voice_state=self.voice, role_ids=role_ids, user_codes=user_codes)
        plan = SupervisionPlan()
        extra = {}
        if not inference:
            codes = torch.cat([text_codes, self.features.encode(target_wave), user_codes], 1)
            codes = torch.cat([prefix, codes], -1)
            supervised = torch.ones_like(codes, dtype=torch.bool)
            supervised[:, :, : prefix.shape[-1]] = False
            payload.update(codes=codes, supervised=supervised)
            plan, extra = aligned_duplex_supervision(
                codes,
                supervised,
                text_card=self.processor.get_piece_size(),
                audio_card=2048,
                text_weight=self.config.text_loss_weight,
                audio_weight=self.config.speech_loss_weight,
                user_weight=self.config.personaplex_user_loss_weight,
            )
        tensors, layout = pack_tree(payload)
        tensors.update(extra)
        return EncodedExample(
            tensors,
            SampleCost(
                sample_id=sample.id,
                prediction_id=prediction_id,
                modalities=("text", "audio"),
                contract_id=contract_id,
                llm_tokens=total,
                codec_frames=frames * (1 if inference else 2),
                decoded_media_bytes=decoded,
                stream_events=0 if inference else 3 * codes.shape[-1],
            ),
            layout=layout,
            answer=text,
            supervision=plan,
            metadata={
                "persona_frames": prefix.shape[-1],
                "voice_frames": len(self.voice["embeddings"]),
                "conversation_frames": frames,
                "frame_us": 80000,
                "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            },
        ).validate(inference=inference)

    def collate(self, rows):
        return rows[0] if len(rows) == 1 else {"examples": rows}

    @staticmethod
    def to_device(batch, model, config):
        return move_tree(batch, device=config.device, dtype=getattr(torch, config.dtype))

    def forward(self, model, batch):
        if "examples" in batch:
            losses = LossBundle.combine(
                [self.forward(model, row).losses for row in batch["examples"]]
            )
        else:
            losses = duplex_loss(
                model,
                batch["codes"],
                batch["supervised"],
                voice_state=batch["voice_state"],
                agent_only_depth=self.config.personaplex_agent_only_depth,
                text_weight=self.config.text_loss_weight,
                audio_weight=self.config.speech_loss_weight,
                user_weight=self.config.personaplex_user_loss_weight,
            )
        return attach_losses(SimpleNamespace(), losses)

    @torch.no_grad()
    def generate_speech(self, model, batch, *, max_new_tokens, max_codec_tokens):
        if batch["user_codes"].shape[-1] > max_codec_tokens:
            raise OmniFactoryError("E_CODEC_LENGTH", "session exceeds requested frame limit")
        generator = LMGen(
            model,
            device=model.device,
            sample_rate=24000,
            frame_rate=12.5,
            text_prompt_tokens=batch["role_ids"],
            audio_silence_frame_cnt=6,
            use_sampling=False,
            check=True,
        )
        generator.voice_prompt_embeddings = batch["voice_state"]["embeddings"].to(
            model.text_emb.weight
        )
        generator.voice_prompt_cache = batch["voice_state"]["cache"]
        audio, text = [], []
        with no_compile(), generator.streaming(1):
            generator.step_system_prompts(None)
            for frame in batch["user_codes"].split(1, -1):
                out = generator.step(frame)
                if out is not None:
                    audio.append(out[0, 1:9, 0].cpu().tolist())
                    token = int(out[0, 0, 0])
                    if token not in (0, 3):
                        text.append(token)
        if len(audio) != batch["user_codes"].shape[-1]:
            raise OmniFactoryError(
                "E_DUPLEX_OUTPUT", "session did not produce one output frame per input frame"
            )
        return dict(
            text=self.processor.decode(text),
            codec_tokens=audio,
            speech_finished=True,
            speech_eos=None,
            stop_reason="input_session_end",
            input_frames=len(audio),
        )

    def generate(self, model, batch, *, max_new_tokens):
        return [
            self.generate_speech(
                model,
                batch,
                max_new_tokens=max_new_tokens,
                max_codec_tokens=self.config.max_codec_tokens,
            )["text"]
        ]

    def speech_decoder(self, model, *, speech_assets=None, reference_audio=None):
        if speech_assets is not None or reference_audio is not None:
            raise OmniFactoryError(
                "E_VOICE_STATE", "use the bundled personaplex_voice in the recipe"
            )
        return self.features.decode


def export_personaplex(checkpoint, output, *, format=None):
    from omnifactory.training.engine import load_checkpoint
    from omnifactory.training.policy import merge_lora

    if format not in (None, "merged", "vllm-omni", "sglang-omni"):
        raise OmniFactoryError("E_EXPORT_FORMAT", "PersonaPlex requires native complete export")
    model, adapter, config, metadata = load_checkpoint(checkpoint, for_export=True)
    model = merge_lora(model).to(torch.bfloat16)
    with new_directory(output) as root:
        for name in (
            "config.json",
            MIMI_NAME,
            TEXT_TOKENIZER_NAME,
            "voices.tgz",
            "voices",
            "README.md",
            "LICENSE",
            "LICENSE-MIT",
        ):
            source = adapter.root / name
            if source.is_dir():
                shutil.copytree(source, root / name)
            elif source.is_file():
                shutil.copy2(source, root / name)
        if format == "sglang-omni":
            # SGLang otherwise unpacks voices on first use, mutating the sealed export.
            from omnifactory.adapters.personaplex_sglang_assets import unpack_voices

            unpack_voices(root)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            root / "model.safetensors",
        )
        result = dict(
            format=format or "merged",
            adapter="personaplex",
            inference_config={**config.model_dump(mode="json"), "duplex_text_alignment": None},
            source_checkpoint=str(Path(checkpoint).resolve()),
            source_fingerprint=metadata["source_fingerprint"],
            output=str(Path(output).resolve()),
            lora_merged=True,
            vllm_deployment_verified=False,
            native_architecture="PersonaPlexTalkerForConditionalGeneration",
        )
        write_json(root / "omnifactory_export.json", result)
    return result


def validate_export(root, metadata):
    inspect_native_weights(root)
    for name in (MIMI_NAME, TEXT_TOKENIZER_NAME):
        if not (root / name).is_file():
            raise OmniFactoryError("E_EXPORT_ASSET", "missing frozen PersonaPlex asset: " + name)
    voice = metadata["inference_config"]["personaplex_voice"]
    if metadata.get("format") == "sglang-omni" and not (root / "voices" / voice).is_file():
        raise OmniFactoryError("E_EXPORT_ASSET", "SGLang-Omni requires an unpacked selected voice")
    read_voice(root, voice)
