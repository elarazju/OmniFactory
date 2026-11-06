"""Native PersonaPlex training primitives; shared adapter integration is separate.

The native forward aligns its own delayed logits. Never apply another HF-style
label shift, or multiply the NaN-filled invalid tail by a zero mask.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from omnifactory._vendor.personaplex.models.lm import (
    LMGen,
    LMModel,
    LMOutput,
    _delay_sequence,
    _undelay_sequence,
)
from omnifactory._vendor.personaplex.models.loaders import _lm_kwargs
from omnifactory._vendor.personaplex.modules.transformer import StreamingTransformerLayer
from omnifactory._vendor.personaplex.utils.compile import no_compile
from omnifactory.errors import OmniFactoryError
from omnifactory.training.losses import LossBundle, LossTerm

FRAME_US = 80_000
DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)


class PersonaPlexCheckpointLayer(StreamingTransformerLayer):
    gradient_checkpointing = False

    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            if self._streaming_state is not None:
                raise OmniFactoryError(
                    "E_DUPLEX_CHECKPOINT", "checkpointed training cannot mutate streaming caches"
                )

            def native(value):
                with no_compile():
                    return super(PersonaPlexCheckpointLayer, self).forward(value)

            return checkpoint(native, x, use_reentrant=False)
        with no_compile():
            return super().forward(x)


class PersonaPlexTrainingModel(LMModel):
    """Keep native tensor names and causal attention in temporal and depth blocks."""

    def __init__(self, **kwargs):
        super().__init__(layer_class=PersonaPlexCheckpointLayer, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if (gradient_checkpointing_kwargs or {}).get("use_reentrant", False):
            raise OmniFactoryError(
                "E_CHECKPOINT", "PersonaPlex requires non-reentrant checkpointing"
            )
        for module in self.modules():
            if isinstance(module, PersonaPlexCheckpointLayer):
                module.gradient_checkpointing = True

    def agent_depth(self, sequence, hidden):
        """Execute the causal depth prefix; input user audio stays in the temporal model.

        Native per-codebook matrices are indexed by position and support a prefix.
        Do not change dep_q or mutate layer state: checkpoint recomputation and
        native streaming inference must retain the complete 16-codebook model.
        """
        batch, _, frames = sequence.shape
        inputs = []
        for cb in range(8):
            linear = cb if self.depformer_multi_linear else 0
            if self.depformer_multi_linear and self.depformer_weights_per_step_schedule is not None:
                linear = self.depformer_weights_per_step_schedule[cb]
            token = (
                self.depformer_text_emb(sequence[:, 0])
                if cb == 0
                else self.depformer_emb[cb - 1](sequence[:, cb + self.audio_offset - 1])
            )
            inputs.append(token + self.depformer_in[linear](hidden))
        output = self.depformer(torch.stack(inputs, 2).reshape(batch * frames, 8, -1))
        return torch.stack(
            [self.linears[cb](output[:, cb]).reshape(batch, frames, -1) for cb in range(8)], 1
        )

    def forward_train(self, codes, voice_state=None, *, agent_only_depth=False):
        if voice_state is None and not agent_only_depth:
            return super().forward_train(codes)
        if voice_state is None:
            initial = self._get_initial_token().expand(codes.shape[0], -1, -1)
            targets = _delay_sequence(self.delays, codes, initial)
            inputs = torch.cat([initial, targets[:, :, :-1]], -1)
            hidden, text = self.forward_codes(inputs)
        else:
            inputs, targets = voice_conditioned_schedule(codes, voice_state, self)
            voice = voice_state["embeddings"].reshape(1, -1, self.dim)
            embeddings = torch.cat([voice.to(self.text_emb.weight), self.embed_codes(inputs)], 1)
            hidden, text = self.forward_embeddings(embeddings)
            hidden, text = hidden[:, -codes.shape[-1] :], text[:, :, -codes.shape[-1] :]
        audio = (
            self.agent_depth(targets, hidden)
            if agent_only_depth
            else self.forward_depformer_training(targets, hidden)
        )
        depth = audio.shape[1]
        audio, mask = _undelay_sequence(self.delays[1 : depth + 1], audio, fill_value=float("nan"))
        text, text_mask = _undelay_sequence(self.delays[:1], text, fill_value=float("nan"))
        if voice_state is None:
            mask &= codes[:, 1 : depth + 1] != self.zero_token_id
            text_mask &= codes[:, :1] != self.zero_token_id
        return LMOutput(audio, mask, text, text_mask)


def voice_conditioned_schedule(codes, voice_state, model):
    """Use native ring-buffer scheduling with fully teacher-forced suffix codes.

    Replaying V saved voice embedding rows leaves the native generator at V+1.
    All suffix streams are provided, so no model sampling is required to prepare
    its temporal inputs and depth targets. Prefix masking remains caller-owned.
    """
    voice, cache = voice_state["embeddings"], voice_state["cache"]
    if (
        codes.shape[0] != 1
        or voice.ndim != 4
        or voice.shape[1:] != (1, 1, model.dim)
        or not len(voice)
        or cache.shape != (1, 17, 4)
        or cache.dtype != torch.long
        or not torch.isfinite(voice).all()
    ):
        raise OmniFactoryError("E_VOICE_STATE", "invalid native PersonaPlex voice bundle")
    state = SimpleNamespace(
        cache=cache.to(codes.device).clone(),
        provided=torch.zeros_like(cache, device=codes.device, dtype=torch.bool),
        initial=model._get_initial_token(),
        offset=len(voice) + 1,
    )
    owner = SimpleNamespace(_streaming_state=state, lm_model=model, check=True)
    inputs, targets = [], []
    for t in range(codes.shape[-1]):
        result = LMGen.prepare_step_input(
            owner,
            input_tokens=codes[:, 9:, t : t + 1],
            moshi_tokens=codes[:, 1:9, t : t + 1],
            text_token=codes[:, 0, t],
        )
        inp, _, target, previous, _ = result
        inputs.append(inp.clone())
        targets.append(target.clone())
        state.provided[:, :, previous] = False
        state.offset += 1
    return torch.cat(inputs, -1), torch.cat(targets, -1)


def inspect_native_weights(root):
    """Validate official tensor names/shapes on meta, without loading 17 GB weights."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "personaplex" or config.get("version") != "7b-v1":
        raise OmniFactoryError("E_NATIVE_CONFIG", "expected PersonaPlex 7b-v1")
    with torch.device("meta"):
        model = PersonaPlexTrainingModel(**{**_lm_kwargs, "dep_q": 16}, device="meta")
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as weights:
        actual = {name: tuple(weights.get_slice(name).get_shape()) for name in weights.keys()}
    if actual != expected:
        raise OmniFactoryError(
            "E_NATIVE_WEIGHTS", "native tensor inventory differs; missing weights are never copied"
        )
    return model, len(actual)


def load_personaplex(root, *, device="cpu", dtype=torch.float32):
    model, _ = inspect_native_weights(root)
    state = load_file(Path(root) / "model.safetensors", device=str(device))
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype)


def duplex_loss(
    model,
    codes,
    supervised,
    *,
    text_weight=1.0,
    audio_weight=1.0,
    user_weight=0.0,
    voice_state=None,
    agent_only_depth=False,
):
    """Supervise aligned [batch,17,frame] codes, retaining input-only user streams.

    `supervised` explicitly excludes voice/role prefixes and any unlabelled frames.
    User-code prediction is optional auxiliary supervision, not generated agent audio.
    The model owns its 0/1-frame delays, causal temporal mask and causal depth mask.
    """
    if (
        codes.ndim != 3
        or codes.shape[1] != 17
        or codes.shape[0] < 1
        or codes.shape[2] < 1
        or codes.dtype != torch.long
        or supervised.shape != codes.shape
        or supervised.dtype != torch.bool
        or supervised.device != codes.device
        or model.n_q != 16
        or model.dep_q != 16
        or tuple(model.delays) != DELAYS
    ):
        raise OmniFactoryError("E_DUPLEX_LAYOUT", "expected native 17 aligned streams and masks")
    # -1 is the native zero embedding, not a vocabulary symbol to supervise.
    if (
        (codes < -1).any()
        or (codes[:, :1] > model.text_card).any()
        or (codes[:, 1:] > model.card).any()
        or ((codes < 0) & supervised).any()
        or ((codes[:, :1] >= model.text_card) & supervised[:, :1]).any()
        or ((codes[:, 1:] >= model.card) & supervised[:, 1:]).any()
    ):
        raise OmniFactoryError("E_DUPLEX_CODES", "codes or supervised special tokens out of range")
    if agent_only_depth and user_weight:
        raise OmniFactoryError("E_DUPLEX_DEPTH", "agent-only depth cannot supervise user outputs")
    with no_compile():
        output = model.forward_train(
            codes, voice_state=voice_state, agent_only_depth=agent_only_depth
        )
    terms, counts = [], {}
    groups = (
        ("text", output.text_logits, output.text_mask, slice(0, 1), text_weight),
        ("agent", output.logits[:, :8], output.mask[:, :8], slice(1, 9), audio_weight),
        ("user", output.logits[:, 8:], output.mask[:, 8:], slice(9, 17), user_weight),
    )
    for name, logits, native_mask, selection, weight in groups:
        if name == "user" and agent_only_depth:
            counts["user_symbols"] = 0
            terms.append(
                LossTerm(
                    "user_ce", output.text_logits[..., :0].sum(), "user_symbols", 0, "cross_entropy"
                )
            )
            continue
        valid = native_mask & supervised[:, selection]
        # Select before CE: native undelay intentionally puts NaNs in invalid tails.
        selected = logits[valid].float()
        numerator = F.cross_entropy(selected, codes[:, selection][valid], reduction="sum")
        normalizer = name + "_symbols"
        counts[normalizer] = int(valid.sum()) if weight else 0
        terms.append(LossTerm(name + "_ce", numerator, normalizer, weight, "cross_entropy"))
    return LossBundle(tuple(terms), counts, dict.fromkeys(counts, "symbols"))
