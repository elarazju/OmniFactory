"""Native CosyVoice3 builders without executing HyperPyYAML constructors."""

import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint
from transformers import Qwen2Config, Qwen2ForCausalLM

from omnifactory._vendor.cosyvoice3.flow.DiT.dit import DiT
from omnifactory._vendor.cosyvoice3.flow.DiT.modules import DiTBlock
from omnifactory._vendor.cosyvoice3.flow.flow import CausalMaskedDiffWithDiT
from omnifactory._vendor.cosyvoice3.flow.flow_matching import CausalConditionalCFM, ConditionalCFM
from omnifactory._vendor.cosyvoice3.hifigan.f0_predictor import CausalConvRNNF0Predictor
from omnifactory._vendor.cosyvoice3.hifigan.generator import CausalHiFTGenerator
from omnifactory._vendor.cosyvoice3.llm.llm import CosyVoice3LM, Qwen2Encoder
from omnifactory._vendor.cosyvoice3.transformer.upsample_encoder import PreLookaheadLayer
from omnifactory._vendor.cosyvoice3.utils.common import ras_sampling
from omnifactory.errors import OmniFactoryError


def model_root(config):
    root = Path(config.model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        from omnifactory.training.prepare import source_fingerprint

        source_fingerprint(config.model, config.revision)
        root = Path(snapshot_download(config.model, revision=config.revision))
    pointer = root / "cosyvoice3_processor.json"
    return Path(json.loads(pointer.read_text())["source"]) if pointer.exists() else root


def native_config(root):
    path = Path(root) / "cosyvoice3.yaml"
    # BaseLoader only returns strings/containers: !apply and !new never execute.
    raw = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    nodes = dict((k.value, v) for k, v in yaml.compose(path.read_text()).value)
    expected = {
        "llm": "!new:cosyvoice.llm.llm.CosyVoice3LM",
        "flow": "!new:cosyvoice.flow.flow.CausalMaskedDiffWithDiT",
        "hift": "!new:cosyvoice.hifigan.generator.CausalHiFTGenerator",
    }
    if any(nodes[k].tag != tag for k, tag in expected.items()):
        raise OmniFactoryError("E_NATIVE_CONFIG", "expected CosyVoice3 native LLM/flow/HiFT")

    def resolve(value, seen=()):
        if isinstance(value, dict):
            return {k: resolve(v, seen) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v, seen) for v in value]
        if "<" in value:
            parts = value.split("*")
            result = []
            for part in parts:
                name = re.fullmatch(r"\s*<([A-Za-z_][A-Za-z_0-9]*)>\s*", part)
                if not name or name[1] in seen or name[1] not in raw:
                    raise OmniFactoryError("E_NATIVE_CONFIG", "unsupported/cyclic native reference")
                result.append(resolve(raw[name[1]], (*seen, name[1])))
            if len(result) == 1:
                return result[0]
            if any(not isinstance(v, (int, float)) for v in result):
                raise OmniFactoryError("E_NATIVE_CONFIG", "native product requires numbers")
            return int(np.prod(result))
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if value.lower() in ("null", "~"):
            return None
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", value):
            return float(value)
        return value

    cfg = {k: resolve(raw[k]) for k in ("llm", "flow", "hift", "sample_rate", "feat_extractor")}
    flow = cfg["flow"]
    if (
        cfg["sample_rate"] != 24000
        or flow["input_frame_rate"] != 25
        or flow["token_mel_ratio"] != 2
        or flow["output_size"] != 80
        or flow["vocab_size"] != cfg["llm"]["speech_token_size"]
    ):
        raise OmniFactoryError(
            "E_NATIVE_CONFIG", "expected 25 Hz speech tokens and 50 Hz/80-bin mel"
        )
    return cfg


class CosyVoice3Qwen2Encoder(Qwen2Encoder):
    """Adapt native cached decoding masks to Transformers 4.57's cache contract."""

    def forward_one_step(self, xs, masks, cache=None):
        # Native inference sends a mask for only the current chunk after prefill.
        # Transformers 4.57 needs all cached positions to remain visible. This
        # adapter generates one unpadded sample, so the full mask contains ones.
        if xs.shape[0] != 1 or not masks[:, -1, :].all():
            raise OmniFactoryError("E_NATIVE_MASK", "expected one unpadded generation sequence")
        past = 0 if cache is None else cache.get_seq_length()
        mask = torch.ones(1, past + xs.shape[1], device=xs.device, dtype=torch.bool)
        outputs = self.model(
            inputs_embeds=xs,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        return outputs.hidden_states[-1], outputs.past_key_values


def build_talker(root, cfg):
    encoder = CosyVoice3Qwen2Encoder.__new__(CosyVoice3Qwen2Encoder)
    torch.nn.Module.__init__(encoder)
    qwen = Qwen2Config.from_pretrained(Path(root) / "CosyVoice-BlankEN", local_files_only=True)
    if qwen.hidden_size != cfg["llm_input_size"] or cfg["llm_input_size"] != cfg["llm_output_size"]:
        raise OmniFactoryError("E_NATIVE_CONFIG", "native Qwen/LLM dimensions differ")
    qwen.use_cache = False
    encoder.model = Qwen2ForCausalLM(qwen)
    return CosyVoice3LM(
        **{k: v for k, v in cfg.items() if k not in ("llm", "sampling")},
        llm=encoder,
        sampling=ras_sampling,
    )


def build_flow(cfg):
    # Preserve the native fixed inference noise without the upstream constructor's
    # global random/NumPy/CPU/CUDA seed reset. Computational methods stay native.
    decoder = cfg["decoder"]
    cfm = CausalConditionalCFM.__new__(CausalConditionalCFM)
    ConditionalCFM.__init__(
        cfm,
        **{k: v for k, v in decoder.items() if k not in ("cfm_params", "estimator")},
        cfm_params=DictConfig(decoder["cfm_params"]["content"]),
        estimator=DiT(**decoder["estimator"]),
    )
    cfm.rand_noise = torch.randn((1, 80, 50 * 300), generator=torch.Generator().manual_seed(0))
    return CausalMaskedDiffWithDiT(
        **{k: v for k, v in cfg.items() if k not in ("decoder", "pre_lookahead_layer")},
        decoder=cfm,
        pre_lookahead_layer=PreLookaheadLayer(**cfg["pre_lookahead_layer"]),
    )


def build_hift(cfg):
    return CausalHiFTGenerator(
        **{k: v for k, v in cfg.items() if k != "f0_predictor"},
        f0_predictor=CausalConvRNNF0Predictor(**cfg["f0_predictor"]),
    )


def load_weights(module, path):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    # Original HiFT files use an optional generator prefix; other stages don't.
    if all(k.startswith("generator.") for k in state):
        state = {k.removeprefix("generator."): v for k, v in state.items()}
    module.load_state_dict(state, strict=True)
    return module


class CosyVoice3TrainingModel(torch.nn.Module):
    def __init__(self, root, cfg, task):
        super().__init__()
        self.native = cfg
        if task == "speech_sft":
            self.talker = load_weights(build_talker(root, cfg["llm"]), root / "llm.pt")
        else:
            self.flow = load_weights(build_flow(cfg["flow"]), root / "flow.pt")

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self, "talker"):
            self.talker.llm.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            for layer in self.flow.modules():
                if not isinstance(layer, DiTBlock) or hasattr(layer, "_of_forward"):
                    continue
                layer._of_forward = layer.forward

                def forward(*args, _layer=layer, **kw):
                    if _layer.training and torch.is_grad_enabled():
                        return checkpoint(_layer._of_forward, *args, use_reentrant=False, **kw)
                    return _layer._of_forward(*args, **kw)

                layer.forward = forward


def load_cosyvoice3(config):
    root = model_root(config)
    model = CosyVoice3TrainingModel(root, native_config(root), config.task)
    return model.to(device=config.device, dtype=getattr(torch, config.dtype))
