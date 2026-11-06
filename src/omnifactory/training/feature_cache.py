"""Content-addressed disk cache at explicitly frozen Qwen encoder boundaries.

Vision caches BEFORE the trainable merger. Audio caches the complete tower only
when every tower parameter is frozen. No cache of trainable projector outputs.
"""

import hashlib
import inspect
import os
import tempfile
from pathlib import Path
from types import MethodType

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.modeling_outputs import BaseModelOutput

from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import digest
from omnifactory.training.efficiency import environment


def tensor_digest(values):
    h = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().contiguous().cpu()
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class FrozenCache:
    def __init__(self, root, module, *, boundary, excluded_prefix=None):
        self.module = module
        self.excluded_prefix = excluded_prefix
        self.parameters = {
            n: p
            for n, p in module.named_parameters()
            if excluded_prefix is None or not n.startswith(excluded_prefix)
        }
        self.buffers = {
            n: b
            for n, b in module.named_buffers()
            if excluded_prefix is None or not n.startswith(excluded_prefix)
        }
        if any(p.requires_grad for p in self.parameters.values()):
            raise OmniFactoryError(
                "E_FEATURE_CACHE_BOUNDARY", f"{boundary} contains trainable weights"
            )
        for name, child in module.named_modules():
            if excluded_prefix and name.startswith(excluded_prefix.rstrip(".")):
                continue
            if isinstance(child, torch.nn.Dropout) and child.p:
                raise OmniFactoryError("E_FEATURE_CACHE_STOCHASTIC", name)
            if getattr(child, "attention_dropout", 0):
                raise OmniFactoryError("E_FEATURE_CACHE_STOCHASTIC", name)
        tensors = {**self.parameters, **self.buffers}
        self.stamp = self.versions()
        self.contract = digest(
            {
                "boundary": boundary,
                "format": 1,
                "weights": tensor_digest(tensors),
                "config": module.config.to_dict(),
                "environment": environment(next(module.parameters()).device.type),
                "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native_implementation": hashlib.sha256(
                    Path(inspect.getfile(type(module))).read_bytes()
                ).hexdigest(),
            }
        )
        self.root = Path(root) / self.contract
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def versions(self):
        return [
            (n, id(t), t.data_ptr(), t._version, str(t.dtype), str(t.device), t.requires_grad)
            for n, t in {
                **dict(self.module.named_parameters()),
                **dict(self.module.named_buffers()),
            }.items()
            if self.excluded_prefix is None or not n.startswith(self.excluded_prefix)
        ]

    def get(self, inputs, compute):
        if self.versions() != self.stamp:
            raise OmniFactoryError(
                "E_FEATURE_CACHE_CHANGED",
                "encoder weights/dtype/policy changed; rebuild cache binding",
            )
        if any(t.requires_grad for t in inputs.values()):
            raise OmniFactoryError(
                "E_FEATURE_CACHE_INPUT", "cannot detach trainable encoder inputs"
            )
        device = next(iter(inputs.values())).device
        key = digest(
            {
                "inputs": tensor_digest(inputs),
                "autocast": torch.is_autocast_enabled(device.type),
                "autocast_dtype": str(torch.get_autocast_dtype(device.type)),
            }
        )
        path = self.root / f"{key}.safetensors"
        if path.exists():
            values = load_file(path)
            with safe_open(path, framework="pt") as reader:
                metadata = reader.metadata()
            if metadata != {
                "contract": self.contract,
                "inputs": key,
                "content": tensor_digest(values),
            }:
                raise OmniFactoryError("E_FEATURE_CACHE_HASH", str(path))
            self.hits += 1
        else:
            with torch.no_grad():
                values = {n: t.detach().contiguous().cpu() for n, t in compute().items()}
            handle, temporary = tempfile.mkstemp(dir=self.root, suffix=".tmp")
            os.close(handle)
            try:
                save_file(
                    values,
                    temporary,
                    metadata={
                        "contract": self.contract,
                        "inputs": key,
                        "content": tensor_digest(values),
                    },
                )
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.misses += 1
        return {n: t.to(device) for n, t in values.items()}


class _CapturedBoundary(Exception):
    pass


def install_feature_cache(model, config):
    if config.feature_cache is None:
        return
    if hasattr(model, "_omnifactory_feature_caches"):
        raise OmniFactoryError("E_FEATURE_CACHE", "cache already installed")
    caches = {}
    if "vision_encoder" in config.feature_cache_components:
        visual = model.visual
        cache = FrozenCache(
            config.feature_cache,
            visual,
            boundary="qwen-vision-pre-merger-v1",
            excluded_prefix="merger.",
        )
        original = visual.forward

        def visual_forward(module, hidden_states, grid_thw, **kwargs):
            if kwargs:
                raise OmniFactoryError(
                    "E_FEATURE_CACHE_INPUT", "unsupported vision capture arguments"
                )

            def compute():
                captured = {}

                def capture(_module, args):
                    captured["hidden"] = args[0]
                    raise _CapturedBoundary()

                handle = module.merger.register_forward_pre_hook(capture)
                try:
                    original(hidden_states, grid_thw)
                except _CapturedBoundary:
                    pass
                finally:
                    handle.remove()
                if "hidden" not in captured:
                    raise OmniFactoryError(
                        "E_FEATURE_CACHE_BOUNDARY", "native merger boundary not reached"
                    )
                window_index, _ = module.get_window_index(grid_thw)
                captured["reverse"] = torch.argsort(window_index)
                return captured

            values = cache.get({"pixels": hidden_states, "grid": grid_thw}, compute)
            # This remains in the training graph, including projector-only tuning.
            return module.merger(values["hidden"])[values["reverse"]]

        visual.forward = MethodType(visual_forward, visual)
        caches["vision_encoder"] = cache
    if "audio_encoder" in config.feature_cache_components:
        audio = model.audio_tower
        audio_cache = FrozenCache(config.feature_cache, audio, boundary="qwen-audio-output-v1")
        audio_original = audio.forward

        def audio_forward(module, input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
            if kwargs or feature_lens is None or aftercnn_lens is None:
                raise OmniFactoryError(
                    "E_FEATURE_CACHE_INPUT", "requires explicit native audio lengths"
                )
            values = audio_cache.get(
                {"features": input_features, "lengths": feature_lens, "cnn_lengths": aftercnn_lens},
                lambda: {
                    "hidden": audio_original(
                        input_features, feature_lens, aftercnn_lens
                    ).last_hidden_state
                },
            )
            return BaseModelOutput(last_hidden_state=values["hidden"])

        audio.forward = MethodType(audio_forward, audio)
        caches["audio_encoder"] = audio_cache
    model._omnifactory_feature_caches = caches


def cache_stats(model):
    return {
        name: {"hits": cache.hits, "misses": cache.misses, "contract": cache.contract}
        for name, cache in getattr(model, "_omnifactory_feature_caches", {}).items()
    }
