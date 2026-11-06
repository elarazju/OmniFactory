"""Compare cold/warm native AudioVAE encoding without weakening resume checks."""

import argparse
import contextlib
import json

import torch

from omnifactory.adapters.voxcpm2 import VoxCPM2Adapter
from omnifactory.data.io import iter_samples
from omnifactory.training.config import TrainConfig

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--jit-off", action="store_true")
p.add_argument("--deterministic-cudnn", action="store_true")
a = p.parse_args()
torch.set_num_threads(4)
cfg = TrainConfig.model_validate(json.load(open(a.config)))
adapter = VoxCPM2Adapter(cfg)
sample = next(iter_samples(cfg.dataset))
context = torch.jit.optimized_execution(False) if a.jit_off else contextlib.nullcontext()
cudnn = (
    torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False)
    if a.deterministic_cudnn
    else contextlib.nullcontext()
)
with context, cudnn:
    values = [
        adapter.encode(sample, "p0", contract_id="probe").tensors["audio_feats"] for _ in range(4)
    ]
print(
    json.dumps(
        {
            "jit_off": a.jit_off,
            "deterministic_cudnn": a.deterministic_cudnn,
            "max_abs_from_first": [float((v - values[0]).abs().max()) for v in values],
        }
    ),
    flush=True,
)
