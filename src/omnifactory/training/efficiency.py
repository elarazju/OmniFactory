"""Measured execution telemetry; counts native LLM slots, never codec channels as text."""

import os
import platform
import subprocess
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

import torch


def environment(device):
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "flash-attn", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = {
        "python": platform.python_version(),
        "packages": packages,
        "device": device,
        "cuda_runtime": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "sdpa_backends": {
            name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
            for name in ("flash", "mem_efficient", "math", "cudnn")
        },
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        result.update(
            driver_versions=sorted(set(driver)),
            gpu=props.name,
            gpu_total_bytes=props.total_memory,
            compute_capability=[props.major, props.minor],
        )
    return result


def batch_slots(batch):
    """Count the actual collated 2D visibility mask, including per-example execution.

    MiMo masks address LLM patches rather than raw codec symbols. Flow-only
    batches have no LLM mask and deliberately contribute zero LLM slots.
    """
    if "examples" in batch:
        counts = [batch_slots(row) for row in batch["examples"]]
        return tuple(sum(c[i] for c in counts) for i in range(2))
    # Packed execution can report original segment lengths without materializing
    # a dense mask. Adapters must retain the actual lengths in this field.
    if "packing_lengths" in batch:
        length = sum(batch["packing_lengths"])
        return length, length
    if "sequence_length" in batch and "sample_lens" in batch:
        length = batch["sequence_length"]
        if sum(batch["sample_lens"]) != length:
            raise ValueError("native packed sequence length differs from sample lengths")
        return length, length
    if "codes" in batch and "voice_state" in batch:
        # PersonaPlex sums all 17 streams at each temporal position, then
        # prepends cached voice embeddings. Codebook count is not LLM length.
        codes, voice = batch["codes"], batch["voice_state"]["embeddings"]
        if codes.ndim != 3 or codes.shape[:2] != (1, 17) or voice.ndim != 4:
            raise ValueError("expected one PersonaPlex session and native voice embeddings")
        length = codes.shape[-1] + len(voice)
        return length, length
    mask = batch.get("attention_mask")
    if mask is None and "text_mask" in batch and "audio_mask" in batch:
        # VoxCPM advances its causal LM once per text token or audio patch,
        # independently of the number of latent frames/features in that patch.
        text, audio = batch["text_mask"], batch["audio_mask"]
        if text.shape != audio.shape or text.ndim != 2:
            raise ValueError("text/audio slot masks must have the same 2D shape")
        mask = text.bool() | audio.bool()
    if mask is None:
        return 0, 0
    if mask.ndim != 2:
        raise ValueError("efficiency telemetry requires a 2D slot mask or packing_lengths")
    return int(torch.count_nonzero(mask)), mask.numel()


class StepMeasurement:
    """Time data loading, collation, transfer, forward/backward and optimizer step.

    Checkpoint writing and initial dataset preparation/model loading are excluded.
    CUDA boundaries synchronize; peaks are per measured optimizer step, not a
    cumulative high-water mark. Reserved memory includes allocator history.
    """

    def __init__(self, device):
        self.device = device
        self.useful_slots = self.allocated_slots = self.examples = self.microbatches = 0
        self.host_seconds = {"data_load": 0.0, "collate": 0.0}

    @contextmanager
    def host_stage(self, name):
        """Host wall time only; pending GPU work is deliberately not synchronized."""
        started = perf_counter()
        try:
            yield
        finally:
            self.host_seconds[name] += perf_counter() - started

    def start(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started = perf_counter()
        return self

    def observe(self, batch, examples):
        useful, allocated = batch_slots(batch)
        self.useful_slots += useful
        self.allocated_slots += allocated
        self.examples += examples
        self.microbatches += 1

    def finish(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - self.started
        return {
            "step_seconds": elapsed,
            **{f"{key}_host_seconds": value for key, value in self.host_seconds.items()},
            "examples": self.examples,
            "microbatches": self.microbatches,
            "examples_per_second": self.examples / elapsed,
            "useful_llm_slots": self.useful_slots,
            "allocated_llm_slots": self.allocated_slots,
            "llm_slots_per_second": self.useful_slots / elapsed,
            "llm_padding_fraction": 1 - self.useful_slots / self.allocated_slots
            if self.allocated_slots
            else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if self.device == "cuda"
            else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()
            if self.device == "cuda"
            else None,
        }


def summarize_steps(rows):
    """Ratio of totals, not mean of per-step rates; units remain explicit."""
    seconds = sum(r["step_seconds"] for r in rows)
    useful = sum(r["useful_llm_slots"] for r in rows)
    allocated = sum(r["allocated_llm_slots"] for r in rows)
    return {
        "steps": len(rows),
        "seconds": seconds,
        "examples": sum(r["examples"] for r in rows),
        "examples_per_second": sum(r["examples"] for r in rows) / seconds,
        "llm_slots_per_second": useful / seconds,
        "useful_llm_slots": useful,
        "allocated_llm_slots": allocated,
        "llm_padding_fraction": 1 - useful / allocated if allocated else None,
        **{
            key: max((r[key] for r in rows if r[key] is not None), default=None)
            for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
        },
    }
