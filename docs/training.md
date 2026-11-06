# Training

Start from a [model recipe](models.md). A training YAML selects the model and revision, dataset, task, trainable components, precision, resource budgets, and output directory. Every relative path resolves from that YAML's directory.

## Tune selected components

Components can be frozen, fully trained, or adapted with LoRA. Supported names and target modules depend on the model. For MiniCPM language LoRA and a separately tuned audio projector:

```yaml
components:
  language_model:
    mode: lora
    rank: 8
    alpha: 16
    lr: 0.0001
    target_modules: [q_proj, v_proj]
  audio_projector:
    mode: lora
    rank: 4
    alpha: 8
    lr: 0.00002
    target_modules: [linear1, linear2]
```

Omitted components stay frozen. For projector-only training, configure only that component and supply examples that exercise it. A text-only example cannot train an audio projector. Invalid or unmatched component targets fail before optimization.

For small learning rates with BF16 trainable parameters, set
`optimizer_master_weights: true` to keep AdamW's master weights and moments in
FP32. Each step accumulates the update in FP32 and projects the result back to the
model's existing execution dtype. This retains updates smaller than a single
BF16 rounding interval; it does not change forward precision or guarantee a
quality improvement. The option defaults to `false`.

Start a new run when changing this option. Exact resume requires the original
`optimizer.pt`, including its master weights; a model-only checkpoint cannot
reconstruct those hidden bits. Check an intermediate checkpoint against the
uninterrupted final checkpoint with:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra train python scripts/verify_training_resume.py \
  --checkpoint outputs/run/checkpoint-00000064 \
  --expected outputs/run/checkpoint-00000128 --output outputs/resume-check
```

Use the checkpoint paths from your run. This compares model parameters, optimizer
state and CPU/CUDA RNG state exactly; enable `deterministic: true` for a reproducible
CUDA comparison.

## Prepare once and reuse

```bash
uv run --extra train omnifactory prepare \
  --config examples/train-tiny.yaml --output outputs/prepared
```

Add this field to the same YAML before training:

```yaml
prepared_dataset: ../outputs/prepared
```

The original dataset, media, and base model must remain accessible for validation. You can change learning rates or component policies for a new run without re-encoding compatible data. Model, encoder, label, or input changes invalidate reuse.

- Interrupted preparation: repeat the command with `--resume` and the same output path.
- Incremental preparation after editing the dataset: choose a new output path and pass `--reuse-from outputs/prepared`.
- Corrupted or incompatible artifacts: regenerate from original data; they are not silently accepted.

Current prepared data uses **v4**: multiple records share each safetensors shard, and a SQLite index maps sample/prediction IDs to record-local tensor names. Discrete tokens, continuous latents, and aligned streams retain the same supervision contracts. Older v1–v3 data must be regenerated from original inputs. This is local/shared-filesystem processing; remote object-storage streaming and distributed training are not implemented.

### Shards and bounded prefetch

Defaults:

```yaml
prepared_shard_records: 256
prepared_shard_bytes: 268435456       # 256 MiB of buffered tensor data
prepared_prefetch_records: 2
prepared_prefetch_bytes: 67108864     # 64 MiB of queued/running tensor loads
prepared_cache_bytes: 0
```

A shard flushes when either record count or tensor bytes reaches its limit. An individual record larger than the byte target gets its own shard; it must still pass the configured sample budgets. Encoding needs additional workspace and one incoming record, so the shard target is not a total process-memory limit.

Preparation publishes a complete shard before committing its index rows together. After interruption, `--resume` verifies completed shards and rebuilds the unfinished shard. The original inputs, encoding and shard settings must match the interrupted run. Incremental `--reuse-from` can select individual records from existing shards and repack them with new records without invoking their encoders again. Changing only storage settings does not alter encoding semantics; repacking creates a new artifact fingerprint, so retain the original prepared artifact for exact training recovery.

The reader loads only the requested record's tensors. Prefetch uses one CPU worker within the current gradient-accumulation window, preserves order, and does not advance the saved training cursor. Both queued and running loads reserve bytes. A record larger than the prefetch byte limit loads synchronously. Set `prepared_prefetch_records: 0` to disable it. Python objects, the active batch, temporary copies and any separately enabled cache need additional memory. Metrics under `prepared_cache` report the separate cache and prefetch limits, peaks and bypass counts.

Dataset duplicate/split checks and prepared source-group membership use disk-backed indexes. Checkpoints retain a hashed `groups.sqlite` file for held-out evaluation; keep it with the checkpoint. Local temporary disk is needed for validation indexes.

To check storage and read correctness at scale:

```bash
uv run --extra train python scripts/benchmark_prepared_storage.py \
  --output outputs/storage-check --records 100000
```

This produces synthetic tensor shards and checks every requested record, reporting file count, tensor-buffer peaks, process peak RSS and timings. It does not measure native media encoding, GPU training throughput or model quality.

For repeated epochs on a small prepared working set, optionally set
`prepared_cache_bytes: 134217728` (128 MiB). The default is `0`, which keeps disk
loading. This CPU LRU cache limits retained tensor bytes; current batches, Python
metadata and temporary copies need additional host memory. Oversized records bypass
the cache. Hits return independent tensor copies, and changed file identities or
record metadata force normal loading and validation. Cache size does not change
the data order or resume cursor. Training and benchmark records include hit/miss
counts and retained bytes when enabled.

Compare host loading on your prepared data before enabling a large cache:

```bash
uv run --extra train python scripts/benchmark_prepared_loading.py \
  --prepared outputs/prepared --output outputs/loading-comparison.json \
  --cache-bytes 134217728 --repeats 3
```

This comparison checks tensor-byte parity and alternates measurement order after
warmup. Its loading speed ratio is not an end-to-end training speedup; use optimizer
step measurements to assess the effect on your workload.

For a paired production-training comparison with your configured dataset:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra train python scripts/benchmark_training_optimization.py \
  --config examples/train-qwen2.5-asr-baseline.yaml \
  --output outputs/cache-training-comparison --optimization cache
```

Each arm runs in a fresh process from the same base and prepared data, with the
same optimizer steps and schedule. The script disables validation in both arms,
requires identical final trainable weights, and resumes an uncached checkpoint
with caching enabled to check exact recovery. It records both initial loading
and performance after the first epoch. Set enough steps to revisit the working
set; one-pass training cannot benefit from cache hits. A single paired run is
not a statistical performance guarantee.

For PersonaPlex, `--optimization agent_depth` compares full-depth training with
the optional agent-only path and checks exact resume within the optimized path.
Unlike CPU caching, floating-point kernel shape changes are compared with explicit
numerical tolerances, retained per tensor in `weight-parity.json`. Check logits,
losses and trainable gradients on an existing trained checkpoint first:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra train --extra personaplex \
  python scripts/validate_gpu_personaplex_depth.py \
  --checkpoint outputs/personaplex/checkpoint-00000016 \
  --prepared outputs/personaplex/prepared --output outputs/depth-numerics
```

Use your actual checkpoint and prepared directory. The numerical check includes
every prepared record; its forward/backward timings exclude optimizer updates.

## Memory and batching

Use gradient checkpointing, a small per-batch budget, and gradient accumulation to control the active workload:

```yaml
gradient_checkpointing: true
gradient_accumulation_steps: 4
budget:
  max_examples: 1
  max_llm_tokens: 2048
  max_vision_patches: 4096
  max_audio_frames: 4096
  max_codec_frames: 1024
  max_latent_elements: 1000000
  max_decoded_media_bytes: 64000000
```

These are illustrative counts, not a model-independent fit guarantee. Use your model's recipe and inspect `metrics.jsonl` for measured CUDA allocation and padding. Long videos, high-resolution images, and long audio can require very different budgets. Oversized samples raise errors instead of silently truncating targets.

| Option | Available scope |
| --- | --- |
| NF4 QLoRA | Qwen2.5-Omni understanding on CUDA/BF16; [3B](../examples/train-qwen2.5-omni-3b.yaml) and [7B](../examples/train-qwen2.5-omni-7b-nf4.yaml) |
| FlashAttention varlen packing | Qwen2.5 understanding, unquantized BF16 text/image/audio; [recipe](../examples/train-qwen2.5-omni-3b-packed.yaml). Video uses per-example execution; speech packing is unsupported. |
| Frozen encoder cache | Qwen2.5 understanding with frozen selected encoders and an unquantized base; [recipe](../examples/train-qwen2.5-omni-3b-cached.yaml). Projector gradients remain active. |

Packing changes floating-point execution order; bitwise equality with independent BF16 execution is not guaranteed. Cache usefulness depends on encoder compute versus storage/transfer overhead. Neither option changes the exported model format.

For workload measurements, compare runs with the same model, data, budgets, warmup, and repeats:

```bash
uv run --extra train omnifactory benchmark \
  --config examples/train-qwen2.5-omni-3b-bf16.yaml \
  --output outputs/benchmark --batching modal --warmup-epochs 1 --repeats 3
```

The benchmark excludes initial preparation, model loading, and checkpoint writes. `calibrate-memory --help` describes how to derive a bounded profile from representative CUDA measurements; profiles are specific to the measured execution settings.

## Validation and longer runs

```yaml
validation_dataset: ../data/validation.jsonl
validation_root: ../data
max_steps: 1000
save_steps: 100
eval_steps: 100
save_total_limit: 3
lr_scheduler: cosine
scheduler_steps: 1000
warmup_steps: 50
weight_decay: 0.01
```

Schedules are `constant`, `linear`, or `cosine`, applied per optimizer step while preserving component learning-rate ratios. Nonconstant schedules require an explicit horizon; their learning rate reaches zero at that horizon. Validation requires disjoint source groups and writes native-objective loss to `evaluation.jsonl`. It does not measure perceptual speech or image quality.

Checkpoint retention protects the newest and best checkpoints; `save_total_limit` must be at least two. The run result reports `best_checkpoint` and `best_metric`. Export the checkpoint you select explicitly. The [VoxCPM2 longer-run recipe](../examples/train-voxcpm2-pilot.yaml) includes these controls.

## Resume training

For paired real-recording evaluation, see [the baseline workflow](#paired-real-data-baselines).

Copy your YAML while preserving its relative paths, then change:

```yaml
output_dir: ../outputs/resumed-run
resume_from: ../outputs/previous-run/checkpoint-00000050
max_steps: 100
```

`max_steps` is the total desired optimizer step, not additional steps. Resume restores optimizer and RNG state and validates the model, data, and execution contract. Changing learning rates or other execution settings requires a new run. CPU/CUDA resume is supported; MPS resume is not.

For runs requiring identical repeated results on the same environment, set `deterministic: true`. CUDA runs must also be launched with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. This selects deterministic PyTorch kernels, disables cuDNN autotuning for the run, and fails if an operation has no deterministic implementation. It can reduce throughput; it does not promise identical results across hardware or library versions. The option is part of the resume contract and does not change prepared labels.

Checkpoint metadata refers to the base model and data paths. Keep them available when resuming or exporting; automatic relocation between machines is not implemented. Use a new output directory so completed checkpoints remain intact.

## Timed attention

MiniCPM's offline text/audio path accepts `duplex_schedule`, a JSON array of `TokenSchedule` records. Each record binds a sample/prediction to the SHA-256 of its processor-expanded `int64 input_ids`; `token_fingerprint(encoded.payload["input_ids"])` computes this binding. Changing token expansion requires a new schedule.

Events partition the entire expanded sequence without gaps or overlaps. Each has `id`, a half-open `start_token`/`end_token` range, `start_us`/`end_us`, `available_us`, `direction` (`input`, `condition`, or `output`), and `modality`. An input becomes visible only when the complete event is available. Keep each encoded audio clip in one event; its time range must cover the waveform. Images/video are not supported in this timed path.

For each supervised target, include both that token and its preceding query slot in the same output event. Set `max_stream_events` and `max_attention_pairs` to bound schedule and dense-mask allocation. The [recipe](../examples/train-minicpm-duplex.yaml) needs a schedule matching your own prepared examples; waveform duration alone cannot supply token alignment. The typed [schedule schema](../src/omnifactory/adapters/duplex.py) defines the complete validation contract.

This is offline teacher-forced training with explicit visibility constraints. It does not provide live event scheduling, interruption, streaming generation, or packed attention. Inference with a timed training schedule is rejected rather than silently dropping its timing constraints.

## Paired real-data baselines

The [Qwen2.5 ASR recipe](../examples/train-qwen2.5-asr-baseline.yaml) and
[VoxCPM2 speech recipe](../examples/train-voxcpm2-baseline.yaml) provide an explicit
starting point for real-recording comparisons. Prepare [speaker-disjoint data](datasets.md#use-real-data),
then set the model and dataset paths in the recipe. The baseline runner requires
CUDA and an unquantized base. It creates a new output directory and independently
loads the base model, trained adapter and complete native export.

These recipes keep optimizer accumulation in FP32 while executing the model in
BF16. This preserves small updates; choose optimizer and component learning rates
using held-out task metrics. For a precision ablation, copy the recipe, set
`optimizer_master_weights: false`, and start a fresh run with the same data,
component policies, learning rates and seed.

Run the paired comparison:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra train python scripts/run_real_data_baseline.py \
  --config examples/train-qwen2.5-asr-baseline.yaml \
  --output outputs/qwen2.5-quality-baseline

CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra train python scripts/run_real_data_baseline.py \
  --config examples/train-voxcpm2-baseline.yaml \
  --asr-model /path/to/pinned/Qwen3-Omni-30B-A3B-Instruct \
  --output outputs/voxcpm2-quality-baseline
```

Every validation prediction participates. Generation uses a stable per-case seed
across all three branches; waveform hashes, transcripts, termination status and
individual timings remain in the output directory. The TTS comparison requires a
separate Qwen3-Omni ASR scorer and enough memory to load it. Its WER measures an
ASR-based intelligibility proxy, not human preference or speaker similarity.
The English scorer applies NFKC and lowercase normalization, preserves apostrophes,
and replaces other punctuation with spaces. It does not expand numbers or
abbreviations; inspect the saved transcripts when these affect a comparison.
The scorer covers each complete waveform. Audio longer than its model's 30-second
limit is split into balanced, non-overlapping segments; boundaries and individual
transcripts are retained, then joined for WER. Segment boundaries can affect ASR
errors. Synthesis termination is reported independently. If only scoring fails, preserve
the run and score its saved waveforms in a fresh directory:

```bash
uv run --extra train python scripts/rescore_saved_speech.py \
  --run outputs/voxcpm2-quality-baseline \
  --asr-model /path/to/pinned/Qwen3-Omni-30B-A3B-Instruct \
  --output outputs/voxcpm2-rescore
```

This verifies the original datasets, export and waveform hashes before evaluation.
Its report retains the generation identity and records the new scorer identity.

`report.json` records each branch's WER, export differences, training throughput
and CUDA peaks. `preparation.json` records preparation time, while `*-stage.json`
records complete stage wall times. Per-step `data_load_host_seconds` and
`collate_host_seconds` are host timings without extra CUDA synchronization; they
must not be interpreted as directly recoverable GPU idle time. Throughput excludes
the first two optimizer steps, and the report also retains the all-step allocation
peak. Compare the base under the same task instruction and text preprocessing
before attributing a score change to training.

For TTS, also score original recordings with the same ASR model to estimate scorer
errors. Inspect reference-conditioned and unconditioned cases separately, retain
every output, and listen to paired samples. A small WER change on a development set
alone does not establish a reliable quality gain. The diagnostic
`scripts/run_recipe_screen.py` provides prompt/casing controls and an original-audio
ASR check before a new training run.

### Evaluate a frozen speech recipe

Change one component learning rate at a time while holding the training data,
optimizer, seed, step count and loss weights fixed. Select the recipe on the
development set, then evaluate an [independent speech set](datasets.md#prepare-an-independent-speech-confirmation-set).
Keep its results out of further recipe selection. Repeat the control and selected
recipe with multiple training seeds; keep the generation seed fixed between them.

Evaluate an existing VoxCPM2 checkpoint without retraining:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra train python scripts/evaluate_speech_recipe.py generate \
  --config outputs/run/checkpoint-00000128/checkpoint.json \
  --checkpoint outputs/run/checkpoint-00000128 --mode adapter \
  --dataset outputs/tts-confirmation/test.jsonl --generation-seed 42 \
  --output outputs/confirmation-selected
```

The recipe must use deterministic CUDA execution. Use `--mode base` without
`--checkpoint` to evaluate its base model, or `--mode export --export outputs/export`
to evaluate the complete export. Every arm retains its identity, waveform hashes,
termination and trailing low-energy measurements at fixed -50/-60 dBFS thresholds.
These energy measurements help diagnose ending behavior; they are not VAD labels
or listening scores.

Generate a control arm in a separate directory, then create `outputs/speech-score.json`
with paths relative to the repository root:

```json
{
  "dataset": "outputs/tts-confirmation/test.jsonl",
  "config": "outputs/run/checkpoint-00000128/checkpoint.json",
  "arms": {
    "control": "outputs/confirmation-control/adapter-predictions.jsonl",
    "selected": "outputs/confirmation-selected/adapter-predictions.jsonl"
  },
  "groundtruth": true
}
```

```bash
uv run --extra train python scripts/evaluate_speech_recipe.py score \
  --manifest outputs/speech-score.json \
  --asr-model /path/to/pinned/Qwen3-Omni-30B-A3B-Instruct \
  --output outputs/confirmation-scores
```

Scoring verifies complete paired identities, references and waveform hashes, and
keeps the real-recording scorer floor separate. For repeated training seeds, average
paired errors within each utterance before resampling speakers for uncertainty;
repeated seeds do not create more independent test recordings. Report each seed,
reference condition and merged-model result alongside the aggregate.

### Diagnose differences after merging

Merging LoRA changes the order of low-precision operations. Verify exported
weights, fixed-input model outputs, and generation separately. A matching WER
does not prove matching tensors, and a different waveform does not by itself mean
worse speech. The diagnostic supports Qwen2.5 ASR and VoxCPM2:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra train python scripts/diagnose_merge.py \
  --checkpoint outputs/run/checkpoint-00000128 --export outputs/export \
  --data outputs/data/validation.jsonl --output outputs/merge-check \
  --cases 8 --arm weights
```

Run additional arms in the same fresh diagnostic directory: `adapter_bf16`,
`export_bf16`, `adapter_fp32`, `merged_fp32`, `export_fp32`, and
`runtime_merge_bf16`. Each arm saves fixed-input outputs and repeats its forward
pass with the same RNG. Compare completed arms with
`scripts/summarize_merge_diagnostics.py --root outputs/merge-check
--left adapter_bf16 --right export_bf16 --output outputs/merge-comparison.json`.
Use `--sample-ids selection.json` with a JSON list and matching `--cases` to
reproduce particular failures. Every arm must use the same selection.

`merged_fp32` isolates merging arithmetic before native export storage rounding.
Loading a BF16 export in FP32 cannot recover discarded precision. These are
numerical diagnostics; use the full paired evaluation to assess task quality.

## Capability-aware mixed batches

`heterogeneous_batches: true` opts understanding training into compatible text,
image and audio batches for `qwen2_5_omni` and `minicpm_o_4_5`. The default is
`false`; check the numerical behavior of your dtype, attention implementation and
recipe before enabling it. This option does not enable sequence packing.

The planner keeps bounded windows, resource limits and deterministic cursor
ordering. Model revisions, processors and training recipes remain separate
contracts. Qwen concatenates sparse media tensors and uses native padded attention.
MiniCPM batches the language model after encoding vision per sample and audio per
clip; its nested media bounds are shifted with left padding. Video and timed
attention use explicit sample-local fallback. Speech, flow and other tasks keep
conservative execution declarations; understanding support does not imply Talker,
flow or multistream batching support.

`metrics.jsonl` includes `batch_execution`: effective mode, fallback reason,
media stage execution and forward group sizes. `budget_cost` contains useful
versus conservatively allocated token/media/latent/event costs, rather than a
measurement of tensor memory. Existing slot counters measure the executed batch
layouts (`useful_llm_slots`, `allocated_llm_slots`), while
`cuda_peak_allocated_bytes` and `cuda_peak_reserved_bytes` are measured CUDA peaks.
Production `budget_cost.estimated_gpu_bytes` reuses the validated memory profile
and packing/capability/timed policy; without a profile this estimate is null.
Neither counter establishes a throughput improvement or an OOM guarantee.

For a native fixture regression, run from the project checkout:

```bash
python scripts/check_batch_contract.py --config train.yaml --output outputs/batch-contract.json
```

The default FP32 tolerance is `atol=rtol=2e-5`. The check compares objective sums,
normalization measures and all trainable gradients against independent execution,
and records native decoder call shapes. Supply an explicit tolerance for a
low-precision investigation; a passing FP32 test does not establish BF16 parity.

MiniCPM's native BF16 decoder can exceed strict independent-versus-batch gradient
tolerances even with correct padding and media alignment. For a more expensive
mixed computation alternative, explicitly set:

```yaml
dtype: bfloat16
attention: sdpa
minicpm_precision: fp32_decoder_compute
# Independent opt-in; remains false by default.
heterogeneous_batches: true
```

This policy keeps base weights and media encoding in BF16, but computes the
entire language decoder in FP32, including attention, residuals, projections and
the output head. LoRA parameters and gradients are FP32. It is **not native BF16
numerical parity**. Transient FP32 weight copies and larger activations increase
memory and computation; checkpointing reduces retained copies during training.
Do not reuse a memory calibration profile from the native policy. TF32 matmul
must remain disabled (PyTorch's `highest` float32 matmul precision); the runtime
rejects TF32 rather than changing global settings.

The configuration is limited to untimed CUDA/BF16 MiniCPM understanding SFT with
language-model LoRA only and SDPA. Offline video retains sample-local fallback.
Speech, timed attention and projector training are outside this policy's scope.
The choice is recorded in the execution/resume contract and export metadata;
native tensor keys and training-time BF16 base storage are preserved. The
existing merged export still loads/merges on CPU in FP32 to retain small LoRA
updates; this policy does not change that export rule. OmniFactory reload
reapplies the computation policy with the requested loading dtype. An upstream loader without that policy uses its own
computation precision. Validate tolerances for your weights, hardware and recipe;
this option makes no general quality, speed or minimum-memory guarantee.
