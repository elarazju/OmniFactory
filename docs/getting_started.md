# Getting Started

Start with a pretrained Qwen2.5-Omni 3B model on CUDA. The first run uses generated text/image/audio fixtures to check preparation, training, exact checkpoint recovery, export, and inference. It runs two optimizer steps; it is not a recipe for improving model quality.

## 1. Install and initialize

From the repository root, install the training dependencies and activate the environment. If you installed a [wheel](installation.md#install-a-built-package), use its activated environment and start at the `export` command below.

```bash
uv sync --locked --extra train
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
omnifactory init --recipe qwen2.5-omni-3b --output outputs/qwen-first
```

`init` creates `train.yaml`, train/validation fixtures and `project.json`. The recipe pins the upstream model revision. The first workflow loads the processor and weights on demand after `doctor`. Allow disk space for both the model snapshot and the merged export, which can be larger than the source weights. See [Installation](installation.md) for CUDA dependencies.

The initializer and its recipes are included in the installed package, so the same commands work outside a source checkout. To use an existing complete local snapshot, add `--model /path/to/Qwen2.5-Omni-3B`. A custom remote repository requires `--model owner/repository --revision FULL_COMMIT_SHA` and must match the selected architecture. Use a new output directory each time.

## 2. Check the recipe and run the complete workflow

```bash
omnifactory doctor --config outputs/qwen-first/train.yaml --strict
omnifactory acceptance run \
  --config outputs/qwen-first/train.yaml \
  --output outputs/qwen-first/acceptance
```

`doctor` checks installed dependencies, the device, local paths or a pinned remote identity, dataset structure, train/validation group separation, and media file hashes. It does not load model weights or certify that an upstream repository is downloadable. The subsequent workflow validates native encoding and training.

The workflow prepares data, trains, resumes from the intermediate checkpoint, compares final weights **and optimizer/RNG/data-cursor state**, exports, verifies the export inventory, and generates predictions from the reloaded export. A failure leaves `acceptance.json` with the failed stage and diagnostic; successful earlier checkpoints remain available.

| Output below `outputs/qwen-first/acceptance/` | Contents |
| --- | --- |
| `prepared/` | Sharded tensors, on-disk index and saved processor |
| `run/` | Metrics and the original training checkpoints |
| `resumed/` | Checkpoints produced by exact recovery |
| `export/` | Merged/native weights, processor and export inventory |
| `predictions.jsonl` | Reloaded-model predictions |
| `acceptance.json` | Separate preparation, training, recovery, export and inference results |

The five workflow gates should say `passed`. Native-reference, HTTP deployment and quality gates remain `not_run` unless separately verified. This command does not start a vLLM-Omni server.

## 3. Try continuous speech training

Use the same installed environment with a fresh project:

```bash
omnifactory init --recipe voxcpm2 --output outputs/vox-first
omnifactory doctor --config outputs/vox-first/train.yaml --strict
omnifactory acceptance run \
  --config outputs/vox-first/train.yaml \
  --output outputs/vox-first/acceptance
```

This uses pretrained VoxCPM2 weights and its native flow and stop objectives. The generated tone fixtures verify the framework's continuous-supervision path; they are not speech training data. Reloaded audio outputs are linked from `predictions.jsonl`. See [VoxCPM2](models.md#voxcpm2) for real TTS data and component settings.

## 4. Run each stage separately

Use the project created in step 1. These commands write separate artifacts from the automated run in step 2. Prepare the training split and add `prepared_dataset: prepared` to `outputs/qwen-first/train.yaml` so training reuses it:

```bash
omnifactory prepare --config outputs/qwen-first/train.yaml --output outputs/qwen-first/prepared
```

```yaml
prepared_dataset: prepared
```

```bash
omnifactory train --config outputs/qwen-first/train.yaml
```

Copy `train.yaml` to `resume.yaml` in the same directory. In the copy, replace `output_dir` with `resumed` and add `resume_from`. Keep the other fields, including `max_steps: 2` and `prepared_dataset`:

```yaml
output_dir: resumed
resume_from: run/checkpoint-00000001
```

```bash
omnifactory train --config outputs/qwen-first/resume.yaml
omnifactory export \
  --checkpoint outputs/qwen-first/resumed/checkpoint-00000002 \
  --output outputs/qwen-first/export --format merged
omnifactory deploy verify-export --model outputs/qwen-first/export
omnifactory infer \
  --model outputs/qwen-first/export \
  --dataset outputs/qwen-first/data/validation.jsonl --root outputs/qwen-first/data \
  --output outputs/qwen-first/predictions.jsonl --device cuda --dtype bfloat16
```

The resumed run starts at step 1 and stops at step 2. Use `acceptance run` when you want it to compare the uninterrupted and resumed runs automatically. See [Training](training.md#prepare-once-and-reuse) for preparation recovery, incremental reuse and sharding.

## 5. Bring your own data

Import your licensed data into the [canonical format](datasets.md), then initialize with both splits:

```bash
omnifactory init --recipe qwen2.5-omni-3b --output outputs/my-project \
  --dataset /path/to/data/train.jsonl \
  --validation-dataset /path/to/data/validation.jsonl \
  --dataset-root /path/to/data --validation-root /path/to/data
```

Review `train.yaml`: extend the two-step setup check for your experiment, choose trainable components and budgets, and configure validation if needed. All relative YAML paths resolve from that file. `init` only creates files; it does not train or overwrite an existing project.

Run `doctor --strict` against the new YAML, then use the individual stages above with your project paths. Training prepares data automatically when `prepared_dataset` is omitted. Choose a checkpoint produced by your run and use your validation dataset and media root for inference.

## Optional: CPU setup check

Without a GPU or pretrained weights, run a tiny random-model workflow:

```bash
omnifactory smoke --output outputs/cpu-check --steps 12
```

This does not download pretrained weights. Its answers are not useful. To try a second tiny architecture, add `--adapter minicpm_o_4_5` and choose a different output directory.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Missing dataset/media or overlapping source groups | Fix the resolved paths or split source groups; rerun `doctor --strict` |
| Remote download fails | Check repository access and network, or supply a complete local snapshot with `--model`; `HF_ENDPOINT` can select a reachable compatible mirror |
| Interrupted preparation | Repeat `prepare` with the same output and `--resume`; only completed shards are reused |
| Prepared version/encoder mismatch | Prepare again into a new directory; v1–v3 artifacts are not loaded as v4 |
| Output directory exists | Choose a new run directory; use `resume_from` to recover training |
| Sample exceeds a resource budget | Inspect that sample and explicitly resize/segment it or adjust the relevant budget |

For other families, select a [Model Recipe](models.md). For serving, follow [Export and Deployment](deployment.md).
