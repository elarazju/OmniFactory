<div align="center">

# OmniFactory

### Unified Efficient Fine-Tuning of Omni Models

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Pre--alpha-orange)](#getting-started)

[English](README.md) | [中文](README_zh.md)

</div>

OmniFactory is an open-source library for **fine-tuning Omni models across text, images, audio, and video**. Prepare your data, tune selected model components, and export supported pipelines for native inference or vLLM-Omni deployment.

**Pre-alpha · 0.1.0a8.** Start with a supported model recipe. Available training tasks, memory options, and export formats vary by model.

[Features](#features) · [Quickstart](#getting-started) · [Models](#supported-models) · [Training](#training-capabilities) · [Data](#data-and-examples) · [Requirements](#requirements) · [Deployment](#export-and-deployment) · [Docs](docs/README.md)

## Features

- **Train across modalities.** Fine-tune understanding, speech generation, or image generation with model-specific recipes and a shared data format.
- **Choose what to adapt.** Apply LoRA or full tuning to selected language, vision, audio, and generation components, with separate learning rates.
- **Control the training workload.** Set modality budgets and gradient accumulation; use checkpointing and the memory optimizations available for your adapter.
- **Reuse work across runs.** Resume data preparation and training, reuse compatible prepared data, and select checkpoints using validation loss.
- **Take your model to inference.** Export native weights and required assets, then run local predictions or serve a supported pipeline with vLLM-Omni.

## Getting Started

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked --extra train
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
omnifactory init --recipe qwen2.5-omni-3b --output outputs/qwen-first
omnifactory doctor --config outputs/qwen-first/train.yaml --strict
omnifactory acceptance run --config outputs/qwen-first/train.yaml --output outputs/qwen-first/acceptance
```

This CUDA workflow uses **pretrained weights** and locally generated fixtures to run preparation, two training steps, exact checkpoint recovery, export, and reloaded-model inference. The pinned weights download on first model load. The fixtures check the complete workflow; they are not a quality benchmark.

Use `--recipe voxcpm2` and a fresh output directory to check continuous speech supervision. Add `--model /path/to/model` to `init` for a complete local snapshot. Both initializer recipes ship in the installed package and work without repository scripts.

For a CPU-only setup check, run `omnifactory smoke --output outputs/cpu-check --steps 12` with tiny random weights.

Follow [Getting Started](docs/getting_started.md) for outputs, your own data, individual commands and troubleshooting. Other model recipes are listed below. Relative YAML paths resolve from that file's directory.

## Supported Models

Choose a model family by the input and target you want to train. Each row summarizes its linked recipes. Check the [Model Recipes](docs/models.md) guide for the modalities each task accepts and the [deployment guide](docs/deployment.md) for serving support. Video inputs are offline.

<!-- model-capabilities:start -->
| Model family | Training inputs | Training outputs | Fine-tuning | Recipes |
| --- | --- | --- | --- | --- |
| Qwen2.5-Omni · 3B / 7B | Text, image, audio, video | Text / speech | LoRA / full; NF4 for understanding | [Understanding](examples/train-qwen2.5-omni-3b-bf16.yaml) · [Speech](examples/train-qwen-omni.yaml) · [All recipes](docs/models.md#qwen25-omni) |
| Qwen3-Omni · 30B-A3B-Instruct | Text, image, audio, video | Text / speech | LoRA / selected full tuning | [Understanding](examples/train-qwen3-omni.yaml) · [Speech](examples/train-qwen3-omni-speech.yaml) |
| MiniCPM-o · 4.5 | Text, image, audio, video | Text / speech | LoRA / full | [Understanding](examples/train-minicpm-o-4.5.yaml) · [Speech](examples/train-minicpm-omni.yaml) |
| MiMo-Audio · 7B-Instruct | Text, audio | Text / speech | LoRA / full | [Speech](examples/train-mimo-audio.yaml) · [Setup](docs/models.md#mimo-audio) |
| VoxCPM2 | Utterance text; optional reference audio | Speech | LoRA / full | [TTS](examples/train-voxcpm2.yaml) · [Setup](docs/models.md#voxcpm2) |
| BAGEL · 7B-MoT | Text | Image | Generation-branch LoRA / full | [Text-to-image](examples/train-bagel.yaml) · [Setup](docs/models.md#bagel-dit) |
| Qwen3-TTS · 1.7B Base | Utterance text; fixed speaker reference | Speech | LoRA / selected full tuning | [Single-speaker TTS](examples/train-qwen3-tts.yaml) · [Setup](docs/models.md#qwen3-tts) |
| CosyVoice3 · 0.5B-2512 | Utterance text | Speech | LLM / flow LoRA or selected full tuning | [Speech LLM](examples/train-cosyvoice3.yaml) · [Flow](examples/train-cosyvoice3-flow.yaml) · [Setup](docs/models.md#cosyvoice3) |
| Qwen-Image-Edit · original | Source image and editing instruction | Edited image | DiT LoRA / selected full tuning | [Image editing](examples/train-qwen-image-edit.yaml) · [Setup](docs/models.md#qwen-image-edit) |
| PersonaPlex · 7B-v1 | User audio; system persona | Time-aligned text / speech | Component LoRA / selected full tuning | [Duplex speech](examples/train-personaplex.yaml) · [Setup](docs/models.md#personaplex) |
<!-- model-capabilities:end -->

LoRA and full tuning apply to selected components; frozen codecs and other model-specific exclusions still apply. Qwen speech requires matched native labels. See [Model Recipes](docs/models.md) for asset requirements, component choices, and all 3B/7B configurations.

MiniCPM also has advanced recipes for [standalone Token2Wav flow training](docs/models.md#minicpm-token2wav-stage) and [offline timed attention](docs/training.md#timed-attention). The standalone decoder does not export a complete vLLM-Omni pipeline, and timed attention does not provide live duplex interaction.


PersonaPlex supports [duplex speech training](docs/models.md#personaplex) from synchronized user/agent recordings and explicit text timing, with streaming native inference and WebSocket deployment.

## Training Capabilities

| Capability | Current scope |
| --- | --- |
| Supervised fine-tuning | Discrete text and speech tokens, continuous speech and image targets, or time-aligned streams, depending on the recipe |
| Component LoRA / full tuning | Selected language, vision, audio, and generation components; separate learning rates |
| NF4 QLoRA | Qwen2.5-Omni 3B/7B understanding on CUDA |
| Memory and batching | Gradient checkpointing, gradient accumulation, modality-aware budgets; packing and frozen-feature caching for selected Qwen2.5 understanding workloads |
| Mixed-modality batches | Opt-in text/image/audio understanding for Qwen2.5-Omni and MiniCPM-o 4.5; see [precision and fallback rules](docs/training.md#capability-aware-mixed-batches) |
| Data reuse | Resumable sharded preparation, incremental reuse, disk-backed indexes and byte-bounded prefetch |
| Training controls | Learning-rate schedules, validation loss, checkpoint retention, and resume |

See [Training](docs/training.md) for exact options and restrictions. The current workflow runs on a single device; distributed training is not implemented.

## Data and Examples

Bring your own licensed data using the shared [multimodal format](docs/datasets.md#canonical-multimodal-data), or use these tools:

| Entry point | Purpose |
| --- | --- |
| `datasets import-alpaca` | Import text instruction data |
| `datasets import-tts` | Import text/recording pairs for supported TTS training |
| `datasets import-duplex` | Import synchronized user/agent recordings with explicit text timing for PersonaPlex |
| `datasets qwen-speech` | Synthesize model-matched Qwen teacher speech labels |
| `datasets demo` / `omni-demo` | Generate understanding and audiovisual workflow fixtures |
| `datasets image-demo` / `flow-demo` / `mimo-demo` | Generate image, continuous speech, and multi-codebook fixtures |

Generated fixtures check execution; they are not training corpora or quality benchmarks. Qwen teacher preparation synthesizes target audio rather than encoding arbitrary recordings. See [Datasets](docs/datasets.md) for import, splitting, and media requirements.

## Requirements

| Workflow | Setup |
| --- | --- |
| Local setup check | Python 3.11+, CPU, `train` extra |
| Pretrained training | Linux/CUDA and the selected recipe's dependencies; GPU and host memory must fit its workload |
| NF4 training | `train` + `cuda` extras on Linux/CUDA |
| MiniCPM speech / Token2Wav | `train` + `speech` extras and native speech assets |
| Video input | FFmpeg and ffprobe on `PATH` |
| vLLM-Omni serving | A separate serving environment compatible with the deployment configuration |

Training uses Transformers **4.57.6**. Match PyTorch/CUDA and optional FlashAttention builds to your environment. Memory needs depend on the model, trainable components, media sizes, sequence lengths, and batch budgets; no general minimum-VRAM figure is published yet. See [Installation](docs/installation.md).

## Export and Deployment

For standalone inference from the quickstart, use `outputs/qwen-first/acceptance/export`. To export another checkpoint from that run:

```bash
uv run --extra train omnifactory export \
  --checkpoint outputs/qwen-first/acceptance/run/checkpoint-00000002 \
  --output outputs/qwen-export --format merged
```

Use the checkpoint path from your own run. Export formats include `merged`, `understanding`, and `vllm-omni`, subject to the adapter's supported scope. Speech pipelines need complete base weights and decoder assets. See [Export and Deployment](docs/deployment.md) for native text/audio/image inference, server configurations, and export verification.

## Documentation

| Guide | Topics |
| --- | --- |
| [Installation](docs/installation.md) | Dependencies, CUDA, speech assets, serving environment |
| [Getting Started](docs/getting_started.md) | Your first complete workflow |
| [Datasets](docs/datasets.md) | Importers, schemas, modalities, supervision |
| [Training](docs/training.md) | LoRA policies, memory options, preparation, validation, resume |
| [Model Recipes](docs/models.md) | Family-specific setup and supported tasks |
| [Export and Deployment](docs/deployment.md) | Native inference and vLLM-Omni serving |

## Contributing

Bug reports, reproducible examples, adapter improvements, and documentation fixes are welcome. Include your version, model revision, configuration, and relevant error output; remove private data and credentials. Model support should state its training task, supervision, export format, and limitations explicitly.

## License and Acknowledgements

OmniFactory is licensed under [Apache 2.0](LICENSE). Model weights, datasets, and bundled third-party components retain their respective licenses.

The project draws inspiration from LLaMA-Factory and ms-swift and builds on PyTorch, Transformers, PEFT, and the supported model implementations. Thanks to the vLLM-Omni community for the deployment ecosystem.
