# OmniFactory Documentation

Start with [Getting Started](getting_started.md) for a complete pretrained Qwen or VoxCPM2 workflow, then bring your own data. A CPU-only random-model setup check is also available.

| Guide | What you will find |
| --- | --- |
| [Installation](installation.md) | Python dependencies, CUDA, speech assets, and a separate serving environment |
| [Getting Started](getting_started.md) | A runnable first training, evaluation, export, and inference workflow |
| [Datasets](datasets.md) | Provided fixtures, Alpaca/TTS import, canonical multimodal data, and split rules |
| [Training](training.md) | Component LoRA, data reuse, batching, memory options, validation, and resume |
| [Model Recipes](models.md) | Supported tasks, model-specific prerequisites, configurations, and limits |
| [Export and Deployment](deployment.md) | Native exports, local text/audio/image inference, and vLLM-Omni serving |

The [project README](../README.md) lists supported model families and their training configurations. Paths in commands assume the repository root; paths in YAML resolve relative to that YAML. Example datasets test execution and are not useful training corpora.
