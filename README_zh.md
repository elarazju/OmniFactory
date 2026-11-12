<div align="center">

# OmniFactory

### Unified Efficient Fine-Tuning of Omni Models

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Pre--alpha-orange)](#快速开始)

[English](README.md) | [中文](README_zh.md)

</div>

OmniFactory 是一个面向**文本、图像、音频和视频模型的开源微调库**。使用统一流程准备数据、微调指定模型组件，并将支持的完整模型流水线导出，用于原生推理或 vLLM-Omni 部署。

**Pre-alpha · 0.1.0a8。** 请从已支持的模型配置开始使用。训练任务、显存优化和导出格式因模型而异。

[主要特性](#主要特性) · [快速开始](#快速开始) · [支持模型](#支持模型) · [训练能力](#训练能力) · [数据与示例](#数据与示例) · [环境要求](#环境要求) · [导出与部署](#导出与部署) · [文档](docs/README.md)

## 主要特性

- **跨模态微调：** 通过各模型的配置和统一数据格式，训练理解、语音生成或图像生成任务。
- **按需选择组件：** 为语言、视觉、音频和生成组件选择 LoRA 或全量微调，并设置独立学习率。
- **控制训练开销：** 配置模态预算、梯度累积和梯度检查点，使用适配器支持的显存优化。
- **复用已有工作：** 恢复中断的数据准备与训练，复用兼容的 prepared 数据，依据验证损失选择检查点。
- **训练后直接衔接推理：** 导出原生权重与必要资产，进行本地预测或部署到 vLLM-Omni。

## 快速开始

使用 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)，从仓库根目录开始：

```bash
uv sync --locked --extra train
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
omnifactory init --recipe qwen2.5-omni-3b --output outputs/qwen-first
omnifactory doctor --config outputs/qwen-first/train.yaml --strict
omnifactory acceptance run --config outputs/qwen-first/train.yaml --output outputs/qwen-first/acceptance
```

这条 CUDA 流程使用**真实预训练权重**和本地生成的测试数据，完成准备、两步训练、检查点恢复一致性校验、导出及重新加载后的推理。模型首次加载时下载固定版本权重；测试数据用于检查完整链路，不用于展示效果提升。

改用 `--recipe voxcpm2` 并选择新的输出目录，可以检查连续语音监督链路。已有完整模型快照时，可在 `init` 中加入 `--model /path/to/model`。这两个初始化配置随安装包提供，无需引用仓库内的脚本。

只有 CPU 时，可运行 `omnifactory smoke --output outputs/cpu-check --steps 12`，使用随机小模型检查环境。

[快速开始指南](docs/getting_started.md)包含产物说明、自有数据接入、分阶段命令与故障处理。其他模型见下表；YAML 中的相对路径均从该 YAML 所在目录解析。

## 支持模型

按希望训练的输入和目标选择模型族。每行汇总对应配置的训练能力。各任务接受的模态见[模型指南](docs/models.md)，服务支持范围见[部署指南](docs/deployment.md)。视频输入均为离线视频。

<!-- model-capabilities:start -->
| 模型族 | 训练输入 | 训练输出 | 微调方式 | 配置入口 |
| --- | --- | --- | --- | --- |
| Qwen2.5-Omni · 3B / 7B | 文本、图像、音频、视频 | 文本 / 语音 | LoRA / 全量；理解任务支持 NF4 | [理解](examples/train-qwen2.5-omni-3b-bf16.yaml) · [语音](examples/train-qwen-omni.yaml) · [完整配置](docs/models.md#qwen25-omni) |
| Qwen3-Omni · 30B-A3B-Instruct | 文本、图像、音频、视频 | 文本 / 语音 | LoRA / 指定组件全量 | [理解](examples/train-qwen3-omni.yaml) · [语音](examples/train-qwen3-omni-speech.yaml) |
| MiniCPM-o · 4.5 | 文本、图像、音频、视频 | 文本 / 语音 | LoRA / 全量 | [理解](examples/train-minicpm-o-4.5.yaml) · [语音](examples/train-minicpm-omni.yaml) |
| MiMo-Audio · 7B-Instruct | 文本、音频 | 文本 / 语音 | LoRA / 全量 | [语音](examples/train-mimo-audio.yaml) · [使用条件](docs/models.md#mimo-audio) |
| VoxCPM2 | 待合成原文；可选参考音频 | 语音 | LoRA / 全量 | [语音合成](examples/train-voxcpm2.yaml) · [使用条件](docs/models.md#voxcpm2) |
| BAGEL · 7B-MoT | 文本 | 图像 | 生成分支 LoRA / 全量 | [文生图](examples/train-bagel.yaml) · [使用条件](docs/models.md#bagel-dit) |
| Qwen3-TTS · 1.7B Base | 待合成原文；固定音色参考录音 | 语音 | LoRA / 指定组件全量 | [单说话人微调](examples/train-qwen3-tts.yaml) · [使用条件](docs/models.md#qwen3-tts) |
| CosyVoice3 · 0.5B-2512 | 待合成原文 | 语音 | LLM / flow LoRA 或指定组件全量 | [语音 LLM](examples/train-cosyvoice3.yaml) · [Flow](examples/train-cosyvoice3-flow.yaml) · [使用条件](docs/models.md#cosyvoice3) |
| Qwen-Image-Edit · 原始版 | 源图和编辑指令 | 编辑后图像 | DiT LoRA / 指定组件全量 | [图像编辑](examples/train-qwen-image-edit.yaml) · [使用条件](docs/models.md#qwen-image-edit) |
| PersonaPlex · 7B-v1 | 用户音频；系统角色文本 | 时间对齐文本 / 语音 | 组件 LoRA / 指定组件全量 | [双工语音](examples/train-personaplex.yaml) · [使用条件](docs/models.md#personaplex) |
<!-- model-capabilities:end -->

LoRA 和全量微调针对指定组件，仍受各模型冻结 codec 等限制。Qwen 语音训练需要匹配的原生标签。资产要求、组件选择及完整的 3B/7B 配置见[模型指南](docs/models.md)。

MiniCPM 另有[独立 Token2Wav flow 训练](docs/models.md#minicpm-token2wav-stage)与[离线时序注意力](docs/training.md#timed-attention)高级配置。独立解码器不导出完整 vLLM-Omni 流水线；时序注意力不提供实时双工交互。


PersonaPlex 支持基于同步用户/助手录音与显式文本时间标注的[双工语音训练](docs/models.md#personaplex)，可衔接原生流式推理和 WebSocket 部署。

## 训练能力

| 能力 | 当前范围 |
| --- | --- |
| 监督微调 | 按配置支持离散文本与语音 token、连续语音与图像目标，或时间对齐的多路流 |
| 组件 LoRA / 全量微调 | 指定语言、视觉、音频和生成组件，可分别配置学习率 |
| NF4 QLoRA | CUDA 上的 Qwen2.5-Omni 3B/7B 理解任务 |
| 显存与批次控制 | 梯度检查点、梯度累积、模态预算；部分 Qwen2.5 理解任务支持 Packing 与冻结特征缓存 |
| 混合模态批次 | Qwen2.5-Omni 和 MiniCPM-o 4.5 的文本、图像、音频理解任务可显式开启，见[精度与回退规则](docs/training.md#capability-aware-mixed-batches) |
| 数据复用 | 分片准备与中断恢复、增量复用、磁盘索引与有字节上限的预取 |
| 训练管理 | 学习率调度、验证损失、检查点保留与断点续训 |

具体选项与限制见[训练指南](docs/training.md)。当前工作流使用单设备，尚未实现分布式训练。

## 数据与示例

可以使用自己的授权数据，转换为统一的[多模态格式](docs/datasets.md#canonical-multimodal-data)，或使用以下工具：

| 入口 | 用途 |
| --- | --- |
| `datasets import-alpaca` | 导入文本指令数据 |
| `datasets import-tts` | 导入文本与录音配对数据，用于支持的 TTS 训练 |
| `datasets import-duplex` | 导入同步的用户/助手双路录音及显式文本时间标注，用于 PersonaPlex |
| `datasets qwen-speech` | 合成与源模型匹配的 Qwen 教师语音标签 |
| `datasets demo` / `omni-demo` | 生成理解与音视频工作流样例 |
| `datasets image-demo` / `flow-demo` / `mimo-demo` | 生成图像、连续语音、多码本流程样例 |

生成样例用于检查执行流程，不是有效训练语料或质量基准。Qwen 教师准备会合成目标音频，不是对任意录音进行编码。导入、划分及媒体要求见[数据指南](docs/datasets.md)。

## 环境要求

| 工作流 | 所需环境 |
| --- | --- |
| 本地环境检查 | Python 3.11+、CPU、`train` 依赖组 |
| 预训练模型微调 | Linux/CUDA 及对应模型依赖，显存与主机内存需满足具体任务 |
| NF4 训练 | Linux/CUDA 上的 `train` + `cuda` 依赖组 |
| MiniCPM 语音 / Token2Wav | `train` + `speech` 依赖组及原生语音资产 |
| 视频输入 | `PATH` 中可用的 FFmpeg 与 ffprobe |
| vLLM-Omni 部署 | 与部署配置兼容的独立推理环境 |

训练使用 Transformers **4.57.6**。请为实际环境匹配 PyTorch/CUDA 及可选的 FlashAttention 构建。显存需求取决于模型、可训练组件、媒体尺寸、序列长度与批次预算，目前尚未发布通用最低显存数字。详细步骤见[安装指南](docs/installation.md)。

## 导出与部署

独立推理可直接使用快速开始生成的 `outputs/qwen-first/acceptance/export`。要从同一次运行另行导出检查点：

```bash
uv run --extra train omnifactory export \
  --checkpoint outputs/qwen-first/acceptance/run/checkpoint-00000002 \
  --output outputs/qwen-export --format merged
```

请使用自己训练产生的检查点路径。导出格式包括 `merged`、`understanding` 和 `vllm-omni`，具体取决于适配器支持范围；Qwen3-Omni、MiniCPM-o 4.5、Qwen3-TTS CustomVoice、CosyVoice3 和 PersonaPlex 还支持 `sglang-omni`，前提条件和验收范围见部署指南。语音流水线需要完整基座权重与解码资产。原生文本/语音/图像推理、服务配置和导出校验见[导出与部署指南](docs/deployment.md)。

## 使用文档

| 指南 | 内容 |
| --- | --- |
| [安装](docs/installation.md) | 依赖、CUDA、语音资产、推理环境 |
| [快速开始](docs/getting_started.md) | 第一条完整工作流 |
| [数据](docs/datasets.md) | 导入工具、格式、模态和监督 |
| [训练](docs/training.md) | LoRA 策略、显存优化、数据准备、验证和续训 |
| [模型配置](docs/models.md) | 各模型族的前置条件与训练范围 |
| [导出与部署](docs/deployment.md) | 原生推理与 vLLM-Omni 服务 |

详细指南目前以英文维护。

## 参与贡献

欢迎反馈问题、提供可复现样例、完善适配器和修正文档。反馈时请附版本、模型修订版本、配置和相关报错，并移除私有数据及凭据。新增模型支持应明确训练任务、监督形式、导出格式和限制。

## 许可证与致谢

OmniFactory 使用 [Apache 2.0](LICENSE) 许可证。模型权重、数据集和第三方组件保留各自的许可证。

项目借鉴 LLaMA-Factory 与 ms-swift 的使用体验，基于 PyTorch、Transformers、PEFT 及各模型原生实现构建。感谢 vLLM-Omni 社区提供的部署生态。
