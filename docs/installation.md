# Installation

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Run the commands below from the OmniFactory checkout. Linux is the GPU execution platform; a CPU environment is sufficient for the small local workflow.

## Choose dependencies

```bash
# Dataset schemas, import, planning, and basic CLI tools.
uv sync --locked

# Model preparation, training, export, and local inference.
uv sync --locked --extra train
uv run --extra train omnifactory doctor
```

For the documented NF4 recipes, add the CUDA extra on a CUDA machine:

```bash
uv sync --locked --extra train --extra cuda
```

The training stack pins Transformers to **4.57.6**. Choose PyTorch/CUDA wheels and optional native extensions compatible with your GPU and Python ABI. A resolved Python dependency list does not guarantee that a particular CUDA wheel or FlashAttention build works on your host. Check your actual recipe before training:

```bash
uv run --extra train omnifactory init --recipe qwen2.5-omni-3b --output outputs/qwen-first
uv run --extra train omnifactory doctor --config outputs/qwen-first/train.yaml --strict
```

Use the same extras on subsequent `uv run` commands. If a command in a model guide is shown as `omnifactory`, run it through `uv run --extra train` or activate an environment with the required dependencies already installed.

The Qwen2.5-Omni 3B walkthrough was checked on Linux with Python 3.13 and PyTorch 2.10.0+cu128, TorchVision 0.25.0 and TorchAudio 2.10.0. If your driver requires a different CUDA build than the lockfile resolves, use the wheel installation below and select a compatible PyTorch, TorchVision and TorchAudio set in that environment. Run `doctor --strict` with the generated recipe before training.

## Install a built package

The `init` recipes and runtime data ship inside the wheel. Build from the checkout, then install the wheel in a fresh directory outside it:

```bash
uv build --wheel
wheel="$PWD/dist/omnifactory-0.1.0a8-py3-none-any.whl"
mkdir -p ../omnifactory-first-run
cd ../omnifactory-first-run
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "${wheel}[train]"
source .venv/bin/activate
omnifactory doctor
```

Match the wheel filename to the built version. Continue with the [pretrained workflow](getting_started.md#1-install-and-initialize) in this directory. Scripts under `scripts/` require a checkout. Model weights are downloaded separately or supplied as a complete local snapshot. For network-restricted environments, install the required packages and model assets before starting an offline workflow.

## Audio and video

Install FFmpeg and ensure both `ffmpeg` and `ffprobe` are on `PATH` for video input and generated audiovisual fixtures.

MiniCPM speech and Token2Wav additionally require the speech extra and explicit native asset download:

```bash
uv sync --locked --extra train --extra speech
uv run --extra train --extra speech omnifactory models fetch-speech-assets \
  --output outputs/speech-assets
```

Set the model recipe's tokenizer and decoder paths to the downloaded assets. Model weights are not bundled with OmniFactory; retain the source model's license. Qwen speech uses its own native labels/decoder and does not use MiniCPM assets. MiMo requires its matching codec snapshot. See [Model Recipes](models.md).

The CosyVoice3 adapter requires its own optional dependencies:

```bash
uv sync --locked --extra train --extra cosyvoice
```

Use the same extras for subsequent commands. Its checkpoint must contain the native speech tokenizer and speaker ONNX files, LLM, flow, vocoder and tokenizer/config assets; MiniCPM assets cannot substitute for them.

## Image editing

The Qwen-Image-Edit adapter additionally requires the pinned Diffusers
runtime. Use both extras for preparation, training, export and native inference:

```bash
uv sync --locked --extra train --extra image
```

## Duplex speech

PersonaPlex requires native audio and tokenizer dependencies:

```bash
uv sync --locked --extra train --extra personaplex
```

Accept the official checkpoint access terms and authenticate with Hugging Face before
downloading. Keep tokens in the client credential store, outside recipes and datasets.

## Keep serving separate

Install vLLM-Omni in its own environment. Its PyTorch and Transformers requirements can differ from the training stack. The supplied stage deployment configurations target upstream commit `0f79bd8b0e5ff9379a9a38e0c585db8f8335bf7c`; review the pinned upstream requirements when building that environment.

See [Export and Deployment](deployment.md) for running a trained model and checking its API. Training, CPU export, and serving have different memory requirements; choose budgets and hardware for each stage.

## Record your environment

```bash
uv run --extra train omnifactory capture-environment --output outputs/environment
```

This records the active distribution versions and Python/platform without serializing credentials or package-index URLs. Keep the record alongside your training recipe and pinned model revision. It is a version inventory, not an installer for CUDA-native binaries.

## External adapter example

The existing `omnifactory.adapters` entry-point group loads external adapter
specifications lazily. Discovery works with the base installation. Native encoding,
training and regression require the `train` extra.

From an independent OmniFactory checkout, install the runnable onboarding fixture:

```bash
python -m pip install '.[train]' ./examples/external_adapter
cd /tmp
omnifactory-example-check --output /tmp/omnifactory-example-run
```

The example creates random tiny Qwen weights and procedural text/image/audio data,
checks the shared batching/loss/gradient contract, then runs public prepare,
training, exact resume, export and native inference acceptance. It demonstrates
plugin packaging and hook reuse; it is not a newly supported pretrained model.
Its configuration is `examples/external_adapter/config.yaml` in the checkout.
No sibling repositories or engine edits are required. Choose a new output directory
for each run.

A plugin returns `AdapterSpec`, including one `TaskBatchCapability` per task,
loader, loss components, native exporter and artifact validator. Declarations and
encoded supervision are validated before training. Use
`omnifactory.adapters.contracts.check_adapter_contract` with native fixtures for
shared regression and its `export_reload` callback for a native round trip.
Continuous custom/flow objectives require a controlled `forward_probe(model, batch)`
with fixed per-sample noise and time fixtures; arbitrary RNG reseeding does not
prove equivalence. When combining `forward_probe` with `export_reload`, also supply
`reload_probe_factory(reloaded_adapter)`. It returns the equivalent controlled
`probe(model, batch)` bound to the new adapter; the probe must use its explicit
model argument. This avoids accidentally testing the source model or adapter
again. Missing factories are rejected. Discrete, continuous and timed supervision retain separate
representations and normalization units.
