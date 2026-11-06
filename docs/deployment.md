# Export and Deployment

Export a training checkpoint before standalone inference. Training checkpoints contain optimizer and resume state; exports contain the weights and assets required for prediction.

## Export formats

| Format | Use |
| --- | --- |
| `merged` | Merge adaptation weights and save native model assets for local inference |
| `understanding` | Export the supported understanding-only path; excludes speech generation |
| `vllm-omni` | Save a supported complete native pipeline and deployment metadata for vLLM-Omni |

Available formats depend on the adapter. The standalone MiniCPM Token2Wav flow stage is not a complete vLLM-Omni pipeline. Speech exports require the full base checkpoint and frozen decoder assets. See [Model Recipes](models.md) before selecting a format.

To prepare the [quickstart](getting_started.md) Qwen2.5-Omni checkpoint for vLLM-Omni:

```bash
uv run --extra train omnifactory export \
  --checkpoint outputs/qwen-first/acceptance/run/checkpoint-00000002 \
  --output outputs/qwen-vllm-export --format vllm-omni
uv run --extra train omnifactory deploy verify-export --model outputs/qwen-vllm-export
```

If you followed the individual stages in Getting Started, use `outputs/qwen-first/resumed/checkpoint-00000002`. Export manifests bind model files and required assets to their content hashes. Preserve the complete directory when moving an export.

Compare adapter and exported generation with the same inputs, preprocessing,
generation limits and seeds. BF16 LoRA merging changes rounding and operation
order, so audio waveforms can differ even when every exported weight is correct.
For Qwen2.5 ASR and VoxCPM2, use the [merge diagnostics](training.md#diagnose-differences-after-merging)
to separate weight errors, numerical differences and task-level changes.

## Local inference

Use the native export and held-out fixtures already produced by the quickstart:

```bash
uv run --extra train omnifactory infer \
  --model outputs/qwen-first/acceptance/export \
  --dataset outputs/qwen-first/data/validation.jsonl --root outputs/qwen-first/data \
  --output outputs/qwen-first/native-predictions.jsonl --device cuda --dtype bfloat16
```

Add `--generate-audio` for a supported speech export or `--generate-image` for BAGEL. For speech, `--speaker` selects a native named speaker where supported; `--reference-audio` supplies independent voice conditioning only where supported. Consult the [model-specific examples](models.md) for generation bounds and prerequisites. Utterance generation that reaches its bound without EOS is incomplete, even if an audio file was written. PersonaPlex instead completes at the end of its bounded input session and has no utterance EOS.

## vLLM-Omni serving

Use a **separate serving environment** with compatible vLLM-Omni dependencies. The supplied deployment configurations target upstream revision `0f79bd8b0e5ff9379a9a38e0c585db8f8335bf7c`; compatibility with other revisions requires checking their model and configuration interfaces. A trainable component does not automatically constitute a complete serving pipeline.

Example deployment configurations:

| Pipeline | Configuration | Request type |
| --- | --- | --- |
| Qwen2.5-Omni | [Qwen2.5](../examples/serve-qwen2.5-omni-single-gpu.yaml) | Multimodal chat / speech |
| Qwen3-Omni | [Qwen3](../examples/serve-qwen3-omni-single-gpu.yaml) | Multimodal chat / speech |
| MiniCPM-o | [MiniCPM](../examples/serve-minicpm-o-4.5-single-gpu.yaml) | Multimodal chat / speech |
| MiMo-Audio | [MiMo](../examples/serve-mimo-audio-single-gpu.yaml) | Text/audio chat / speech |
| VoxCPM2 | [VoxCPM2](../examples/deploy-voxcpm2.yaml) | `/v1/audio/speech` |
| BAGEL | [BAGEL](../examples/deploy-bagel.yaml) | `/v1/images/generations` |
| Qwen3-TTS | [Qwen3-TTS](../examples/deploy-qwen3-tts.yaml) | `/v1/audio/speech` |
| CosyVoice3 | [CosyVoice3](../examples/deploy-cosyvoice3.yaml) | `/v1/audio/speech` |
| Qwen-Image-Edit | [Qwen-Image-Edit](../examples/deploy-qwen-image-edit.yaml) | `/v1/images/edits` |
| PersonaPlex | [PersonaPlex](../examples/deploy-personaplex.yaml) | `/v1/realtime?duplex=1` WebSocket |

For Qwen3, MiniCPM, and MiMo, replace `/path/to/vllm-omni` in the YAML’s `base_config` with the absolute path of your pinned upstream checkout. These overlays inherit upstream pipeline definitions. The repositories can be located independently.

Adjust device allocation, memory budgets, and request limits for your hardware. Launch from the repository root using the serving environment's executable:

```bash
/path/to/serving-env/bin/vllm serve outputs/qwen-vllm-export --omni \
  --deploy-config examples/serve-qwen2.5-omni-single-gpu.yaml \
  --host 127.0.0.1 --port 8091 --served-model-name qwen-first
```

### MiMo serving

MiMo additionally requires the codec bundled with its native export and the chat template from the matching vLLM-Omni checkout:

```bash
export MIMO_AUDIO_TOKENIZER_PATH="$PWD/outputs/mimo-export/assets/mimo_codec"
/path/to/serving-env/bin/vllm serve outputs/mimo-export --omni \
  --deploy-config examples/serve-mimo-audio-single-gpu.yaml \
  --chat-template /path/to/vllm-omni/examples/online_serving/mimo_audio/chat_template.jinja \
  --port 8092
```

## Check your deployment

Run `deploy validate` as an alternative to starting the server manually. It verifies the export, starts its own server on a loopback port, sends API requests, writes a report and stops that server. For the Qwen quickstart, it sends the held-out text, image and audio inputs to `/v1/chat/completions` and checks for text responses:

```bash
uv run --extra train omnifactory deploy validate \
  --model outputs/qwen-vllm-export --dataset outputs/qwen-first/data/validation.jsonl \
  --output outputs/deployment-check \
  --executable /path/to/serving-env/bin/vllm \
  --deploy-config examples/serve-qwen2.5-omni-single-gpu.yaml
```

The output directory contains `acceptance.json`, server logs, and request artifacts. Keep this report alongside the export. Its `artifact_id` identifies the exact export tested, while the export manifest records structural verification. Models requiring extra launch arguments, such as MiMo’s chat template, need a wrapper executable that supplies them. Use `deploy check` for text requests against an existing server and `deploy validate` to tie results to a specific artifact.

### Qwen video input and speech output

With the Qwen server running as shown above, use the repository's client and the generated video fixture. Fixture generation requires FFmpeg:

```bash
uv run --extra train omnifactory datasets omni-demo --output outputs/qwen-av-demo
uv run --extra train python scripts/validate_vllm_qwen.py --video-only \
  --url http://127.0.0.1:8091 --model qwen-first \
  --dataset outputs/qwen-av-demo/validation.jsonl --output outputs/qwen-av-check
```

The client sends the MP4 with its soundtrack to `/v1/chat/completions` once for text and once for text plus speech. It checks for nonempty text and a decodable, non-silent 24 kHz waveform on the speech request, then saves the requests, responses and WAV. The fixture checks execution. Use representative clips and a separate evaluation for task quality.

### PersonaPlex duplex

For PersonaPlex, install the `personaplex` extra in the validation client environment.
The validator streams the user recording at 80 ms intervals and requires output
while input is still arriving, continuous response identity, native scheduler/KV
evidence and complete 24 kHz frame coverage of the original recording. One explicit
80 ms silent drain tick completes delayed residual codebooks; it is counted
separately and cannot establish input/output overlap. It saves request/response events and
audio. Passing an offline WAV request does not substitute for this duplex check.
