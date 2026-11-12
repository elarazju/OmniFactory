# Export and Deployment

Export a training checkpoint before standalone inference. Training checkpoints contain optimizer and resume state; exports contain the weights and assets required for prediction.

## Export formats

| Format | Use |
| --- | --- |
| `merged` | Merge adaptation weights and save native model assets for local inference |
| `understanding` | Export the supported understanding-only path; excludes speech generation |
| `vllm-omni` | Save a supported complete native pipeline and deployment metadata for vLLM-Omni |
| `sglang-omni` | Save a supported complete native pipeline for SGLang-Omni; see the model/configuration table below |

Available formats depend on the adapter. The standalone MiniCPM Token2Wav flow stage is not a complete vLLM-Omni pipeline. Speech exports require the full base checkpoint and frozen decoder assets. See [Model Recipes](models.md) before selecting a format.

Serving exports retain each model's native weights and assets. The backend name
selects an integration target; it does not introduce a separate tensor format.
Export integrity and successful deployment are checked separately.

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

`deploy validate` defaults to `--backend vllm-omni`. Select `--backend sglang-omni`
for the five native model integrations described below. Reports record the selected backend,
launch command, deployment configuration hash and exact export artifact ID.
The export's verification flags remain false; a successful deployment is recorded
in the separate acceptance report.

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

### PersonaPlex duplex on vLLM-Omni

For PersonaPlex, install the `personaplex` extra in the validation client environment.
The validator streams the user recording at 80 ms intervals and requires output
while input is still arriving, continuous response identity, native scheduler/KV
evidence and complete 24 kHz frame coverage of the original recording. One explicit
80 ms silent drain tick completes delayed residual codebooks; it is counted
separately and cannot establish input/output overlap. It saves request/response events and
audio. Passing an offline WAV request does not substitute for this duplex check.

## SGLang-Omni serving

The integration covers five native model families at upstream revision
`eeb418d210bde215ca3905b5aa0519dad43746d9`. Select the configuration for the
exported model; a configuration for another family is rejected.

| Model | Configuration | Supported export/task | Validation API |
| --- | --- | --- | --- |
| Qwen3-TTS | [Qwen3-TTS](../examples/deploy-qwen3-tts-sglang.yaml) | Single-speaker CustomVoice `speech_sft` | `/v1/audio/speech` |
| Qwen3-Omni (Ethan voice) | [Qwen3-Omni](../examples/deploy-qwen3-omni-sglang.yaml) | Complete native `understanding_sft` or `speech_sft` | `/v1/chat/completions` |
| MiniCPM-o 4.5 | [MiniCPM-o](../examples/deploy-minicpm-o-4_5-sglang.yaml) | Complete native `understanding_sft` or `speech_sft` | `/v1/chat/completions` |
| CosyVoice3 | [CosyVoice3](../examples/deploy-cosyvoice3-sglang.yaml) | Complete native `speech_sft` or `flow_sft` | `/v1/audio/speech` |
| PersonaPlex | [PersonaPlex](../examples/deploy-personaplex-sglang.yaml) | Native `speech_sft`, packaged voice, full duplex | `/v1/realtime` WebSocket |

Bundled verification records describe the exercised model paths and their limits.
Configuration availability does not prove that your own exported weights have
passed GPU validation; run acceptance against that exact export. Models outside this table, Qwen3-TTS Base reference cloning
and VoiceDesign, and standalone MiniCPM Token2Wav exports are not covered.

### Serving prerequisites

Use a separate CUDA serving environment and follow the pinned upstream
[installation instructions](https://github.com/sgl-project/sglang-omni/blob/eeb418d210bde215ca3905b5aa0519dad43746d9/docs/get_started/installation.md).
Its executable is `sgl-omni`. Keep the training environment's dependency versions
separate. Install each model's prerequisites in the serving environment:

- **Qwen3-TTS:** follow the [Qwen3-TTS recipe](https://github.com/sgl-project/sglang-omni/blob/eeb418d210bde215ca3905b5aa0519dad43746d9/docs/cookbook/qwen3_tts.md),
  including `qwen-tts==0.1.1` installed with `--no-deps`. The export must include
  the embedded fixed voice and speech tokenizer.
- **Qwen3-Omni:** follow the [Qwen3-Omni recipe](https://github.com/sgl-project/sglang-omni/blob/eeb418d210bde215ca3905b5aa0519dad43746d9/docs/cookbook/qwen3_omni.md).
  Preserve the thinker, talker and code2wav weights. The supplied configuration
  places the full speech pipeline on one GPU; adjust the stage memory allocations
  together for your hardware. An understanding-only export is insufficient.
- **MiniCPM-o 4.5:** install the pinned checkout's `minicpm-o` extra and follow its
  [MiniCPM-o recipe](https://github.com/sgl-project/sglang-omni/blob/eeb418d210bde215ca3905b5aa0519dad43746d9/docs/cookbook/minicpm_o.md).
  Preserve `llm` and `tts` weights and the complete `assets/token2wav` decoder.
  Speech requests need either a dataset `reference_audio` condition without a
  transcript or the exported `assets/HT_ref_audio.wav` default reference.
- **CosyVoice3:** install `sox` and the pinned checkout's `fun-cosyvoice3` extra.
  Follow the [CosyVoice3 recipe](https://github.com/sgl-project/sglang-omni/blob/eeb418d210bde215ca3905b5aa0519dad43746d9/docs/cookbook/fun_cosyvoice3.md)
  to add CosyVoice revision `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` and Matcha-TTS
  revision `dd9105b34bf2be2230f4aa1e4769fb586a3c824e` to `PYTHONPATH`. Use the serving
  environment's dependency pins, not CosyVoice's separate requirements file.
  Both training stages export the full `llm.pt`, `flow.pt`, HiFT, ONNX frontends
  and native tokenizer assets.
- **PersonaPlex:** obtain the complete licensed native checkpoint and follow the
  [PersonaPlex recipe](https://github.com/sgl-project/sglang-omni/blob/eeb418d210bde215ca3905b5aa0519dad43746d9/docs/cookbook/personaplex.md).
  Keep the Mimi codec, SentencePiece tokenizer and packaged voice presets. A new
  `sglang-omni` export unpacks voices before sealing so the server does not change
  the artifact on first use. Install OmniFactory's `personaplex` extra in the
  validation client environment as well.

For video requests, FFmpeg/ffprobe must be on the validation client's `PATH`,
and FFmpeg shared libraries must also be discoverable by the serving environment's
TorchCodec. With a separate FFmpeg installation prefix, configure its `bin` on
`PATH` and its `lib` on the service process's library search path.

### Export and validate

Export in the training environment. For example, for Qwen3-TTS:

```bash
uv run --extra train omnifactory export \
  --checkpoint outputs/qwen3-tts/run/checkpoint-00000016 \
  --output outputs/qwen3-tts-sglang --format sglang-omni
uv run --extra train omnifactory deploy verify-export \
  --model outputs/qwen3-tts-sglang
```

Use the same `sglang-omni` format for the other model families, replacing the
checkpoint and output paths with those from your run. Install the adapter extras
from [Model Recipes](models.md) when exporting. CosyVoice3 speech and flow
checkpoints each produce a complete speech pipeline; validate each exported
checkpoint separately. Existing complete native exports may be reusable, but
PersonaPlex must already contain the selected unpacked `voices/<name>.pt` preset.
Re-export archive-only PersonaPlex assets before serving.

Run acceptance from the OmniFactory checkout with the matching configuration:

```bash
uv run --extra train omnifactory deploy validate \
  --backend sglang-omni \
  --model outputs/qwen3-tts-sglang \
  --dataset outputs/qwen3-tts/data/validation.jsonl \
  --output outputs/qwen3-tts-sglang-check \
  --executable /path/to/sglang-env/bin/sgl-omni \
  --deploy-config examples/deploy-qwen3-tts-sglang.yaml \
  --startup-timeout 900 --max-audio-units 512
```

For another family, replace `--model`, `--dataset`, `--output` and `--deploy-config`
using the table above. Add `--extra personaplex` to `uv run` for PersonaPlex.
The validator starts its own server on a loopback port, preserves requests,
responses and audio, writes `acceptance.json`, stops that server, and verifies
that the export still matches its manifest. For PersonaPlex it supplies
`--enable-realtime` automatically.

Choose `--max-audio-units` for the model and input: it bounds speech generation
for TTS/chat speech, and bounds the number of 80 ms input frames for PersonaPlex.
A recording longer than the PersonaPlex bound is rejected rather than cropped.

### Requests and completion checks

**Qwen3-TTS and CosyVoice3.** The validator requires HTTP 200, finite nonempty,
non-silent decoded audio, and `X-Finish-Reason: stop`. Missing stop metadata or a
generation-limit finish fails acceptance. Set a bound large enough for the
utterance; receiving a WAV alone does not prove completion.

Qwen3-TTS requests use `voice: <speech_speaker>` and `task_type: CustomVoice` with
literal user text. Do not send reference audio to this fixed-voice export.
CosyVoice3 requires one reference recording and its literal transcript, supplied
as a dataset condition or bundled during training. The client sends the complete
native prompt as `ref_text`: the system instruction, `<|endofprompt|>`, then the
reference transcript. The pinned upstream accepts this prompt verbatim and
rejects a separate nonempty `instructions` field alongside `ref_text`.

**Qwen3-Omni and MiniCPM-o.** The client sends prediction-visible text, image,
audio and video content to `/v1/chat/completions`, preserving video soundtrack
selection. Timeline placement and unsupported conditioning are rejected.
Exports trained with `understanding_sft` are checked for
text responses. Speech-training exports are checked with both text-only and
text-plus-audio requests for each prediction.

Every chat choice must return nonempty text or a transcript and
`finish_reason: stop`. Speech responses must also contain decodable, finite,
non-silent WAV audio. The pinned upstream's chat finish reason describes the
**thinker**, not the speech decoder. Audio is checked against the maximum
waveform length implied by the requested speech-token budget; hitting that cap
fails acceptance. Staying below the cap does not prove speech EOS. These results
establish chat execution and bounded audio output, not verified natural speech
completion from a separate talker EOS signal.

Audio and video slices are materialized before transmission. Video crops require
FFmpeg/ffprobe; selected source frame timestamps and soundtrack sample counts
are checked, and excluded soundtracks are removed. Cropping is bounded to source
videos of at most 60 seconds, 4 Mi pixels per frame, 120 fps and 512 MiB of decoded
video, within the 16 MiB inline request budget. Pre-trim larger sources into
independent assets. Explicit timeline placements remain unsupported by this chat path.

**PersonaPlex.** The validator uses the native session configuration, not the
offline HTTP pipeline. It runs one separately recorded warmup session using the
first input; warmup is excluded from request counts and acceptance evidence.
Each measured session streams 24 kHz PCM16 input every 80 ms and requires audio
to arrive before the final real input frame is sent. It sends no synthetic drain
frames. After `sglang.input_audio.end`, the server must confirm all input consumed
with zero discarded frames, complete one continuous response with status
`completed` and reason `input_audio_ended`, and close the session normally.
The pinned model emits a 160 ms first output unit followed by 80 ms units, so N
input frames require N+1 output frames. A single offline WAV response cannot
satisfy this check. Completion means the bounded input session ended; it does
not mean the model reached an utterance EOS.

All checks above establish the reported execution and protocol properties.
Assess intelligibility, answer quality and speaker similarity separately on
representative data.

### Persistent serving

For a persistent server, launch directly in the serving environment:

```bash
/path/to/sglang-env/bin/sgl-omni serve \
  --model-path outputs/qwen3-tts-sglang \
  --config examples/deploy-qwen3-tts-sglang.yaml \
  --host 127.0.0.1 --port 8093 --model-name qwen3-tts
```

Select the matching export and configuration for other families. PersonaPlex
also requires `--enable-realtime` and uses `PersonaPlexSessionPipelineConfig`;
its endpoint is `/v1/realtime`. Consult the pinned
[full-duplex guide](https://github.com/sgl-project/sglang-omni/blob/eeb418d210bde215ca3905b5aa0519dad43746d9/docs/developer_reference/full_duplex_guide.md#native-full-duplex-personaplex)
for client session negotiation and events.

For direct launches, set the YAML's top-level `model_path` to the export directory.
`deploy validate` writes a resolved `deploy.yaml` with the verified directory
before starting SGLang-Omni, whose config parser requires this field even when
`--model-path` is supplied. Stage-level `model_path` overrides are rejected so
they cannot select another checkpoint. The configurations bound concurrency for
initial validation; tune memory allocations and graph settings for your hardware.
