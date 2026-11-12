# Model Recipes

Choose a supported task and start with its linked YAML. All recipes require your dataset paths, a pinned complete model revision or local snapshot, and suitable resource budgets. The [README model table](../README.md#supported-models) summarizes capabilities; it does not imply that every upstream variant or modality is trainable.

The common command is:

```bash
uv run --extra train omnifactory train --config examples/train-qwen2.5-omni-3b-bf16.yaml
```

Replace the recipe for your chosen model. See [Training](training.md) for components, data reuse, validation, and resume, and [Deployment](deployment.md) for output formats.

For a two-step check with generated fixtures, use the installed [Qwen2.5-Omni 3B or VoxCPM2 initializer](getting_started.md). The checkout YAMLs below are starting points for configuring your own run. Paths in YAML are relative to the file, and the linked guides describe any required labels or assets. `omnifactory models list` also includes architecture contracts, identified as `contract_only` when they have no training adapter.

For SGLang-Omni, complete native exports are available for Qwen3-Omni, MiniCPM-o 4.5, Qwen3-TTS CustomVoice, CosyVoice3 (speech and flow), and PersonaPlex. Use `--format sglang-omni` and the corresponding [deployment configuration and prerequisites](deployment.md#sglang-omni-serving). Other adapters are not integrated with this backend.

## Qwen2.5-Omni

- Inputs: text, images, audio, offline video; targets: text or text/native speech.
- Understanding: [3B BF16](../examples/train-qwen2.5-omni-3b-bf16.yaml), [3B NF4](../examples/train-qwen2.5-omni-3b.yaml), [7B BF16](../examples/train-qwen2.5-omni-7b.yaml), [7B NF4](../examples/train-qwen2.5-omni-7b-nf4.yaml).
- Speech: [3B Thinker/Talker](../examples/train-qwen-omni.yaml), [7B Thinker/Talker](../examples/train-qwen2.5-omni-7b-speech.yaml).
- ASR: [real-recording recipe](../examples/train-qwen2.5-asr-baseline.yaml) with an explicit transcript-only instruction and [paired evaluation](training.md#paired-real-data-baselines).
- Components include the language model, vision/audio encoders and projectors, and speech generator/projection. Speech decoding stays frozen.

NF4 is available for understanding, not Talker training. The text/image/audio packing and frozen-feature-cache options are also understanding-specific; see [memory options](training.md#memory-and-batching).

For video, set `video_fps`, frame/duration limits and image budgets. All videos in one prediction must share the same `audio_mode`; mixed included/excluded soundtracks within one prediction are rejected. Named speech speakers come from the checkpoint; the supplied speech recipes use Chelsie. Arbitrary reference-voice cloning is unsupported.

## Qwen3-Omni

The current variant is **Qwen3-Omni-30B-A3B-Instruct**. Inputs include text, image, audio, and offline video.

- [Understanding recipe](../examples/train-qwen3-omni.yaml): text targets, Thinker attention/expert and projector policies.
- [Speech recipe](../examples/train-qwen3-omni-speech.yaml): native Talker plus residual code predictor, with 16-codebook labels.
- [ASR recipe](../examples/train-qwen3-asr-pilot.yaml): speech input with transcription targets.

Routers and Code2Wav remain frozen. Speech components can have separate learning rates. NF4, packing, frozen-feature caching, live duplex, and arbitrary-waveform speech encoding are not enabled for this adapter. The default speech speaker is Ethan; use a voice present in the checkpoint.

## Qwen speech labels

Qwen2.5, Qwen3-Omni, Qwen3-TTS, and unrelated codecs are not interchangeable. Both supported Qwen Omni adapters require native labels bound to their source model and target sample. Use compatible labels or explicitly synthesize a teacher dataset.

For Qwen3, starting with fresh fixture paths:

```bash
uv run --extra train omnifactory datasets omni-demo --output outputs/omni-data
uv run --extra train omnifactory datasets qwen-speech \
  --config examples/prepare-qwen3-speech.yaml \
  --validation-dataset outputs/omni-data/validation.jsonl --output outputs/qwen3-speech-data
uv run --extra train omnifactory train --config examples/train-qwen3-omni-speech.yaml
```

The fixture requires FFmpeg. Replace it with real conversations and selected transcripts for useful training. In both preparation and training YAML, use the same pinned full model and speaker. For Qwen2.5, use the matching [3B](../examples/prepare-qwen2.5-omni-3b-speech.yaml) or [7B](../examples/prepare-qwen2.5-omni-7b-speech.yaml) preparation recipe and update its input/validation paths. Never reuse 3B teacher labels for 7B.

Set `speech_codes` to the resulting manifest. Editing transcripts, target crops, audio, speaker, or source identity requires new matching labels. Teacher synthesis replaces target WAVs and does not encode arbitrary recordings. Speech export requires `merged` or a supported serving format, preserving frozen decoder assets. Qwen3-Omni additionally supports `sglang-omni`; Qwen2.5-Omni does not.

## MiniCPM-o 4.5

- Inputs: text, image, audio, offline video; targets: selected assistant text or native text/speech.
- [Understanding recipe](../examples/train-minicpm-o-4.5.yaml), [speech recipe](../examples/train-minicpm-omni.yaml).
- Components: language model, vision encoder/resampler, audio encoder/projector, and speech stages. For audio-projector LoRA, use the native `linear1`/`linear2` targets.

Speech uses the model's native S3 tokenizer and Token2Wav assets. Install the `speech` extra and download assets as described in [Installation](installation.md#audio-and-video); update `speech_tokenizer`, its expected hash, and `speech_assets` in the recipe. Do not substitute another model's codec.

A local speech workflow with tiny random weights:

```bash
uv run --extra train --extra speech omnifactory smoke \
  --adapter minicpm_o_4_5 --speech-assets outputs/speech-assets \
  --output outputs/minicpm-speech-demo --steps 4
```

Use an independent reference recording for voice conditioning during native synthesis. `infer --reference-audio` selects it explicitly; target validation audio is never selected automatically. NF4, packed training, and live streaming/duplex are unsupported.

The [timed-attention recipe](../examples/train-minicpm-duplex.yaml) is an advanced offline option requiring an explicit event/token schedule. It does not implement real-time conversation scheduling or interruption. See [the schedule contract](training.md#timed-attention) before preparing data.

### MiniCPM Token2Wav stage

The separate [flow recipe](../examples/train-minicpm-flow.yaml) trains the native flow decoder on continuous mel targets, with a separate reference recording. Set its model path to the downloaded Token2Wav asset directory. `datasets flow-demo` supplies a generated execution fixture; replace tones with real data for useful training.

Its export contains decoder-stage assets rather than a full Omni model. Decode native S3 codes with:

```bash
uv run --extra train --extra speech omnifactory flow-decode \
  --assets outputs/flow-export --codes /path/to/native-s3-codes.json \
  --reference-audio /path/to/reference.wav --output outputs/flow.wav --device cuda
```

`--format vllm-omni` is not supported for this standalone stage. Full serving assembly requires the complete MiniCPM pipeline.

## MiMo-Audio

MiMo-Audio-7B-Instruct accepts text/audio and trains text or multi-codebook speech targets. It does not accept images/video.

Use the [MiMo recipe](../examples/train-mimo-audio.yaml) with complete pinned model and tokenizer snapshots. The frozen external codec is required during preparation and audio decoding. Compute its fingerprint and replace the recipe's placeholder:

```bash
uv run --extra train omnifactory models fingerprint --path /path/to/MiMo-Audio-Tokenizer
```

A speech prediction selects one transcript block and one associated audio block. The model's trainable audio encoder/projector is distinct from the frozen waveform codec. NF4, voice conditioning, assistant audio history, duplex, and packing are unsupported; codec extraction and the current native inference path run on CPU.

For a local random-model workflow:

```bash
uv run --extra train omnifactory smoke \
  --adapter mimo_audio --speech --steps 80 --output outputs/mimo-demo
```

MiMo uses explicit supervised-symbol normalization, which differs from the upstream trainer's channel-mean weighting. Loss values are not directly comparable across those trainers. Native export bundles the codec; serving additionally needs the MiMo chat template and `MIMO_AUDIO_TOKENIZER_PATH` described in [Deployment](deployment.md#mimo-serving).

## VoxCPM2

VoxCPM2 supports **literal utterance text → speech**, optionally conditioned on a separate reference recording. The text is what should be spoken, not a chat instruction. Training uses continuous AudioVAE targets, flow matching, and stop prediction; the native AudioVAE remains frozen.

Use [the speech recipe](../examples/train-voxcpm2.yaml) or [the recipe with validation and scheduling](../examples/train-voxcpm2-pilot.yaml). Import recordings with [the TTS importer](datasets.md#import-text-and-speech-pairs). Trainable components include the language/residual models, audio decoder, projections, and stop predictor, with separate policies and learning rates.

For real-recording comparisons, use the [baseline recipe](../examples/train-voxcpm2-baseline.yaml)
with explicit [text preparation](datasets.md#use-real-data), FP32 optimizer accumulation,
and [base/adapter/export evaluation](training.md#paired-real-data-baselines).

```bash
uv run --extra train omnifactory train --config examples/train-voxcpm2.yaml
uv run --extra train omnifactory export \
  --checkpoint outputs/voxcpm2-run/checkpoint-00000100 \
  --output outputs/voxcpm2-export --format vllm-omni
uv run --extra train omnifactory infer \
  --model outputs/voxcpm2-export --dataset outputs/tts-data/validation.jsonl \
  --output outputs/voxcpm2-predictions.jsonl \
  --device cuda --dtype bfloat16 --generate-audio --max-audio-units 128
```

For this adapter, the audio generation bound counts latent patches. It does not provide text answers, images/video, conversational speech, continuation prompts, NF4, cross-example packing, or duplex interaction.

## BAGEL DiT

The supported task is **one text prompt → one image**, using native continuous VAE flow-matching supervision. This adapter does not train image understanding/editing, interleaved generation, or video.

Use [the BAGEL recipe](../examples/train-bagel.yaml). `image_generator` selects generation-branch weights; latent projector, timestep embedder, and image head have separate policies. The understanding weights and VAE remain frozen. CUDA image generation needs a compatible FlashAttention build.

```bash
uv run --extra train omnifactory datasets image-demo --output outputs/image-data --size 256
uv run --extra train omnifactory train --config examples/train-bagel.yaml
uv run --extra train omnifactory export \
  --checkpoint outputs/bagel-run/checkpoint-00000100 \
  --output outputs/bagel-export --format vllm-omni
uv run --extra train omnifactory infer \
  --model outputs/bagel-export --dataset outputs/image-data/validation.jsonl \
  --output outputs/bagel-images.jsonl \
  --device cuda --dtype bfloat16 --generate-image --image-steps 24
```

Replace the procedural image fixture with your own text/image pairs. Canvas dimensions must fit the checkpoint's spatial grid and stride; the official model uses a stride of 16. The supplied recipe uses 256×256. Higher resolutions require larger attention and latent budgets. Native `image_steps` counts time-grid points, so N points perform N−1 Euler updates. NF4 and multi-example packing are unsupported.


## CosyVoice3

The `cosyvoice3` adapter targets **FunAudioLLM/Fun-CosyVoice3-0.5B-2512** at revision `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`. Install `train` and `cosyvoice` extras. Speech LLM and flow training export complete native pipelines for waveform inference and vLLM-Omni deployment.

- [Speech LLM recipe](../examples/train-cosyvoice3.yaml): literal text and paired recordings supervise native 25 Hz discrete speech tokens, with unistream or 5:15 text/speech interleaving. `cosyvoice_sequence_mode: mixed` chooses a reproducible layout per prepared sample; changing the mode requires re-preparation.
- [Flow recipe](../examples/train-cosyvoice3-flow.yaml): the native frozen frontend derives aligned 50 Hz, 80-bin mel targets, speech tokens and speaker embeddings. Train the DiT with `audio_decoder` LoRA (`to_q`, `to_v`) or selected full components. This stage does not train the speech LLM.

Use `datasets import-tts` to import exact transcripts with their recordings. Training records must select the paired transcript/audio targets and have no reference condition. Preparation computes frozen native features, preserves the discrete/continuous supervision distinction, and enforces audio duration, codec-frame and latent-element budgets. QLoRA and packed attention are not supported.

For generation, supply a per-record `reference_audio` condition, or configure a fixed `speech_reference` and `speech_reference_sha256` before training to bundle it with the export. Use a separate recording from the training and validation targets. The vLLM-Omni speech API also needs its exact transcript (`speech_reference_text` for a bundled reference or `transcript` on the condition). An optional system message supplies the synthesis instruction. The user message is the literal text to speak. Do not insert native control tokens into either text.

Both tasks export a native pipeline containing `llm.pt`, `flow.pt`, `hift.pt` and frontend assets. Only the trained stage changes; a fixed reference is bundled for portable inference. Use [the deployment recipe](../examples/deploy-cosyvoice3.yaml) with the exported pipeline. The current path is offline generation, not real-time duplex or streaming acceptance.

The flow recipe enables deterministic training for exact resume on a fixed environment:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra train --extra cosyvoice omnifactory train --config examples/train-cosyvoice3-flow.yaml
```

The pinned vLLM-Omni runtime uses `s3tokenizer` for reference conditioning. For offline serving, reuse the exported native tokenizer in its cache:

```bash
mkdir -p ~/.cache/s3tokenizer
cp outputs/cosyvoice3-export/speech_tokenizer_v3.onnx ~/.cache/s3tokenizer/speech_tokenizer_v3_25hz.onnx
```

If `XDG_CACHE_HOME` is set, use its `s3tokenizer` subdirectory instead. The tokenizer package verifies this asset's expected SHA-256.

## Qwen3-TTS

The `qwen3_tts` adapter starts from **Qwen3-TTS-12Hz-1.7B-Base** and exports a
single-speaker **CustomVoice** checkpoint. CPU tests cover the native 16-codebook
codec, causal supervision, LoRA training/resume, native export/reload and waveform
inference. The pretrained 1.7B checkpoint has also passed bounded BF16 LoRA
training, exact resume, native export/reload, complete-EOS waveform generation
and CustomVoice serving with vLLM-Omni and SGLang-Omni. These checks establish the execution path,
not speech quality improvement.

Use [the training configuration](../examples/train-qwen3-tts.yaml) with the `train`
dependency group. Before running `doctor`, set `speech_reference` to a separate recording of the target
speaker and set `speech_reference_sha256` to the first column from `sha256sum /path/to/reference.wav`.
`speech_speaker`
is the exported lowercase voice identifier. All training recordings must be of
that same speaker. Keep validation utterances separate from training and from
the reference; a same-speaker split is appropriate for this single-voice task.

Import literal utterance text plus its target recording with `datasets import-tts`
([format](datasets.md)). The bundled native 12 Hz tokenizer derives all 16
codebooks from the recording. The speaker encoder and codec remain frozen.
An optional per-record reference must match the full configured fixed recording;
reference cropping and continuation transcripts are unsupported. You may omit
per-record references and configure the fixed recording only in YAML.

Train `speech_generator` and `code_predictor` with LoRA or full component tuning.
Text projection/embedding and audio embedding are separately selectable components.
Primary-codebook CE includes one EOS; residual CE covers all 15 remaining codebooks.
There is one temporal label shift and no additional codebook shift. Prompt packing
matches the native non-streaming generation layout. Multi-speaker training,
VoiceDesign training, NF4 and streaming training are not implemented.

Use the shared prepare/train/export commands in [Getting Started](getting_started.md).
Export with `--format merged`, `--format vllm-omni` or `--format sglang-omni`; native prediction requires
`--generate-audio`. The export bundles the tokenizer/codec and embeds its voice,
so generation does not require the original fixed-reference file. Its speech API
request uses `task_type: CustomVoice` and `voice: <speech_speaker>`; do not send a
Base voice-cloning request to this export.

For SGLang-Omni, use the [CustomVoice deployment guide](deployment.md#sglang-omni-serving)
and [configuration](../examples/deploy-qwen3-tts-sglang.yaml). The same complete
native checkpoint can target either serving backend.

Native inference and the deployment recipe use the upstream sampling defaults
(temperature 0.9, top-k 50 for both Talker and residual codebooks). Forcing greedy
decoding may repeat frames without reaching EOS. The shared inference command
seeds its random generators; exact outputs can still differ across backends.

For deployment checks, pass `--max-audio-units 256` to `deploy validate` for longer
utterances. This is an output cap in codec frames (12.5 frames/s); increase the
stage budgets with it when necessary. The [deployment configuration](../examples/deploy-qwen3-tts.yaml)
has been exercised with the single-speaker export. A successful HTTP response alone
does not establish natural EOS or speech quality.

## Qwen-Image-Edit

The `qwen_image_edit` adapter targets the original **Qwen/Qwen-Image-Edit**
checkpoint, with native training, export and vLLM-Omni image-edit serving. EditPlus,
Edit-2509 and Edit-2511 are separate variants and are rejected by this adapter.

Use one source image and a literal edit instruction as input, with a separate
edited target image ([data format](datasets.md)). Preparation caches the frozen
Qwen2.5-VL semantic features and normalized VAE latents. The target image enters
only the VAE target path. Training loads the DiT without the frozen frontend and
supervises target patches with continuous flow matching; source patches remain
clean conditioning. Variable layouts execute separately within a budgeted batch.

Install `train` and `image` extras and start from the
[training recipe](../examples/train-qwen-image-edit.yaml). LoRA targets can include
`to_q`, `to_v`, `add_q_proj` and `add_v_proj` in `image_generator`.
`latent_projector`, `text_projector`, `timestep_embedder` and `image_head` are
separately selectable DiT components. The VL encoder and VAE stay frozen;
QLoRA, cross-example packing and multi-image edits are unsupported.

The native condition image is resized to approximately one megapixel independently
of `image_height` and `image_width`, which control the target. Lower target
resolution therefore does not remove the source-conditioning cost. Budgets cover
the expanded VL sequence, both latent grids, vision patches and decoded images.
Output dimensions must be multiples of 16 for the original checkpoint.

The shared prepare/train/export commands produce a complete native Diffusers
pipeline with the merged transformer and frozen frontend assets. Native inference
uses `infer --generate-image`; serving uses `/v1/images/edits` with a multipart
source image, not a text-only generation request. The
[deployment recipe](../examples/deploy-qwen-image-edit.yaml) launches the original
single-stage pipeline. These execution checks do not establish editing quality
or consumer-GPU memory requirements.

## PersonaPlex

The `personaplex` adapter targets NVIDIA PersonaPlex 7B v1. Access requires accepting
the model terms on Hugging Face and authenticating locally. Install the `train` and
`personaplex` extras and use the [training recipe](../examples/train-personaplex.yaml).
The adapter supports native training, merged export and streaming inference, with
vLLM-Omni WebSocket duplex deployment. Conversation quality requires evaluation on
representative held-out speech; execution fixtures are not a quality benchmark.

Supply synchronized user and agent audio tracks covering the complete session,
including silence and overlapping speech, plus the agent transcript and explicit
token timing ([data format](datasets.md#duplex-recordings-and-token-timing)). Frozen
Mimi encodes each track independently into eight codebooks at 12.5 Hz. The native
17-stream temporal/depth schedule owns attention and delays; assistant text and
agent audio are supervised separately. User audio is input conditioning by default.
`personaplex_user_loss_weight` enables an optional auxiliary user-code objective.

For longer recordings, start from the [explicit session-budget recipe](../examples/train-personaplex-real.yaml).
`max_audio_seconds` bounds physical audio; `max_codec_tokens` bounds 80 ms frames.
`max_length` and `max_attention_pairs` must also accommodate the voice/persona
prefix. The example permits up to 180 seconds but is not a minimum-VRAM guarantee.

The experimental `personaplex_agent_only_depth: true` option skips the eight
user-prediction depth positions when `personaplex_user_loss_weight: 0`. User input
audio still conditions the temporal model. It supports temporal/text-head tuning
and agent projection/head LoRA targets 0–7; other depth parameters must remain
frozen. Native weights and streaming inference retain all 16 audio codebooks.
This option defaults to false; benchmark your recipe before enabling it.

The selected `personaplex_voice` is a native cached voice prefix, followed by the
system persona. Keep input embeddings and Mimi frozen. Supported LoRA targets are
`temporal_model.out_proj`, numbered `audio_projector` layers and numbered
`speech_generator` heads; indices 0–7 correspond to agent codebooks. Full component
tuning is also available. Fused attention and depth layers that read weights directly
reject ordinary Linear LoRA injection. Quantization and cross-example packing are
not supported; non-reentrant gradient checkpointing covers temporal and depth layers.

Shared prepare/train/export commands produce a complete native checkpoint with Mimi,
tokenizer and voices. `infer --generate-audio` replays the input session through the
native streaming generator. Completion is `input_session_end`, with one output frame
per user frame; this protocol has no utterance EOS. The generation bound must cover
the whole session. The [deployment recipe](../examples/deploy-personaplex.yaml) targets
vLLM-Omni's WebSocket duplex API, rather than the offline speech endpoint.

For realtime validation, the original recording is followed by one explicit 80 ms
silent drain tick: vLLM-Omni needs the next step to complete delayed residual
codebooks. Reports separate original input frames, drain frames and output frames,
and require output during the original input. This does not add session resume,
barge-in control, multi-session load guarantees or a latency benchmark.
