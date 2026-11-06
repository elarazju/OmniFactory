# Datasets

OmniFactory provides generated fixtures and importers for your own local data. It does not bundle third-party training corpora or download them implicitly.

| Dataset or importer | Inputs → targets | Use |
| --- | --- | --- |
| `datasets demo` | Text/image/audio → text | Small runnable training workflow |
| `datasets omni-demo` | Text/image/audio/video → text/audio | Audiovisual fixtures; requires FFmpeg and matching model assets/labels |
| `datasets image-demo` | Text → image | Procedural image fixtures for BAGEL |
| `datasets duplex-demo` | Synchronized user/agent audio + timed text | Synthetic tones and explicit timing for PersonaPlex execution checks |
| `datasets image-edit-demo` | Source image + instruction → edited image | Procedural editing triples for the Qwen-Image-Edit adapter |
| `datasets flow-demo` | Native audio conditions → continuous audio targets | Token2Wav stage fixtures |
| `datasets import-alpaca` | Text → text | Convert existing instruction JSONL |
| `datasets import-tts` | Literal text + optional reference audio → speech | Import recordings for VoxCPM2 |
| Canonical JSONL | Model-dependent | Your multimodal data and explicit prediction targets |

Generated tones, shapes, and random-model answers test execution; they are not speech or image-quality training datasets. `examples/samples.jsonl` demonstrates the schema but its media paths are placeholders.

For image editing, put the source `image` block and editing `text` block in the user
message, and a separate target `image` block in the assistant message. Select only
the assistant image in `target_block_ids`. The original Qwen-Image-Edit adapter
accepts one source image per instruction. Keep related source images in the same
`group_id` across splits. Generate runnable example records with:

```bash
uv run --extra train --extra image omnifactory datasets image-edit-demo \
  --output outputs/image-edit-data --size 256
```

## Generate a fixture

```bash
uv run --extra train omnifactory datasets demo --output outputs/demo
uv run --extra train omnifactory data validate outputs/demo/train.jsonl
uv run --extra train omnifactory data manifest outputs/demo/train.jsonl \
  --root outputs/demo --output outputs/demo-manifest.jsonl
```

## Import text-only Alpaca data

Input is JSONL, one object per line:

```json
{"instruction":"Answer the question.","input":"What is 1 + 1?","output":"2"}
```

```bash
uv run omnifactory datasets import-alpaca \
  --source /path/to/alpaca.jsonl --output outputs/train.jsonl \
  --name my-instructions --split train
```

Use the same dataset name for its validation split. Group IDs derive from the prompt so repeated prompts can be identified across splits. Extra fields, including images and audio, are rejected instead of silently discarded. Source dataset licenses remain unchanged.

## Canonical multimodal data

Each JSONL record contains `schema_version`, `id`, `provenance`, `assets`, `messages`, and `predictions`. `speech_pairs` is optional. A simple image-understanding record:

```json
{
  "schema_version": "0.1",
  "id": "image-001",
  "provenance": {"dataset": "my-data", "source_id": "001", "group_id": "scene-001", "split": "train"},
  "assets": {"image": {"uri": "images/001.png", "modality": "image"}},
  "messages": [
    {"id": "m0", "role": "user", "content": [
      {"id": "image-block", "type": "image", "asset_id": "image"},
      {"id": "question", "type": "text", "text": "Describe this image."}
    ]},
    {"id": "m1", "role": "assistant", "content": [
      {"id": "answer", "type": "text", "text": "A blue square."}
    ]}
  ],
  "predictions": [{"id": "p0", "target_message_id": "m1", "target_block_ids": ["answer"]}]
}
```

The formatted object above must occupy one line in JSONL. Relative asset paths resolve from `dataset_root` in the training recipe, defaulting to the JSONL's directory. Local media must exist; the execution path does not fetch remote URLs. `group_id` should identify the same recording/session/source across train and evaluation splits, not just a row number.

For audio, use an `audio` block with `media.asset_id` and optional half-open `start_us` / `end_us`. Declared hashes and sampling rates are verified before encoding. Execution support is adapter-specific: Qwen/MiniCPM include offline video and speech targets, VoxCPM2 uses continuous speech targets, and BAGEL uses continuous image targets. Explicit timed attention is implemented for MiniCPM offline training. PersonaPlex uses synchronized duplex recordings and its native streaming attention; see the recording and timing requirements below.

Use `omnifactory data schema` to inspect the complete schema and [Getting Started](getting_started.md) for training commands.

## Import text and speech pairs

Use one JSON object per line:

```json
{"id":"utterance-001","group_id":"speaker-01","text":"Hello world.","audio":"audio/001.wav","reference_audio":"audio/another.wav"}
```

`reference_audio` is optional and must differ from the target. Paths resolve relative to the source manifest. Split speakers or source sessions before import, and preserve the original dataset license.

```bash
uv run --extra train omnifactory datasets import-tts \
  --source train-pairs.jsonl --output outputs/tts-data/train.jsonl --name my-tts --split train
uv run --extra train omnifactory datasets import-tts \
  --source validation-pairs.jsonl --output outputs/tts-data/validation.jsonl --name my-tts --split validation
```

This imports recordings; it does not make them compatible with every speech model. [Model Recipes](models.md) explains native codec/label requirements. In particular, Qwen speech currently needs native Qwen labels, not arbitrary WAV targets.

## Audio, video, and speech targets

A video block declares whether the soundtrack participates:

```json
{"id":"clip","type":"video","media":{"asset_id":"movie","start_us":250000,"end_us":2250000},"audio_mode":"include"}
```

Use `exclude` for silent frames. Frame sampling and audio slicing follow the selected model recipe, bounded by duration, frame, pixel, sequence, and decoded-byte limits. Offline video support does not imply live duplex interaction.

For speech tasks that select a transcript and waveform together, associate their block IDs:

```json
{"predictions":[{"id":"p0","target_message_id":"answer","target_block_ids":["text","audio"]}],"speech_pairs":[{"text_block_id":"text","audio_block_id":"audio","alignment":"utterance"}]}
```

Discrete speech codes and continuous speech/image targets use the same canonical records. The adapter builds its native supervision during preparation; users do not need to force continuous data into an `audio_tokens` field. Model-specific constraints still apply.

## Qwen teacher labels

`datasets qwen-speech` explicitly synthesizes matched native codes and WAVs from supplied transcripts. It writes `speech_codes.jsonl`, preserves train/validation groups, and binds labels to sample, transcript, audio, and source identity. It **replaces target recordings**; it is not a waveform encoder or voice-cloning tool.

Default preparation rejects outputs that reach the generation limit. Increase the limit for complete labels; `--allow-truncated` is intended for explicit execution fixtures and does not add fabricated EOS labels. See the [Qwen speech setup](models.md#qwen-speech-labels).

## Use real data

Keep validation speakers, recordings, or source groups disjoint from training. Supply actual media files and correctly paired targets, and retain licenses and provenance. Local paths must be accessible at training time. Optional LibriSpeech preparation scripts in `scripts/` provide a small real-speech example; review their download and split parameters before running them. They do not constitute an official benchmark.

For the paired ASR/TTS baseline recipes, download the fixed source and prepare it
from the repository root:

```bash
uv run --extra train hf download openslr/librispeech_asr clean/validation/0000.parquet \
  --repo-type dataset --revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1 \
  --local-dir outputs/librispeech-source
uv run --extra train --with pyarrow python scripts/prepare_librispeech.py \
  --parquet outputs/librispeech-source/clean/validation/0000.parquet \
  --output outputs/real-asr --train-count 256 --validation-count 64 --max-seconds 12
uv run --extra train python scripts/prepare_asr_recipe.py \
  --source outputs/real-asr --output outputs/real-asr-transcript
uv run --extra train python scripts/prepare_librispeech_tts.py \
  --source outputs/real-asr --output outputs/real-tts \
  --train-count 256 --validation-count 64 --text-case sentence
```

The importer verifies the source parquet's pinned SHA-256 and preserves the
LibriSpeech CC BY 4.0 attribution. This repartitions the development subset by
speaker; results are not official test-clean scores. The TTS conversion preserves
target recordings and partitions. Reference-conditioned cases use another
utterance by the same speaker within the same split; speakers with only one
available recording use text-only conditioning. Keep the source ASR media
directory because the TTS records reference its files.

The ASR variant explicitly asks for a verbatim transcript without a preamble.
Use the same instruction for training and all evaluation branches, including the
base model, so format compliance is not mistaken for better recognition. Targets,
speaker groups and recording hashes are preserved. This system prompt is specific
to transcription; do not reuse it for Qwen's speech generation tasks.

LibriSpeech transcripts are uppercase. The explicit TTS `--text-case sentence`
option changes only casing, keeps original text in `text-normalization.json`, and
applies the same transformation to training and validation. It loses proper-name
and acronym capitalization: review it for your task, or choose `original` to retain
the source text. For your own recordings, prefer reviewed natural text and keep
pronunciation-sensitive spelling. Neither the adapter nor scorer silently rewrites
user text.

### Prepare an independent speech confirmation set

Freeze your recipe using development data before evaluating a separate test set.
The optional test importer uses the pinned LibriSpeech test-clean parquet, checks
speaker/sample/waveform separation against every supplied exclusion manifest, and
selects recordings by a fixed hash order. Its defaults produce 96 targets across
24 speakers, with equal reference-conditioned and text-only cases. Reference
recordings are separate from scored targets.

```bash
uv run --extra train hf download openslr/librispeech_asr clean/test/0000.parquet \
  --repo-type dataset --revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1 \
  --local-dir outputs/librispeech-source
uv run --extra train --with pyarrow python scripts/prepare_librispeech_test.py \
  --parquet outputs/librispeech-source/clean/test/0000.parquet \
  --exclude outputs/real-tts/train.jsonl \
  --exclude outputs/real-tts/validation.jsonl --output outputs/tts-confirmation
```

The output preserves `test` provenance, original transcripts, licenses, media
hashes and the selection seed. It uses the same explicit sentence-case convention
as the TTS recipe. Include all manifests used for training or recipe selection in
`--exclude`. This is a TTS confirmation subset, not the official full ASR test.
Use it with [frozen-checkpoint evaluation](training.md#evaluate-a-frozen-speech-recipe).

## Duplex recordings and token timing

Use `datasets import-duplex` to import synchronized recordings and their annotations:

```bash
uv run --extra train --extra personaplex omnifactory datasets import-duplex \
  --source recordings.jsonl --output data/duplex \
  --tokenizer /path/to/personaplex/tokenizer_spm_32k_3.model
```

Each source row declares its provenance, participating speakers, common crop and
agent transcript. Paths resolve relative to the source JSONL. For example, with
two mono recordings and reviewed word intervals:

```json
{"id":"session-1","dataset":"my-recordings","group_id":"session-1","split":"train","license":"your-source-license","source_url":"your-source-reference","speaker_ids":["speaker-a","speaker-b"],"user_audio":"user.wav","agent_audio":"agent.wav","start_us":0,"end_us":10000000,"transcript":"Hello","alignment":"word_uniform","words":[{"text":"Hello","start_us":1000000,"end_us":1400000}]}
```

Alternatively, supply `stereo_audio`, `user_channel` and `agent_channel` instead
of the two mono paths. Channels must differ. Sources must share the same sample
rate and length; crop boundaries must lie on source samples. The importer retains
silence and overlap and writes the common crop as float32 mono WAV files. It does
not align independently recorded clocks or repair drift.

`alignment: "tokens"` is the default and requires `token_ids` plus one absolute
`token_times_us` per token, matching the complete transcript's SentencePiece
encoding. `word_uniform` explicitly estimates subword timing inside the supplied
word intervals. Its transcript must equal the word texts joined with spaces;
attach punctuation to its word. The report records this estimation. An optional
`--word-slack-us 80000` allows up to one extra frame around word intervals;
otherwise the importer permits no extra slack. Grid quantization itself has 80 ms
resolution. Infeasible token density, overlapping word intervals, incorrect tokens
or crop boundaries fail without publishing a partial dataset.

The output contains split JSONL files, media, `text-alignment.json`, source records
and `import-report.json`. Point `duplex_text_alignment` at the generated sidecar.
The importer checks source recording, group and speaker leakage across supplied
splits. It validates structural timing, not whether the words were actually spoken;
keep human or forced-aligner provenance and review uncertain intervals.

PersonaPlex requires one complete mono user track and one complete mono agent track,
of equal duration after resampling to 24 kHz. Both audio blocks declare
`placement: {timeline_id: session, offset_us: 0}`; slices and staggered file offsets
are rejected. Preserve overlap and silence within each recording. An optional system
message supplies the literal persona. Pair the assistant transcript with its audio
using `speech_pairs`, selecting both blocks in the prediction.

Set `duplex_text_alignment` to a JSON array. Each entry contains `sample_id`,
`prediction_id`, `tokenizer_sha256`, `audio_sha256`, `text_sha256`, `token_ids`, and
`frames`. Tokens must exactly match the bundled SentencePiece encoding of the
transcript; frames are strictly increasing zero-based 80 ms frame indices. Hashes
bind annotations to the tokenizer, full agent recording and UTF-8 transcript.
Obtain timing from aligned recordings or reviewed external annotations; the adapter
does not infer timing or silently spread words across the recording.

A synthetic execution fixture is available:

```bash
uv run --extra train --extra personaplex omnifactory datasets duplex-demo \
  --output data/duplex --tokenizer /path/to/personaplex/tokenizer_spm_32k_3.model
```

It creates paired tones, synthetic transcript timing and separate training/validation
records. It is useful for checking the pipeline, not for learning or evaluating
conversational speech. Replace it with licensed, correctly aligned recordings for
real training, and split by source session or speaker.
