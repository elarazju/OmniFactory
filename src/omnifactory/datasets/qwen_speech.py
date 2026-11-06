"""Explicit native Qwen teacher synthesis of paired waveform/code training labels."""

import json
from pathlib import Path

import soundfile as sf
import torch

from omnifactory.adapters.qwen_speech import speech_target, text_hash
from omnifactory.adapters.runtime import adapter_class, load_model
from omnifactory.adapters.speech import file_sha256
from omnifactory.data.io import iter_samples
from omnifactory.data.schema import CanonicalSample
from omnifactory.errors import OmniFactoryError
from omnifactory.training.artifacts import digest, new_directory, write_json
from omnifactory.training.prepare import source_fingerprint


@torch.no_grad()
def synthesize_dataset(
    config, output: Path, *, datasets=None, max_codec_tokens=750, allow_truncated=False
):
    """Teacher-force provided transcripts, synthesize matching native codes and audio.

    Existing target audio is replaced explicitly, not encoded. Original input
    media remains referenced by absolute paths; generated audio is local.
    """
    if config.adapter not in ("qwen2_5_omni", "qwen3_omni") or config.task != "speech_sft":
        raise OmniFactoryError(
            "E_SPEECH_MODEL", "native teacher synthesis requires Qwen speech_sft config"
        )
    if config.adapter == "qwen3_omni":
        from omnifactory.adapters import qwen3_speech as native

        if allow_truncated:
            raise OmniFactoryError(
                "E_SPEECH_TRUNCATED", "Qwen3 labels require complete native frames and EOS"
            )
    else:
        from omnifactory.adapters import qwen_speech as native
    torch.manual_seed(config.seed)
    if config.device == "cpu":
        torch.set_num_threads(2)
    fingerprint = source_fingerprint(config.model, config.revision)
    model = load_model(config).eval()
    decoder = native.make_decoder(config)
    adapter = adapter_class(config.adapter).from_config(
        config.model_copy(update={"task": "understanding_sft"})
    )
    count, manifests, split_rows, seen = 0, [], {}, set()
    with new_directory(output) as root:
        (root / "media").mkdir()
        for dataset in datasets or [config.dataset]:
            adapter.resolver.root = (config.dataset_root or Path(dataset).parent).resolve()
            for sample in iter_samples(Path(dataset)):
                if len(sample.predictions) != 1:
                    raise OmniFactoryError(
                        "E_SPEECH_TARGET",
                        "teacher preparation currently requires one prediction per sample",
                    )
                pred = sample.predictions[0]
                if sample.id in seen:
                    raise OmniFactoryError("E_DUPLICATE_ID", sample.id)
                seen.add(sample.id)
                answer, block = speech_target(sample, pred.id)
                raw = sample.model_dump(mode="json")
                text_id = next(
                    p.text_block_id for p in sample.speech_pairs if p.audio_block_id == block.id
                )
                understanding = sample.model_dump(mode="json")
                understanding["predictions"][0]["target_block_ids"] = [text_id]
                encoded = adapter.encode(
                    CanonicalSample.model_validate(understanding),
                    pred.id,
                    contract_id="native-teacher",
                )
                batch = adapter.to_device(encoded.payload, model, config)
                ids = batch["input_ids"]
                end = encoded.prompt_length + int(
                    (
                        ids[0, encoded.prompt_length :] == adapter.processor.tokenizer.eos_token_id
                    ).nonzero()[0]
                )
                forward = model.thinker(**batch, output_hidden_states=True, use_cache=False)
                codes, end_token = native.generate_talker(
                    model,
                    native.talker_condition(model, batch, forward, encoded.prompt_length, end),
                    max_codec_tokens,
                    do_sample=True,
                )
                finished = end_token is not None
                if not finished and not allow_truncated:
                    raise OmniFactoryError(
                        "E_SPEECH_TRUNCATED",
                        "teacher reached codec limit; increase it or explicitly allow truncated execution fixtures",
                    )
                waveform, rate = decoder(codes)
                name = f"media/{count:08d}.wav"
                sf.write(root / name, waveform, rate)
                target_asset = block.media.asset_id
                for key, asset in sample.assets.items():
                    if key != target_asset:
                        raw["assets"][key]["uri"] = adapter.resolver.resolve(asset).path
                raw["assets"][target_asset] = {
                    "uri": name,
                    "modality": "audio",
                    "sha256": file_sha256(root / name),
                    "sample_rate_hz": rate,
                }
                target_message = next(
                    m for m in raw["messages"] if m["id"] == pred.target_message_id
                )
                target = next(b for b in target_message["content"] if b["id"] == block.id)
                target["media"] = {"asset_id": target_asset, "start_us": 0, "end_us": None}
                result = CanonicalSample.model_validate(raw)
                split_rows.setdefault(result.provenance.split, []).append(result.model_dump_json())
                manifests.append(
                    {
                        "codec": native.CODEC,
                        "sample_id": sample.id,
                        "prediction_id": pred.id,
                        "audio_block_id": block.id,
                        "audio_sha256": file_sha256(root / name),
                        "transcript_sha256": text_hash(answer),
                        "sample_sha256": digest(result.model_dump(mode="json")),
                        "start_us": 0,
                        "end_us": None,
                        "speaker": config.speech_speaker,
                        "origin": "native_teacher",
                        "source_fingerprint": fingerprint,
                        "source_model": config.model,
                        "revision": config.revision,
                        "codes": codes,
                        "speech_finished": finished,
                        "eos_token_id": end_token,
                    }
                )
                count += 1
        if not count:
            raise OmniFactoryError("E_DATASET", "empty teacher dataset")
        for split, rows in split_rows.items():
            (root / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
        (root / "speech_codes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifests)
        )
        result = {
            "examples": count,
            "codec": native.CODEC,
            "origin": "native_teacher",
            "source_fingerprint": fingerprint,
            "allow_truncated": allow_truncated,
            "waveform_encoder": False,
            "quality_verified": False,
            "sampling": {
                "temperature": 0.9,
                "top_k": 50 if config.adapter == "qwen3_omni" else 40,
                "top_p": 1.0 if config.adapter == "qwen3_omni" else 0.8,
                "repetition_penalty": 1.05,
            },
        }
        write_json(root / "dataset_info.json", result)
    return {**result, "output": str(output.resolve())}
