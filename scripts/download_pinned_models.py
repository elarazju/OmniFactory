"""Download pinned official snapshots through an explicitly selected Hub mirror.

Run on the GPU host; inspect metadata and a small config before requesting weights.
Verifies every LFS object's advertised SHA-256, then records the local fingerprint.
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

MODELS = {
    "qwen2.5-3b": ("Qwen/Qwen2.5-Omni-3B", "f75b40e3da2003cdd6e1829b1f420ca70797c34e"),
    "personaplex": ("nvidia/personaplex-7b-v1", "fdaf4090a61cb315c138a1faee287ffd6c716309"),
    "qwen-image-edit": ("Qwen/Qwen-Image-Edit", "ac7f9318f633fc4b5778c59367c8128225f1e3de"),
    "cosyvoice3": (
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
    ),
    "qwen3-tts": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3"),
    "bagel": ("ByteDance-Seed/BAGEL-7B-MoT", "5019f57d168e5816e8f3f701b17cc816bb7cf24b"),
    "voxcpm2": ("openbmb/VoxCPM2", "32279effe8c19989596f05d353d1447f51d9e915"),
    "qwen3": ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"),
    "minicpm": ("openbmb/MiniCPM-o-4_5", "eb4141de24550a3a1a4082f718be58f46b761728"),
    "mimo": ("XiaomiMiMo/MiMo-Audio-7B-Instruct", "c359441c22c2a1c74be5f99a91e83392680e9cc8"),
    "mimo-codec": ("XiaomiMiMo/MiMo-Audio-Tokenizer", "5df9914f72d3acda1320d7fecde7d91622edb0c1"),
}

# Preserve the native training/inference assets without downloading alternative
# RL weights, duplicate batch tokenizers or optional TensorRT export inputs.
MODEL_FILES = {
    "cosyvoice3": (
        "CosyVoice-BlankEN/*",
        "config.json",
        "configuration.json",
        "cosyvoice3.yaml",
        "llm.pt",
        "flow.pt",
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v3.onnx",
        "README.md",
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=MODELS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    repo, revision = MODELS[args.family]
    api = HfApi(endpoint=args.endpoint)
    info = api.model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    config_file = "model_index.json" if args.family == "qwen-image-edit" else "config.json"
    small = hf_hub_download(
        repo, config_file, revision=revision, endpoint=args.endpoint, local_dir=args.output
    )
    json.loads(Path(small).read_text())
    from fnmatch import fnmatch

    patterns = MODEL_FILES.get(args.family)
    selected = [
        s
        for s in info.siblings
        if patterns is None or any(fnmatch(s.rfilename, pattern) for pattern in patterns)
    ]
    total = sum(s.size or 0 for s in selected)
    print(
        json.dumps(
            {
                "probe_passed": True,
                "repository": repo,
                "revision": revision,
                "bytes": total,
                "files": len(selected),
            }
        ),
        flush=True,
    )
    if args.probe_only:
        return
    snapshot_download(
        repo,
        revision=revision,
        endpoint=args.endpoint,
        local_dir=args.output,
        max_workers=4,
        allow_patterns=list(patterns) if patterns else None,
    )
    checked = []
    for item in selected:
        file = args.output / item.rfilename
        if not file.is_file() or (item.size is not None and file.stat().st_size != item.size):
            raise RuntimeError(f"Missing or truncated snapshot file: {item.rfilename}")
        if item.lfs:
            hasher = hashlib.sha256()
            with file.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    hasher.update(chunk)
            assert hasher.hexdigest() == item.lfs.sha256, item.rfilename
            checked.append(item.rfilename)
    # Snapshot cache metadata is excluded from the source fingerprint by existing
    # source_fingerprint's hidden-file rule only at leaf level. Remove this new
    # download's bookkeeping after success, never model payloads.
    import shutil

    shutil.rmtree(args.output / ".cache/huggingface", ignore_errors=True)
    from omnifactory.training.prepare import source_fingerprint

    report = dict(
        repository=repo,
        revision=revision,
        endpoint=args.endpoint,
        lfs_files_verified=checked,
        bytes=total,
        source_fingerprint=source_fingerprint(str(args.output), None),
    )
    destination = args.output.parent / (args.output.name + "-verification.json")
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
