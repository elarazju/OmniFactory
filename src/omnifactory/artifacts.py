"""Native export integrity: complete shard indexes, required assets and content identity."""

import hashlib
import json
import math
import struct
from pathlib import Path

from omnifactory.data.io import atomic_writer
from omnifactory.errors import OmniFactoryError


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise OmniFactoryError("E_EXPORT_TENSOR", str(path))
        size = struct.unpack("<Q", prefix)[0]
        if size > 100_000_000 or size > path.stat().st_size - 8:
            raise OmniFactoryError("E_EXPORT_TENSOR", "invalid tensor header size")
        header = json.loads(stream.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    bits = {
        "BOOL": 8,
        "U8": 8,
        "I8": 8,
        "F8_E4M3": 8,
        "F8_E5M2": 8,
        "F8_E8M0": 8,
        "I16": 16,
        "U16": 16,
        "F16": 16,
        "BF16": 16,
        "I32": 32,
        "U32": 32,
        "F32": 32,
        "I64": 64,
        "U64": 64,
        "F64": 64,
    }
    for tensor in tensors.values():
        shape, dtype = tensor.get("shape"), tensor.get("dtype")
        if (
            not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or dtype not in bits
        ):
            raise OmniFactoryError("E_EXPORT_TENSOR", "invalid tensor shape/dtype")
        start, stop = tensor["data_offsets"]
        if stop - start != math.prod(shape) * bits[dtype] // 8:
            raise OmniFactoryError(
                "E_EXPORT_TENSOR", "tensor shape/dtype disagrees with data bytes"
            )
    offsets = sorted(v["data_offsets"] for v in tensors.values())
    end = 0
    for start, stop in offsets:
        if start != end or stop < start:
            raise OmniFactoryError("E_EXPORT_TENSOR", "invalid/overlapping tensor offsets")
        end = stop
    if end + size + 8 != path.stat().st_size:
        raise OmniFactoryError("E_EXPORT_TENSOR", "truncated/extra tensor data")
    return tensors


def validate_native(root, metadata):
    """Built-in stage assets; custom adapters can supply a different validation hook."""
    adapter = metadata["adapter"]
    required = ["config.json"]
    if adapter == "bagel_dit":
        required += [
            "llm_config.json",
            "ema.safetensors",
            "ae.safetensors",
            "model.safetensors.index.json",
        ]
    elif adapter == "voxcpm2":
        required += ["model.safetensors"]
        if not any((root / n).is_file() for n in ("audiovae.safetensors", "audiovae.pth")):
            raise OmniFactoryError("E_EXPORT_ASSET", "VoxCPM2 VAE is missing")
    elif adapter == "minicpm_token2wav":
        from omnifactory.adapters.token2wav import verify_assets

        verify_assets(root)
        return
    for name in required:
        if not (root / name).is_file():
            raise OmniFactoryError("E_EXPORT_ASSET", f"missing {name}")
    cfg = metadata.get("inference_config", {})
    base = Path(cfg.get("model", ""))
    frozen = {
        "bagel_dit": ("ae.safetensors",),
        "voxcpm2": ("audiovae.safetensors", "audiovae.pth"),
    }.get(adapter, ())
    for name in frozen:
        if (
            (root / name).is_file()
            and (base / name).is_file()
            and sha256(root / name) != sha256(base / name)
        ):
            raise OmniFactoryError("E_EXPORT_FROZEN", f"frozen asset differs: {name}")


def inspect_export(root):
    from omnifactory.adapters.runtime import adapter_spec, resolve

    root = Path(root).resolve()
    metadata = json.loads((root / "omnifactory_export.json").read_text())
    resolve(adapter_spec(metadata["adapter"]).artifact_validator)(root, metadata)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise OmniFactoryError("E_EXPORT_PATH", "exports must contain files, not symlinks")
    indexes = list(root.rglob("*.safetensors.index.json"))
    for index in indexes:
        value = json.loads(index.read_text())
        mapping = value.get("weight_map", {})
        if not mapping:
            raise OmniFactoryError("E_EXPORT_INDEX", "empty weight map")
        actual, total = {}, 0
        for filename in set(mapping.values()):
            if Path(filename).name != filename:
                raise OmniFactoryError("E_EXPORT_INDEX", "shard path must be a local filename")
            path = index.parent / filename
            if not path.is_file():
                raise OmniFactoryError("E_EXPORT_INDEX", f"missing shard {filename}")
            for key, tensor in tensor_header(path).items():
                if key in actual:
                    raise OmniFactoryError("E_EXPORT_INDEX", f"duplicate key {key}")
                actual[key] = filename
                start, end = tensor["data_offsets"]
                total += end - start
        if actual != mapping or value.get("metadata", {}).get("total_size") != total:
            raise OmniFactoryError(
                "E_EXPORT_INDEX", "index keys, shard mapping or tensor bytes differ"
            )
        if metadata["adapter"] == "bagel_dit" and index.parent == root:
            if set(mapping.values()) != {"ema.safetensors", "ae.safetensors"}:
                raise OmniFactoryError("E_EXPORT_INDEX", "BAGEL index must include core and VAE")
    for path in root.rglob("*.safetensors"):
        tensor_header(path)
    return metadata


def seal_export(root):
    root = Path(root)
    metadata = inspect_export(root)
    files = {
        str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "artifact-manifest.json"
    }
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = dict(
        format="omnifactory-export-v1",
        artifact_id=identity,
        adapter=metadata["adapter"],
        source_fingerprint=metadata["source_fingerprint"],
        files=files,
        structural_validation=True,
        deployment_verified=False,
    )
    with atomic_writer(root / "artifact-manifest.json") as stream:
        stream.write(json.dumps(manifest, indent=2))
    return manifest


def verify_export(root):
    root = Path(root)
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "artifact-manifest.json"
    }
    if actual != set(manifest["files"]):
        raise OmniFactoryError("E_EXPORT_HASH", "export file inventory changed")
    for name, entry in manifest["files"].items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root.resolve())
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise OmniFactoryError("E_EXPORT_HASH", f"export file changed: {name}")
    identity = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    if identity != manifest["artifact_id"]:
        raise OmniFactoryError("E_EXPORT_HASH", "artifact identity changed")
    inspect_export(root)
    return manifest
