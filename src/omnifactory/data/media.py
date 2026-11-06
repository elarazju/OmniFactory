"""Bounded-memory local media provenance. Decoding is intentionally a separate adapter."""

import hashlib
import json
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from omnifactory.data.schema import CanonicalSample, MediaAsset, Record
from omnifactory.errors import OmniFactoryError


class ResolvedAsset(Record):
    path: str
    sha256: str
    size_bytes: int


class MediaManifest(Record):
    kind: Literal["media_manifest"] = "media_manifest"
    schema_version: Literal["0.1"] = "0.1"
    verification: Literal["file_bytes_only"] = "file_bytes_only"
    sample: CanonicalSample
    assets: dict[str, ResolvedAsset]
    fingerprint: str


class LocalMediaResolver:
    def __init__(self, dataset_root: Path):
        self.root = dataset_root.resolve()

    def resolve(self, asset: MediaAsset) -> ResolvedAsset:
        uri = urlparse(asset.uri)
        if uri.scheme not in ("", "file") or (
            uri.scheme == "file" and uri.netloc not in ("", "localhost")
        ):
            raise OmniFactoryError(
                "E_MEDIA_URI", "this resolver supports local paths and local file:// URIs only"
            )
        path = Path(unquote(uri.path) if uri.scheme == "file" else asset.uri).expanduser()
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise OmniFactoryError("E_MEDIA_MISSING", str(path))
        from omnifactory.training.fingerprints import file_fingerprint

        before = path.stat()
        checksum = file_fingerprint(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OmniFactoryError("E_MEDIA_CHANGED", str(path))
        count = after.st_size
        if asset.sha256 is not None and checksum != asset.sha256:
            raise OmniFactoryError("E_MEDIA_HASH", f"checksum mismatch for {path}")
        return ResolvedAsset(path=str(path), sha256=checksum, size_bytes=count)

    def manifest(self, sample: CanonicalSample) -> MediaManifest:
        assets = {key: self.resolve(asset) for key, asset in sample.assets.items()}
        raw = sample.model_dump(mode="json")
        for key, asset in assets.items():
            raw["assets"][key]["sha256"] = asset.sha256
        # Revalidate: equal file contents may reveal previously hidden target/reference aliases.
        snapshot = CanonicalSample.model_validate(raw)
        encoded = json.dumps(
            snapshot.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return MediaManifest(
            sample=snapshot, assets=assets, fingerprint=hashlib.sha256(encoded.encode()).hexdigest()
        )
