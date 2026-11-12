"""Prepare packaged native voices before sealing a SGLang-Omni export."""

import tarfile
from pathlib import Path

from omnifactory.errors import OmniFactoryError


def unpack_voices(root):
    root = Path(root)
    destination = root / "voices"
    archive_path = root / "voices.tgz"
    if not archive_path.is_file():
        if destination.is_dir() and any(destination.glob("*.pt")):
            return
        raise OmniFactoryError("E_EXPORT_ASSET", "PersonaPlex requires packaged voice presets")
    with tarfile.open(archive_path) as archive:
        members = [m for m in archive if m.isfile() and m.name.endswith(".pt")]
        names = [Path(m.name).name for m in members]
        if (
            not 1 <= len(members) <= 256
            or len(names) != len(set(names))
            or any(m.size > 8 * 1024 * 1024 for m in members)
        ):
            raise OmniFactoryError("E_EXPORT_ASSET", "invalid PersonaPlex voice archive")
        destination.mkdir(exist_ok=True)
        for member, name in zip(members, names, strict=True):
            # Copy regular members by basename; never follow archive links or paths.
            if (destination / name).is_file():
                continue
            with archive.extractfile(member) as source:
                (destination / name).write_bytes(source.read())
