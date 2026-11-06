"""Publish complete artifact directories without overwriting existing runs."""

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from omnifactory.errors import OmniFactoryError


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_processor(processor, path: Path):
    # Transformers 4.57 stores both components in preprocessor_config.json;
    # its generic save overwrites the image settings with the audio settings.
    processor.save_pretrained(path)
    settings = {**processor.image_processor.to_dict(), **processor.feature_extractor.to_dict()}
    settings["processor_class"] = type(processor).__name__
    write_json(path / "preprocessor_config.json", settings)


@contextmanager
def new_directory(path: Path):
    path = path.resolve()
    if path.exists():
        raise OmniFactoryError("E_OUTPUT_EXISTS", f"choose a new output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        yield temporary
        if path.exists():
            raise OmniFactoryError("E_OUTPUT_EXISTS", str(path))
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
