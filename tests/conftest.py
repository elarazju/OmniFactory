import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_sample():
    return json.loads((ROOT / "examples/samples.jsonl").read_text().splitlines()[0])


@pytest.fixture
def reference_sample():
    return json.loads((ROOT / "examples/samples.jsonl").read_text().splitlines()[3])


@pytest.fixture
def examples_path():
    return ROOT / "examples/samples.jsonl"
