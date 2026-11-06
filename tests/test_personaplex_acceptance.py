import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("soundfile")
import numpy as np
import soundfile as sf


@pytest.fixture
def acceptance():
    path = Path(__file__).parents[1] / "scripts/validate_gpu_personaplex.py"
    spec = importlib.util.spec_from_file_location("personaplex_acceptance", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.fixture
def prediction(tmp_path):
    path = tmp_path / "generated.wav"
    sf.write(path, np.ones(3840, dtype=np.float32) * 0.1, 24000)
    return dict(
        sample_id="s",
        prediction_id="p",
        stop_reason="input_session_end",
        sample_rate=24000,
        input_frames=2,
        codec_tokens=[[1] * 8] * 2,
        audio=str(path),
    )


def test_session_coverage_requires_exact_frames_and_identity(acceptance, prediction):
    expected = {("s", "p"): 2}
    report = acceptance.verify_session_coverage([prediction], expected)
    assert report[0]["waveform_samples"] == 3840
    for rows in ([], [prediction, prediction], [{**prediction, "sample_id": "wrong"}]):
        with pytest.raises(ValueError, match="identities/count"):
            acceptance.verify_session_coverage(rows, expected)
    for change in (
        dict(codec_tokens=[[1] * 8]),
        dict(input_frames=1),
        dict(stop_reason="max_length"),
        dict(codec_tokens=[[1] * 7] * 2),
    ):
        with pytest.raises(ValueError, match="every original input frame"):
            acceptance.verify_session_coverage([{**prediction, **change}], expected)


@pytest.mark.parametrize(
    "samples,rate,channels", [(3839, 24000, 1), (3840, 16000, 1), (3840, 24000, 2)]
)
def test_waveform_coverage_cannot_be_faked_by_codec_metadata(
    acceptance, prediction, samples, rate, channels
):
    shape = (samples,) if channels == 1 else (samples, channels)
    sf.write(prediction["audio"], np.ones(shape, dtype=np.float32) * 0.1, rate)
    with pytest.raises(ValueError, match="waveform"):
        acceptance.verify_session_coverage([prediction], {("s", "p"): 2})
