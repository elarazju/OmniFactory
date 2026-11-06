"""One entry point for the currently implemented pretrained acceptance protocols.

Each protocol runs preparation, backward/optimizer, exact resume, independent
native reload and generation. HTTP serving and real-data quality remain separate
reports. Unknown task/family combinations fail before starting expensive work.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def command(adapter, task, model, output, *, speech_tokenizer=None, speech_assets=None, data=None):
    scripts = Path(__file__).resolve().parent
    args = ["--model", str(model), "--output", str(output)]
    if adapter == "bagel_dit" and task == "image_sft":
        script = "validate_gpu_bagel.py"
    elif adapter == "voxcpm2" and task == "speech_sft":
        if data is None:
            raise ValueError("VoxCPM2 acceptance requires --data with real canonical TTS pairs")
        script = "validate_gpu_voxcpm2.py"
        args += ["--data", str(data)]
    elif adapter in ("qwen2_5_omni", "qwen3_omni") and task == "understanding_sft":
        script = "validate_gpu_qwen.py"
        args += ["--adapter", adapter]
    elif adapter == "qwen3_omni" and task == "speech_sft":
        script = "validate_gpu_qwen3_speech.py"
    elif adapter == "qwen2_5_omni" and task == "speech_sft":
        script = "validate_gpu_qwen_speech.py"
    elif adapter == "minicpm_token2wav" and task == "flow_sft":
        script = "validate_gpu_flow.py"
        args = ["--assets", str(model), "--output", str(output)]
    elif adapter in ("minicpm_o_4_5", "mimo_audio") and task == "speech_sft":
        if not speech_tokenizer or (adapter == "minicpm_o_4_5" and not speech_assets):
            raise ValueError("native speech validation requires tokenizer/assets")
        script = "validate_gpu_family.py"
        args += ["--adapter", adapter, "--speech-tokenizer", str(speech_tokenizer)]
        if speech_assets:
            args += ["--speech-assets", str(speech_assets)]
    else:
        raise ValueError(f"No unified acceptance protocol for {adapter}/{task}")
    return [sys.executable, str(scripts / script), *args]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True)
    p.add_argument("--task", default="understanding_sft")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--speech-tokenizer", type=Path)
    p.add_argument("--speech-assets", type=Path)
    p.add_argument("--data", type=Path)
    args = p.parse_args()
    try:
        cmd = command(**vars(args))
    except ValueError as exc:
        p.error(str(exc))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
