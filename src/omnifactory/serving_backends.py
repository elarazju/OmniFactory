"""Serving commands kept separate from the native model export format."""

from pathlib import Path

import yaml

from omnifactory.errors import OmniFactoryError

BACKENDS = ("vllm-omni", "sglang-omni")
SGLANG_PIPELINES = {
    "qwen3_tts": ("Qwen3TTSPipelineConfig",),
    "qwen3_omni": ("Qwen3OmniSpeechColocatedPipelineConfig",),
    "minicpm_o_4_5": ("MiniCPMOSpeechPipelineConfig",),
    "cosyvoice3": ("FunCosyVoice3PipelineConfig",),
    "personaplex": ("PersonaPlexSessionPipelineConfig",),
}
SGLANG_PROBES = {
    "qwen3_omni": "omnifactory.adapters.sglang_omni:probe_omni",
    "minicpm_o_4_5": "omnifactory.adapters.sglang_omni:probe_omni",
    "personaplex": "omnifactory.adapters.personaplex_sglang_serving:probe_realtime",
}


def server_command(backend, executable, export, deploy_config, *, adapter, port, model):
    """Bind the launched server to the verified export and a loopback listener."""
    if backend not in BACKENDS:
        raise OmniFactoryError("E_SERVE_BACKEND", f"expected one of {BACKENDS}")
    export, deploy_config = Path(export).resolve(), Path(deploy_config).resolve()
    common = ["--host", "127.0.0.1", "--port", str(port)]
    if backend == "vllm-omni":
        return [
            str(executable),
            "serve",
            str(export),
            "--omni",
            *common,
            "--served-model-name",
            model,
            "--trust-remote-code",
            "--deploy-config",
            str(deploy_config),
            "--disable-log-stats",
        ]
    if adapter not in SGLANG_PIPELINES:
        raise OmniFactoryError(
            "E_SERVE_BACKEND", f"sglang-omni has no integrated native pipeline for {adapter}"
        )
    return [
        str(executable),
        "serve",
        "--model-path",
        str(export),
        "--config",
        str(deploy_config),
        *common,
        "--model-name",
        model,
        *(["--enable-realtime"] if adapter == "personaplex" else []),
    ]


def deployment_config(backend, source, export, *, adapter="qwen3_tts"):
    """Materialize SGLang's required model_path before its file is parsed."""
    raw = Path(source).read_bytes()
    if backend == "vllm-omni":
        return raw
    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise OmniFactoryError("E_SERVE_CONFIG", "invalid deployment YAML") from exc
    expected = SGLANG_PIPELINES.get(adapter, ())
    if not isinstance(config, dict) or config.get("config_cls") not in expected:
        raise OmniFactoryError(
            "E_SERVE_CONFIG", f"expected pipeline class {expected} for {adapter}"
        )

    def check_paths(value):
        if isinstance(value, dict):
            # Stage/factory overrides can otherwise load a different checkpoint,
            # invalidating the artifact identity recorded by deploy validate.
            if "model_path" in value:
                raise OmniFactoryError(
                    "E_SERVE_CONFIG",
                    "omit stage model_path overrides; deploy validate supplies the verified export",
                )
            for item in value.values():
                check_paths(item)
        elif isinstance(value, list):
            for item in value:
                check_paths(item)

    # A top-level path is allowed in reusable upstream configs, but the saved
    # launch config always points at the artifact selected by --model.
    check_paths({k: v for k, v in config.items() if k != "model_path"})
    config["model_path"] = str(Path(export).resolve())
    return yaml.safe_dump(config, sort_keys=False).encode()
