import json
from pathlib import Path

import pytest
import yaml

from omnifactory.errors import OmniFactoryError
from omnifactory.serving_backends import deployment_config, server_command

ROOT = Path(__file__).resolve().parents[1]


def command(tmp_path, backend="sglang-omni", adapter="qwen3_tts", config=None):
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config or {"config_cls": "Qwen3TTSPipelineConfig"}))
    args = server_command(
        backend,
        "server",
        tmp_path / "export",
        path,
        adapter=adapter,
        port=9123,
        model="tested-export",
    )
    path.write_bytes(deployment_config(backend, path, tmp_path / "export", adapter=adapter))
    return args


def test_sglang_command_uses_verified_export_and_native_cli(tmp_path):
    config = yaml.safe_load((ROOT / "examples/deploy-qwen3-tts-sglang.yaml").read_text())
    args = command(tmp_path, config=config)
    assert args == [
        "server",
        "serve",
        "--model-path",
        str(tmp_path / "export"),
        "--config",
        str(tmp_path / "deploy.yaml"),
        "--host",
        "127.0.0.1",
        "--port",
        "9123",
        "--model-name",
        "tested-export",
    ]


def test_vllm_command_remains_compatible(tmp_path):
    assert command(tmp_path, backend="vllm-omni", adapter="mimo_audio") == [
        "server",
        "serve",
        str(tmp_path / "export"),
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        "9123",
        "--served-model-name",
        "tested-export",
        "--trust-remote-code",
        "--deploy-config",
        str(tmp_path / "deploy.yaml"),
        "--disable-log-stats",
    ]


@pytest.mark.parametrize(
    "backend,adapter",
    [
        ("unknown", "qwen3_tts"),
        ("sglang-omni", "qwen2_5_omni"),
        ("sglang-omni", "voxcpm2"),
        ("sglang-omni", "mimo_audio"),
        ("sglang-omni", "bagel_dit"),
        ("sglang-omni", "qwen_image_edit"),
        ("sglang-omni", "minicpm_token2wav"),
    ],
)
def test_unsupported_backend_scope_fails_before_launch(tmp_path, backend, adapter):
    with pytest.raises(OmniFactoryError, match="E_SERVE_BACKEND"):
        command(tmp_path, backend, adapter)


@pytest.mark.parametrize(
    "override",
    [
        {"stages": {"tts_engine": {"factory": {"model_path": "another-model"}}}},
        {"stages": [{"name": "vocoder", "model_path": "another-model"}]},
        {"config_cls": "AnotherPipeline"},
    ],
)
def test_config_cannot_change_tested_model(tmp_path, override):
    with pytest.raises(OmniFactoryError, match="E_SERVE_CONFIG"):
        command(tmp_path, config={"config_cls": "Qwen3TTSPipelineConfig", **override})


@pytest.mark.parametrize("source_path", [None, "another-model"])
def test_sglang_yaml_always_contains_verified_model_before_launch(tmp_path, source_path):
    config = {"config_cls": "Qwen3TTSPipelineConfig"}
    if source_path:
        config["model_path"] = source_path
    command(tmp_path, config=config)
    assert yaml.safe_load((tmp_path / "deploy.yaml").read_text())["model_path"] == str(
        tmp_path / "export"
    )


@pytest.mark.parametrize("backend", ["vllm-omni", "sglang-omni"])
def test_launch_lifecycle_records_backend_and_stops_owned_server(tmp_path, monkeypatch, backend):
    from omnifactory import serving

    artifact = {"adapter": "qwen3_tts", "artifact_id": "sealed-export"}
    monkeypatch.setattr(serving, "verify_export", lambda _: artifact)
    monkeypatch.setattr(serving, "probe", lambda *a, **kw: {"status": "passed", **artifact})
    monkeypatch.setattr(
        serving,
        "http_request",
        lambda *a, **kw: (
            200,
            json.dumps({"data": [{"id": "omnifactory-acceptance"}]}).encode(),
            {},
        ),
    )
    commands, stopped = [], []

    class Process:
        pid = 123456

        def __init__(self, args, **kwargs):
            commands.append(args)
            assert kwargs["start_new_session"]

        def poll(self):
            return None

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(serving.subprocess, "Popen", Process)
    monkeypatch.setattr(serving.os, "killpg", lambda pid, sig: stopped.append(pid))
    config = tmp_path / "deployment.yaml"
    config.write_text("config_cls: Qwen3TTSPipelineConfig\n")
    (tmp_path / "data.jsonl").write_text("{}\n")
    result = serving.validate_deployment(
        tmp_path / "export",
        tmp_path / "data.jsonl",
        tmp_path / "report",
        "server",
        config,
        backend=backend,
    )
    saved = json.loads((tmp_path / "report/acceptance.json").read_text())
    assert result == saved
    assert saved["backend"] == backend and saved["artifact_id"] == "sealed-export"
    assert saved["launch_command"] == commands[0]
    assert saved["export_unchanged_after_serving"] is True
    assert saved["dataset_sha256"] == serving.sha256(tmp_path / "data.jsonl")
    assert saved["deploy_config_sha256"] == serving.sha256(tmp_path / "report/deploy.yaml")
    if backend == "sglang-omni":
        launch_config = Path(commands[0][commands[0].index("--config") + 1])
        assert launch_config == tmp_path / "report/deploy.yaml"
        assert yaml.safe_load(launch_config.read_text())["model_path"] == str(tmp_path / "export")
    assert stopped == [Process.pid]


@pytest.mark.parametrize(
    "adapter,config_name",
    [
        ("qwen3_omni", "qwen3-omni"),
        ("minicpm_o_4_5", "minicpm-o-4_5"),
        ("cosyvoice3", "cosyvoice3"),
        ("personaplex", "personaplex"),
    ],
)
def test_native_pipeline_configs_and_realtime_flag(tmp_path, adapter, config_name):
    config = yaml.safe_load((ROOT / f"examples/deploy-{config_name}-sglang.yaml").read_text())
    args = command(tmp_path, adapter=adapter, config=config)
    assert ("--enable-realtime" in args) == (adapter == "personaplex")
    with pytest.raises(OmniFactoryError, match="E_SERVE_CONFIG"):
        command(tmp_path, adapter=adapter, config={"config_cls": "Qwen3TTSPipelineConfig"})


def test_launch_failure_is_saved_without_masking_error(tmp_path, monkeypatch):
    from omnifactory import serving

    monkeypatch.setattr(
        serving, "verify_export", lambda _: {"adapter": "qwen3_tts", "artifact_id": "id"}
    )

    def fail(*args, **kwargs):
        raise FileNotFoundError("missing executable")

    monkeypatch.setattr(serving.subprocess, "Popen", fail)
    (tmp_path / "data.jsonl").write_text("{}\n")
    config = tmp_path / "deployment.yaml"
    config.write_text("config_cls: Qwen3TTSPipelineConfig\n")
    with pytest.raises(FileNotFoundError, match="missing executable"):
        serving.validate_deployment(
            tmp_path / "export",
            tmp_path / "data.jsonl",
            tmp_path / "report",
            "missing",
            config,
            backend="sglang-omni",
        )
    report = json.loads((tmp_path / "report/acceptance.json").read_text())
    assert report["status"] == "failed" and "missing executable" in report["error"]
    assert report["backend"] == "sglang-omni"


@pytest.mark.parametrize("needs_kill", [False, True])
def test_post_serving_export_mutation_fails_report_after_process_cleanup(
    tmp_path, monkeypatch, needs_kill
):
    import signal

    from omnifactory import serving

    artifact = {"adapter": "qwen3_tts", "artifact_id": "sealed-export"}
    mutation = OmniFactoryError("E_EXPORT_HASH", "voice assets changed during serving")
    verified, lifecycle = [], []

    def verify(export):
        verified.append(export)
        if len(verified) == 1:
            return artifact
        # Final integrity checks must still run after a successful API probe, and
        # a corrupt export must not bypass terminating and reaping the server.
        assert lifecycle[-1] == "reaped"
        raise mutation

    def probe(*args, **kwargs):
        lifecycle.append("probed")
        return {"status": "passed", **artifact, "requests": 1}

    class Process:
        pid = 123456
        killed = False

        def __init__(self, args, **kwargs):
            assert kwargs["start_new_session"] is True
            lifecycle.append("started")

        def poll(self):
            return None

        def wait(self, **kwargs):
            if needs_kill and not self.killed:
                assert kwargs["timeout"] == 15
                lifecycle.append("term-timeout")
                raise serving.subprocess.TimeoutExpired("server", 15)
            lifecycle.append("reaped")
            return 0

    signals = []

    def stop(pid, sig):
        assert pid == Process.pid
        signals.append(sig)
        lifecycle.append("killed" if sig == signal.SIGKILL else "terminated")
        if sig == signal.SIGKILL:
            Process.killed = True

    monkeypatch.setattr(serving, "verify_export", verify)
    monkeypatch.setattr(serving, "probe", probe)
    monkeypatch.setattr(serving.subprocess, "Popen", Process)
    monkeypatch.setattr(serving.os, "killpg", stop)
    monkeypatch.setattr(
        serving,
        "http_request",
        lambda *args, **kwargs: (
            200,
            b'{"data": [{"id": "omnifactory-acceptance"}]}',
            {},
        ),
    )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    config = tmp_path / "deployment.yaml"
    config.write_text("config_cls: Qwen3TTSPipelineConfig\n")
    with pytest.raises(OmniFactoryError, match="E_EXPORT_HASH") as error:
        serving.validate_deployment(
            tmp_path / "export",
            dataset,
            tmp_path / "report",
            "server",
            config,
            backend="sglang-omni",
        )
    assert error.value is mutation
    assert verified == [tmp_path / "export", tmp_path / "export"]
    assert lifecycle[:2] == ["started", "probed"]
    assert lifecycle[-1] == "reaped"
    assert signals == ([signal.SIGTERM, signal.SIGKILL] if needs_kill else [signal.SIGTERM])
    saved = json.loads((tmp_path / "report/acceptance.json").read_text())
    assert saved["status"] == "failed"
    assert saved.get("export_unchanged_after_serving") is not True
    assert saved["artifact_id"] == artifact["artifact_id"]
    assert saved["error"] == str(mutation)
    assert saved["backend"] == "sglang-omni"
    assert saved["launch_command"]
    assert saved["server_log_sha256"] == serving.sha256(tmp_path / "report/server.log")
