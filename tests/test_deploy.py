import json

import pytest

from scripts import deploy

CARD = {
    "model_id": "Qwen/Qwen3-8B-AWQ",
    "quantization": "awq",
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.9,
    "tensor_parallel_size": 1,
    "tool_call_parser": "hermes",
    "reasoning_parser": "qwen3",
}


def flags(argv):
    return argv[argv.index(deploy.IMAGE) + 1:]


def test_build_command_awq():
    argv, _ = deploy.build_command(CARD, runtime="docker")
    assert argv[:3] == ["docker", "run", "-d"]
    f = flags(argv)
    assert f[:2] == ["--model", "Qwen/Qwen3-8B-AWQ"]
    assert "--quantization" not in f  # auto-detected → fast Marlin kernels
    assert f[f.index("--max-model-len") + 1] == "32768"
    assert f[f.index("--gpu-memory-utilization") + 1] == "0.9"
    assert "--enable-auto-tool-choice" in f
    assert f[f.index("--tool-call-parser") + 1] == "hermes"
    assert f[f.index("--reasoning-parser") + 1] == "qwen3"
    assert "--tensor-parallel-size" not in f


def test_build_command_tp2_and_no_reasoning_parser():
    card = dict(CARD, tensor_parallel_size=2, reasoning_parser=None)
    f = flags(deploy.build_command(card, runtime="docker")[0])
    assert f[f.index("--tensor-parallel-size") + 1] == "2"
    assert "--reasoning-parser" not in f


def test_build_command_passes_hf_token_by_name_only(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    argv, _ = deploy.build_command(CARD, runtime="docker")
    assert "HF_TOKEN" in argv
    assert not any("hf_secret" in a for a in argv)


def test_every_flag_has_reason():
    argv, reasons = deploy.build_command(CARD, runtime="docker")
    explained = {flag for flag, _ in reasons}
    for a in flags(argv):
        if a.startswith("--"):
            assert a in explained, a
    for a in ("--gpus", "--ipc", "-p", "-v"):
        assert a in explained


def test_deploy_without_card(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        deploy.main(["--card", str(tmp_path / "missing.json")])
    assert e.value.code == 1
    assert "make find" in capsys.readouterr().out


def test_wait_ready_succeeds_when_models_endpoint_answers(monkeypatch):
    calls = iter([False, False, True])
    monkeypatch.setattr(deploy, "server_ready", lambda url: next(calls))
    monkeypatch.setattr(deploy, "server_state", lambda runtime: "running")
    monkeypatch.setattr(deploy, "last_log_line", lambda runtime: "Loading weights...")
    deploy.wait_ready("http://x/v1", "docker", timeout_s=60, poll_s=0)


def test_wait_ready_reports_crash(monkeypatch):
    monkeypatch.setattr(deploy, "server_ready", lambda url: False)
    monkeypatch.setattr(deploy, "server_state", lambda runtime: "exited")
    monkeypatch.setattr(deploy, "server_logs", lambda runtime, n: "ValueError: not enough memory")
    monkeypatch.setattr(deploy, "last_log_line", lambda runtime: "")
    with pytest.raises(deploy.DeployError) as e:
        deploy.wait_ready("http://x/v1", "docker", timeout_s=60, poll_s=0)
    assert "not enough memory" in str(e.value)


def test_wait_ready_times_out(monkeypatch):
    monkeypatch.setattr(deploy, "server_ready", lambda url: False)
    monkeypatch.setattr(deploy, "server_state", lambda runtime: "running")
    monkeypatch.setattr(deploy, "server_logs", lambda runtime, n: "still loading")
    monkeypatch.setattr(deploy, "last_log_line", lambda runtime: "")
    with pytest.raises(deploy.DeployError) as e:
        deploy.wait_ready("http://x/v1", "docker", timeout_s=0, poll_s=0)
    assert "timed out" in str(e.value).lower()


def test_dry_run_prints_command_without_running(tmp_path, capsys, monkeypatch):
    card = tmp_path / "deployment.json"
    card.write_text(json.dumps(CARD))
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: pytest.fail("should not run"))
    deploy.main(["--card", str(card), "--dry-run", "--runtime", "docker"])
    out = capsys.readouterr().out
    assert "docker run" in out and "--tool-call-parser hermes" in out


def test_default_runtime_is_native(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("RUNTIME", raising=False)
    card = tmp_path / "deployment.json"
    card.write_text(json.dumps(CARD))
    deploy.main(["--card", str(card), "--dry-run"])
    out = capsys.readouterr().out
    assert "uv tool run" in out and "docker run" not in out
    assert deploy.build_command(CARD)[0][0] == "uv"


def test_runtime_env_selects_docker(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RUNTIME", "docker")
    card = tmp_path / "deployment.json"
    card.write_text(json.dumps(CARD))
    deploy.main(["--card", str(card), "--dry-run"])
    assert "docker run" in capsys.readouterr().out


def test_port_bound_to_localhost_only():
    # Docker bypasses host firewalls; binding 0.0.0.0 would expose an
    # unauthenticated model server to the internet on a cloud VM.
    argv, _ = deploy.build_command(CARD, runtime="docker")
    assert argv[argv.index("-p") + 1] == "127.0.0.1:8000:8000"


# --- native runtime (no Docker): vLLM installed by uv --------------------------

def test_native_command_uses_pinned_vllm_on_localhost():
    argv, reasons = deploy.build_command(CARD, runtime="native")
    assert "docker" not in argv
    assert argv[:argv.index("serve") + 1] == [
        "uv", "tool", "run", "--python", "3.12", "--exclude-newer", deploy.EXCLUDE_NEWER,
        "--from", f"vllm=={deploy.VLLM_VERSION}", "vllm", "serve"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    assert "--model" in argv


def test_native_and_docker_share_vllm_flags():
    docker_flags = flags(deploy.build_command(CARD, runtime="docker")[0])
    native = deploy.build_command(CARD, runtime="native")[0]
    native_flags = native[native.index("serve") + 1:]
    for i, a in enumerate(docker_flags):
        assert a in native_flags


def test_native_state_without_pid_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "NATIVE_DIR", tmp_path)
    assert deploy.server_state("native") == "missing"


def test_native_state_with_dead_pid_is_exited(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "NATIVE_DIR", tmp_path)
    (tmp_path / "server.pid").write_text("999999999")
    assert deploy.server_state("native") == "exited"


def test_vllm_version_matches_image_tag():
    assert deploy.IMAGE.endswith(":v" + deploy.VLLM_VERSION)


def test_wait_ready_prints_only_new_log_lines(monkeypatch, capsys):
    ready = iter([False, False, False, True])
    lines = iter(["Loading weights", "Loading weights", "Capturing CUDA graphs"])
    monkeypatch.setattr(deploy, "server_ready", lambda url: next(ready))
    monkeypatch.setattr(deploy, "server_state", lambda runtime: "running")
    monkeypatch.setattr(deploy, "last_log_line", lambda runtime: next(lines))
    deploy.wait_ready("http://x/v1", "native", timeout_s=60, poll_s=0)
    out = capsys.readouterr().out
    assert out.count("Loading weights") == 1
    assert out.count("Capturing CUDA graphs") == 1


# --- review fixes ---------------------------------------------------------------

def test_native_command_pins_dependency_resolution_date():
    argv, _ = deploy.build_command(CARD, runtime="native")
    assert argv[argv.index("--exclude-newer") + 1] == deploy.EXCLUDE_NEWER


def test_docker_runs_as_current_user_with_own_hf_home():
    argv, _ = deploy.build_command(CARD, runtime="docker")
    user = argv[argv.index("--user") + 1]
    assert user.count(":") == 1 and all(p.isdigit() for p in user.split(":"))
    assert "HF_HOME=/hf-cache" in argv
    assert any(a.endswith(":/hf-cache") for a in argv)


def test_refresh_utilization_uses_memory_free_now():
    from scripts.find_model import GPU
    card = dict(CARD, gpu_memory_utilization=0.1, est_vram_gib=12.2)
    out = deploy.refresh_utilization(card, [GPU(0, "g", 24564, 24100, 8.6)])
    assert out["gpu_memory_utilization"] == 0.9


def test_refresh_utilization_fails_when_model_no_longer_fits():
    from scripts.find_model import GPU
    card = dict(CARD, est_vram_gib=12.2)
    with pytest.raises(deploy.DeployError) as e:
        deploy.refresh_utilization(card, [GPU(0, "g", 24564, 6000, 8.6)])
    assert "make find" in str(e.value)
