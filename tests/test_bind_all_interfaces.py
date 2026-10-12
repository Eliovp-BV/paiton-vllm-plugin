"""Model servers and launchers listen on every interface by default; PAITON_BIND or --host keeps them local.

The shell launchers run against a stand-in `docker` on PATH that records its arguments, so nothing starts.
"""
import importlib.util
import os
import sys
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FAKE_DOCKER = """#!/usr/bin/env bash
{ printf 'CALL\\n'; printf '%s\\n' "$@"; printf 'ENV VLLM_API_KEY=%s\\n' "${VLLM_API_KEY:-}"; } >> "$FAKE_DOCKER_LOG"
case "$1" in
  container|inspect) exit 1 ;;
esac
exit 0
"""
needs_kfd = pytest.mark.skipif(not os.path.exists("/dev/kfd") or not os.path.isdir("/dev/dri"),
                               reason="the launcher checks for ROCm devices first")


def run_launcher(tmp_path, script, *args, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    log.write_text("")
    environment = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), "FAKE_DOCKER_LOG": str(log),
                   "XDG_CACHE_HOME": str(tmp_path / "xdg"), **env}
    result = subprocess.run(["bash", str(MODELS / script), *args], env=environment, cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    calls = [call.strip("\n").split("\n") for call in log.read_text().split("CALL\n") if call.strip()]
    runs = [call for call in calls if call[0] == "run"]
    return result, runs


def published(argv):
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == "-p"]


DOCKER_SCRIPTS = [
    ("GPT-OSS-20B/serve-docker.sh", (), {}, "8020:8020", True),
    ("MiniCPM5-2B/serve-docker.sh", (), {}, "8036:8036", True),
    ("Qwen3.8-NEO-CODER-MAX/serve-docker.sh", (), {}, "8000:8000", True),
    ("Qwen-Image-2.1/serve-docker.sh", (), {}, "8191:8191", False),
    pytest.param("Ornith-1.5/serve-docker.sh", (), {}, "8000:8000", True, marks=needs_kfd),
    pytest.param("Qwen3-Coder-30B/serve-docker.sh", (), {"PAITON_CACHE_DIR": "cache"}, "8010:8010", True,
                 marks=needs_kfd),
    pytest.param("FLUX.2-klein/run.sh", ("serve", "--port", "7860"), {"PAITON_OUTPUTS": "outputs"}, "7860:7860",
                 False, marks=needs_kfd),
]


@pytest.mark.parametrize("script,args,env,ports,vllm", DOCKER_SCRIPTS)
def test_docker_launchers_publish_on_every_interface(tmp_path, script, args, env, ports, vllm):
    env = {k: str(tmp_path / v) for k, v in env.items()}
    _, runs = run_launcher(tmp_path, script, *args, **env)
    assert len(runs) == 1 and published(runs[0]) == ["0.0.0.0:" + ports]
    assert not any("VLLM_API_KEY" in arg for arg in runs[0][:-1])
    _, runs = run_launcher(tmp_path, script, *args, PAITON_BIND="127.0.0.1", **env)
    assert published(runs[0]) == ["127.0.0.1:" + ports]
    if vllm:
        _, runs = run_launcher(tmp_path, script, *args, PAITON_API_KEY="k-test-123", **env)
        argv = runs[0]
        assert "VLLM_API_KEY" in argv and argv[argv.index("VLLM_API_KEY") - 1] in ("-e", "--env")
        assert not any("k-test-123" in arg for arg in argv[:-1]) and argv[-1] == "ENV VLLM_API_KEY=k-test-123"


def test_qwen38_dry_run_shows_every_interface(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    result, runs = run_launcher(tmp_path, "Qwen3.8/serve-docker.sh", "--dry-run", HF_HUB_CACHE=str(hub))
    assert runs == [] and "-p 0.0.0.0:8000:8000" in result.stdout
    assert "From another system: http://<this-machine-ip>:8000/v1" in result.stdout
    result, _ = run_launcher(tmp_path, "Qwen3.8/serve-docker.sh", "--dry-run", HF_HUB_CACHE=str(hub),
                             PAITON_BIND="127.0.0.1", PAITON_API_KEY="k-test-123")
    assert "-p 127.0.0.1:8000:8000" in result.stdout and "--env VLLM_API_KEY" in result.stdout
    assert "k-test-123" not in result.stdout and "From another system" not in result.stdout


@pytest.mark.parametrize("compose", ["FLUX.2-klein/compose.yaml", "MiniMax-H3/compose.yaml", "Wan2.2/compose.yaml"])
def test_compose_ports_default_to_every_interface(compose):
    ports = [line for line in (MODELS / compose).read_text().splitlines() if line.strip().startswith("ports:")]
    assert ports and all('["${PAITON_BIND:-0.0.0.0}:' in line for line in ports)


def load_server(directory):
    spec = importlib.util.spec_from_file_location(f"bind_server_{directory}", MODELS / directory / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_of(argv):
    return argv[argv.index("--host") + 1]


@pytest.mark.parametrize("directory", ["GPT-OSS-20B", "Qwen3-Coder-30B", "MiniCPM5-2B"])
def test_python_servers_listen_on_every_interface(directory):
    server = load_server(directory)
    assert host_of(server.command("/checkpoint")) == "0.0.0.0"
    hub = SimpleNamespace(snapshot_download=lambda *a, **kw: "/existing/checkpoint")
    for extra, host in (((), "0.0.0.0"), (("--host", "127.0.0.1", "--api-key", "k-test-123"), "127.0.0.1")):
        with (patch.dict(os.environ, {"VLLM_PLUGINS": "register_paiton_models"}, clear=True),
              patch.dict(sys.modules, {"huggingface_hub": hub}),
              patch.object(sys, "argv", ["server.py", "--offline", *extra]),
              patch.object(server, "prepare", return_value=Path("/existing/checkpoint"), create=True),
              patch.object(os, "execv") as execute):
            server.main()
            argv = execute.call_args.args[1]
            assert host_of(argv) == host
            assert "k-test-123" not in argv and os.environ.get("VLLM_API_KEY") == (extra and "k-test-123" or None)


def test_gpt_oss_qualification_keeps_development_endpoints_local():
    server = load_server("GPT-OSS-20B")
    assert host_of(server.command("/checkpoint", qualification=True)) == "127.0.0.1"
    assert host_of(server.command("/checkpoint", qualification=True, host="0.0.0.0")) == "0.0.0.0"


def test_qwen_image_module_server_default():
    source = (MODELS / "Qwen-Image-2.1" / "paiton_image21" / "__main__.py").read_text()
    assert 'serve.add_argument("--host",default="0.0.0.0"' in source
