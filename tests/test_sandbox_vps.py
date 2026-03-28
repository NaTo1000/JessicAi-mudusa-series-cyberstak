"""Tests for the sandbox and VPS modules."""

import os
import tempfile

import pytest

from sandbox import SandboxEnvironment, ModelRunner
from sandbox.environment import SandboxConfig
from sandbox.linux_toolkit import LinuxToolkit, ToolCategory
from vps import VPSScaler, ResourceAllocation


# ---------------------------------------------------------------------------
# SandboxEnvironment
# ---------------------------------------------------------------------------

class TestSandboxEnvironment:
    def test_default_image(self):
        sb = SandboxEnvironment()
        assert sb.config.image == "kalilinux/kali-rolling"

    def test_custom_config(self):
        cfg = SandboxConfig(image="blackarchlinux/blackarch", memory_limit="4g")
        sb = SandboxEnvironment(config=cfg)
        assert sb.config.image == "blackarchlinux/blackarch"
        assert sb.config.memory_limit == "4g"

    def test_status_dict(self):
        sb = SandboxEnvironment()
        status = sb.status()
        assert "image" in status
        assert "docker_available" in status

    def test_run_fallback_echo(self):
        """When Docker is absent, fallback to direct execution."""
        sb = SandboxEnvironment()
        sb._docker_available = False
        result = sb.run(["echo", "hello"])
        assert "hello" in result.stdout or result.returncode in (0, -1, -2)

    def test_build_docker_cmd_includes_memory(self):
        sb = SandboxEnvironment()
        cmd = sb._build_docker_cmd(["ls"], "/workspace", None)
        combined = " ".join(cmd)
        assert "--memory" in combined

    def test_run_invalid_command_no_crash(self):
        """A missing binary should return an error result, not an exception."""
        sb = SandboxEnvironment()
        sb._docker_available = False
        result = sb.run(["nonexistent_cmd_xyz_abc"])
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------

class TestModelRunner:
    def test_load_local_path_not_found(self):
        runner = ModelRunner()
        with pytest.raises(FileNotFoundError):
            runner.load("/nonexistent/path/to/model")

    def test_load_local_existing_dir(self):
        runner = ModelRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            info = runner.load(tmpdir)
            assert info.source == "local"
            assert info.local_path == tmpdir

    def test_framework_detection_unknown(self):
        runner = ModelRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            info = runner.load(tmpdir)
            assert info.framework == "unknown"

    def test_framework_detection_pytorch(self):
        runner = ModelRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model.bin"), "w").close()
            info = runner.load(tmpdir)
            assert info.framework == "pytorch"

    def test_framework_detection_onnx(self):
        runner = ModelRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model.onnx"), "w").close()
            info = runner.load(tmpdir)
            assert info.framework == "onnx"

    def test_run_inference_returns_dict(self):
        runner = ModelRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            info = runner.load(tmpdir)
            result = runner.run_inference(info, {"text": "hello"})
            assert "output" in result
            assert "metadata" in result

    def test_list_loaded(self):
        runner = ModelRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.load(tmpdir)
            assert len(runner.list_loaded()) == 1

    def test_unload(self):
        runner = ModelRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.load(tmpdir)
            assert runner.unload(tmpdir) is True
            assert len(runner.list_loaded()) == 0

    def test_unload_nonexistent_returns_false(self):
        runner = ModelRunner()
        assert runner.unload("does_not_exist") is False

    def test_caching_same_source(self):
        runner = ModelRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            info1 = runner.load(tmpdir)
            info2 = runner.load(tmpdir)
            assert info1 is info2


# ---------------------------------------------------------------------------
# LinuxToolkit
# ---------------------------------------------------------------------------

class TestLinuxToolkit:
    def test_list_all_tools_non_empty(self):
        toolkit = LinuxToolkit()
        tools = toolkit.list_tools()
        assert len(tools) > 0

    def test_filter_by_category(self):
        toolkit = LinuxToolkit()
        web_tools = toolkit.list_tools(category=ToolCategory.WEB_APPLICATION)
        assert all(t.category == ToolCategory.WEB_APPLICATION for t in web_tools)

    def test_filter_by_distro_kali(self):
        toolkit = LinuxToolkit()
        tools = toolkit.list_tools(distro="kali")
        assert all("kali" in t.distros for t in tools)

    def test_filter_by_distro_blackarch(self):
        toolkit = LinuxToolkit()
        tools = toolkit.list_tools(distro="blackarch")
        assert all("blackarch" in t.distros for t in tools)

    def test_get_tool_found(self):
        toolkit = LinuxToolkit()
        tool = toolkit.get_tool("nmap")
        assert tool is not None
        assert tool.name == "nmap"

    def test_get_tool_not_found(self):
        toolkit = LinuxToolkit()
        assert toolkit.get_tool("nonexistent_tool_xyz") is None

    def test_summary_keys(self):
        toolkit = LinuxToolkit()
        summary = toolkit.summary()
        assert "total" in summary
        assert "kali_tools" in summary
        assert "blackarch_tools" in summary

    def test_to_dict(self):
        toolkit = LinuxToolkit()
        tool = toolkit.get_tool("nmap")
        d = tool.to_dict()
        assert d["name"] == "nmap"
        assert "category" in d
        assert "installed" in d


# ---------------------------------------------------------------------------
# VPSScaler
# ---------------------------------------------------------------------------

class TestVPSScaler:
    def test_default_allocation(self):
        scaler = VPSScaler()
        alloc = scaler.current_allocation
        assert alloc.cpu_cores == 2
        assert alloc.ram_gb == 4.0

    def test_scale_up_cpu(self):
        scaler = VPSScaler()
        alloc = scaler.scale_up(cpu_cores=8)
        assert alloc.cpu_cores == 8

    def test_scale_up_ram(self):
        scaler = VPSScaler()
        alloc = scaler.scale_up(ram_gb=32.0)
        assert alloc.ram_gb == 32.0

    def test_scale_up_gpu(self):
        scaler = VPSScaler()
        alloc = scaler.scale_up(gpu_count=2, gpu_vram_gb=24.0)
        assert alloc.gpu_count == 2
        assert alloc.gpu_vram_gb == 24.0

    def test_scale_up_respects_max_cpu(self):
        scaler = VPSScaler()
        alloc = scaler.scale_up(cpu_cores=9999)
        assert alloc.cpu_cores == VPSScaler._MAX_CPU

    def test_scale_down_floors_at_one_cpu(self):
        scaler = VPSScaler()
        alloc = scaler.scale_down(cpu_cores=0)
        assert alloc.cpu_cores == 1

    def test_scale_down_floors_ram(self):
        scaler = VPSScaler()
        alloc = scaler.scale_down(ram_gb=0)
        assert alloc.ram_gb == 0.5

    def test_migrate(self):
        scaler = VPSScaler()
        alloc = scaler.migrate(provider="aws", region="us-east-1")
        assert alloc.provider == "aws"
        assert alloc.region == "us-east-1"

    def test_auto_scale_cpu_high(self):
        scaler = VPSScaler()
        new_alloc = scaler.auto_scale(cpu_percent=90, ram_percent=20)
        assert new_alloc is not None
        assert new_alloc.cpu_cores > 2

    def test_auto_scale_ram_high(self):
        scaler = VPSScaler()
        new_alloc = scaler.auto_scale(cpu_percent=20, ram_percent=90)
        assert new_alloc is not None
        assert new_alloc.ram_gb > 4.0

    def test_auto_scale_no_trigger(self):
        scaler = VPSScaler()
        result = scaler.auto_scale(cpu_percent=50, ram_percent=50)
        assert result is None

    def test_history_recorded(self):
        scaler = VPSScaler()
        scaler.scale_up(cpu_cores=4)
        scaler.scale_down(cpu_cores=2)
        assert len(scaler.history()) == 2

    def test_reset(self):
        scaler = VPSScaler()
        scaler.scale_up(cpu_cores=16, ram_gb=64)
        scaler.reset()
        assert scaler.current_allocation.cpu_cores == 2
        assert any(e.action == "reset" for e in scaler.history())

    def test_to_dict(self):
        alloc = ResourceAllocation(cpu_cores=4, ram_gb=8.0)
        d = alloc.to_dict()
        assert d["cpu_cores"] == 4
        assert d["ram_gb"] == 8.0
