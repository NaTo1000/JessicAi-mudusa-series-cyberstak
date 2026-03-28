"""Sandbox Environment — manages isolated Linux containers for tool execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxConfig:
    """Runtime configuration for a sandbox instance."""

    image: str = "kalilinux/kali-rolling"
    memory_limit: str = "2g"
    cpu_quota: float = 1.0  # fractional CPUs
    network_mode: str = "none"
    read_only_root: bool = True
    auto_remove: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Output from a sandboxed command execution."""

    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    container_id: str = ""

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "container_id": self.container_id,
        }


class SandboxEnvironment:
    """
    Manages a sandboxed Linux environment for secure tool execution.

    Wraps Docker to provide:
    - Kali-Linux / Black Arch image selection
    - Network isolation
    - Memory and CPU limits
    - Read-only root filesystem

    Usage::

        sb = SandboxEnvironment()
        result = sb.run(["nmap", "-sV", "127.0.0.1"])
        print(result.stdout)
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._docker_available = shutil.which("docker") is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> SandboxConfig:
        return self._config

    @property
    def docker_available(self) -> bool:
        return self._docker_available

    def run(
        self,
        command: list[str],
        workdir: str = "/workspace",
        bind_volume: str | None = None,
        timeout: int = 120,
    ) -> SandboxResult:
        """
        Run *command* inside the sandbox container.

        Parameters
        ----------
        command:
            The command and its arguments to execute.
        workdir:
            Working directory inside the container.
        bind_volume:
            Optional host path to mount at ``/workspace`` inside the container.
        timeout:
            Maximum execution time in seconds.

        Returns
        -------
        SandboxResult
        """
        if not self._docker_available:
            return self._fallback_run(command, timeout)

        docker_cmd = self._build_docker_cmd(command, workdir, bind_volume)
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return SandboxResult(
                command=command,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                command=command,
                stdout="",
                stderr=f"Sandbox timed out after {timeout}s.",
                returncode=-1,
            )

    def run_script(self, script: str, timeout: int = 120) -> SandboxResult:
        """
        Write *script* to a temporary file and run it inside the sandbox.

        The script is executed with ``/bin/bash`` inside the container.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as fh:
            fh.write(script)
            script_path = fh.name

        try:
            return self.run(
                ["/bin/bash", "/workspace/script.sh"],
                bind_volume=os.path.dirname(script_path),
                timeout=timeout,
            )
        finally:
            os.unlink(script_path)

    def pull_image(self) -> bool:
        """Pull the configured sandbox image. Returns True on success."""
        if not self._docker_available:
            return False
        result = subprocess.run(
            ["docker", "pull", self._config.image],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def status(self) -> dict[str, Any]:
        return {
            "image": self._config.image,
            "docker_available": self._docker_available,
            "memory_limit": self._config.memory_limit,
            "cpu_quota": self._config.cpu_quota,
            "network_mode": self._config.network_mode,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_docker_cmd(
        self,
        command: list[str],
        workdir: str,
        bind_volume: str | None,
    ) -> list[str]:
        cfg = self._config
        docker_cmd = [
            "docker", "run",
            "--rm" if cfg.auto_remove else "",
            f"--memory={cfg.memory_limit}",
            f"--cpus={cfg.cpu_quota}",
            f"--network={cfg.network_mode}",
            "--workdir", workdir,
        ]
        if cfg.read_only_root:
            docker_cmd.append("--read-only")
        for key, val in cfg.extra_env.items():
            docker_cmd += ["-e", f"{key}={val}"]
        if bind_volume:
            docker_cmd += ["-v", f"{bind_volume}:/workspace:ro"]
        docker_cmd = [arg for arg in docker_cmd if arg]
        docker_cmd.append(cfg.image)
        docker_cmd.extend(command)
        return docker_cmd

    @staticmethod
    def _fallback_run(command: list[str], timeout: int) -> SandboxResult:
        """Run command directly (no Docker) with a warning."""
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return SandboxResult(
                command=command,
                stdout=proc.stdout,
                stderr="[WARNING] Docker not available — running directly on host.\n"
                + proc.stderr,
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                command=command,
                stdout="",
                stderr=f"Timed out after {timeout}s.",
                returncode=-1,
            )
        except FileNotFoundError as exc:
            return SandboxResult(
                command=command,
                stdout="",
                stderr=str(exc),
                returncode=-2,
            )
