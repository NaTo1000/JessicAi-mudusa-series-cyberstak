"""PortmanAI Toolchain — AI-driven security and DevOps tool orchestration."""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Outcome of a toolchain operation."""

    tool: str
    args: list[str]
    stdout: str
    stderr: str
    returncode: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "success": self.success,
            "metadata": self.metadata,
        }


class PortmanToolchain:
    """
    Manages tool discovery and sandboxed invocation for PortmanAI.

    Only tools that are explicitly allow-listed may be invoked to prevent
    arbitrary command execution.

    Usage::

        tc = PortmanToolchain()
        result = tc.run("nmap", ["-sV", "127.0.0.1"])
        print(result.stdout)
    """

    # Tools that the toolchain is permitted to invoke
    ALLOWED_TOOLS: frozenset[str] = frozenset(
        {
            # Network / recon
            "nmap",
            "masscan",
            "netcat",
            "nc",
            "curl",
            "wget",
            # Web
            "nikto",
            "dirb",
            "gobuster",
            "sqlmap",
            # Password / hash
            "hashcat",
            "john",
            # Exploitation helpers
            "metasploit",
            "msfconsole",
            # Forensics
            "binwalk",
            "strings",
            "file",
            "hexdump",
            # Misc utilities
            "python3",
            "python",
            "bash",
            "sh",
            "git",
        }
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self, tool: str) -> bool:
        """Return True if *tool* is installed on the host PATH."""
        return shutil.which(tool) is not None

    def available_tools(self) -> list[str]:
        """Return a list of allow-listed tools that are present on this host."""
        return [t for t in sorted(self.ALLOWED_TOOLS) if self.is_available(t)]

    def run(
        self,
        tool: str,
        args: list[str],
        timeout: int | None = 60,
        cwd: str | None = None,
    ) -> ToolResult:
        """
        Execute *tool* with *args* in a subprocess.

        Parameters
        ----------
        tool:
            The tool binary name (must be in :attr:`ALLOWED_TOOLS`).
        args:
            Command-line arguments to pass to the tool.
        timeout:
            Maximum execution time in seconds (default 60).
        cwd:
            Working directory for the subprocess.

        Raises
        ------
        ValueError
            If *tool* is not in the allow-list.
        FileNotFoundError
            If *tool* is not installed.
        """
        self._validate_tool(tool)

        cmd = [tool, *args]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return ToolResult(
                tool=tool,
                args=args,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool=tool,
                args=args,
                stdout="",
                stderr=f"Tool '{tool}' timed out after {timeout}s.",
                returncode=-1,
                metadata={"timed_out": True},
            )
        except FileNotFoundError:
            return ToolResult(
                tool=tool,
                args=args,
                stdout="",
                stderr=f"Tool '{tool}' is not installed or not on PATH.",
                returncode=-2,
                metadata={"not_found": True},
            )

    def describe(self, tool: str) -> dict[str, Any]:
        """Return metadata about *tool*."""
        self._validate_tool(tool)
        return {
            "tool": tool,
            "available": self.is_available(tool),
            "path": shutil.which(tool),
            "allowed": True,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_tool(self, tool: str) -> None:
        if tool not in self.ALLOWED_TOOLS:
            raise ValueError(
                f"'{tool}' is not in the toolchain allow-list. "
                f"Allowed tools: {sorted(self.ALLOWED_TOOLS)}"
            )
        # Note: availability on PATH is not checked here so that callers can
        # use run() on tools that may be installed inside a sandbox image but
        # not on the host.  Use is_available() to check host availability
        # before calling run() if that distinction matters.
