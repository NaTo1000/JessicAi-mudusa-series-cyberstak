"""PortmanAI Debugger — root-cause analysis and automated fix suggestions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DebugReport:
    """Result produced by the PortmanAI debugger."""

    issue: str
    severity: str  # "low" | "medium" | "high" | "critical"
    root_cause: str
    suggested_fix: str
    confidence: float  # 0.0 – 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "severity": self.severity,
            "root_cause": self.root_cause,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Severity heuristics (very lightweight without a real LLM backend)
# ---------------------------------------------------------------------------
_SEVERITY_PATTERNS: list[tuple[str, str]] = [
    (r"segfault|null\s*pointer|memory\s*corruption|heap.*overflow", "critical"),
    (r"exception|error|traceback|panic|fatal", "high"),
    (r"warning|deprecated|todo|fixme|hack", "medium"),
    (r"note|info|style|lint", "low"),
]


def _infer_severity(text: str) -> str:
    lower = text.lower()
    for pattern, severity in _SEVERITY_PATTERNS:
        if re.search(pattern, lower):
            return severity
    return "low"


class PortmanDebugger:
    """
    AI-powered debugger for code and runtime artefacts.

    Usage::

        debugger = PortmanDebugger()
        report = debugger.analyse("IndexError: list index out of range\n  File app.py line 42")
        print(report.suggested_fix)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._history: list[DebugReport] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, error_or_code: str) -> DebugReport:
        """
        Analyse *error_or_code* and produce a :class:`DebugReport`.

        Parameters
        ----------
        error_or_code:
            A stack-trace, error message, or code snippet containing a bug.
        """
        if not error_or_code or not error_or_code.strip():
            raise ValueError("Input must not be empty.")

        severity = _infer_severity(error_or_code)
        report = DebugReport(
            issue=self._extract_issue(error_or_code),
            severity=severity,
            root_cause=self._infer_root_cause(error_or_code),
            suggested_fix=self._suggest_fix(error_or_code),
            confidence=self._estimate_confidence(error_or_code),
            metadata={"raw_input_length": len(error_or_code)},
        )
        self._history.append(report)
        return report

    def batch_analyse(self, inputs: list[str]) -> list[DebugReport]:
        """Analyse multiple error strings and return one report per input."""
        return [self.analyse(inp) for inp in inputs]

    def history(self) -> list[DebugReport]:
        """Return all debug reports produced in this session."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Internal heuristic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_issue(text: str) -> str:
        first_line = text.strip().splitlines()[0]
        return first_line[:200]

    @staticmethod
    def _infer_root_cause(text: str) -> str:
        lower = text.lower()
        if "index" in lower and "range" in lower:
            return "List or array accessed beyond its valid index bounds."
        if "none" in lower and "attribute" in lower:
            return "Attempted to access an attribute on a NoneType object."
        if "import" in lower:
            return "Missing or incorrectly named module import."
        if "syntax" in lower:
            return "Invalid Python / language syntax in the source file."
        if "timeout" in lower:
            return "Operation exceeded the allowed time limit."
        if "permission" in lower or "access denied" in lower:
            return "Insufficient filesystem or network permissions."
        return "Unknown root cause — manual inspection recommended."

    @staticmethod
    def _suggest_fix(text: str) -> str:
        lower = text.lower()
        if "index" in lower and "range" in lower:
            return (
                "Guard the access with a bounds check: "
                "`if index < len(collection): ...`"
            )
        if "none" in lower and "attribute" in lower:
            return "Check for None before accessing attributes: `if obj is not None: ...`"
        if "import" in lower:
            return "Verify the module name and ensure it is installed: `pip install <module>`"
        if "syntax" in lower:
            return "Run `python -m py_compile <file>` to locate the syntax error."
        if "timeout" in lower:
            return "Increase the timeout threshold or optimise the slow operation."
        if "permission" in lower or "access denied" in lower:
            return "Check file/directory permissions or run with appropriate privileges."
        return "Review the stack trace carefully and add defensive error handling."

    @staticmethod
    def _estimate_confidence(text: str) -> float:
        keywords = [
            "traceback", "error", "exception", "line", "file", "warning"
        ]
        matches = sum(1 for kw in keywords if kw in text.lower())
        return min(0.5 + matches * 0.08, 0.98)
