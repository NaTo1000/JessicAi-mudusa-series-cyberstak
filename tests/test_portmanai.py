"""Tests for the PortmanAI module."""

import pytest

from portmanai import PortmanDebugger, PortmanToolchain
from portmanai.debugger import DebugReport, _infer_severity


# ---------------------------------------------------------------------------
# Severity inference
# ---------------------------------------------------------------------------

class TestSeverityInference:
    def test_critical_keywords(self):
        assert _infer_severity("segfault in thread 0") == "critical"
        assert _infer_severity("null pointer dereference") == "critical"

    def test_high_keywords(self):
        assert _infer_severity("Traceback\nException: boom") == "high"

    def test_medium_keywords(self):
        assert _infer_severity("WARNING: deprecated API used") == "medium"

    def test_low_default(self):
        assert _infer_severity("this is just a note") == "low"


# ---------------------------------------------------------------------------
# PortmanDebugger
# ---------------------------------------------------------------------------

class TestPortmanDebugger:
    def test_analyse_returns_report(self):
        debugger = PortmanDebugger()
        report = debugger.analyse("IndexError: list index out of range\n  File app.py line 42")
        assert isinstance(report, DebugReport)
        assert report.issue
        assert report.severity
        assert 0.0 <= report.confidence <= 1.0

    def test_analyse_empty_raises(self):
        debugger = PortmanDebugger()
        with pytest.raises(ValueError, match="empty"):
            debugger.analyse("")

    def test_index_error_root_cause(self):
        debugger = PortmanDebugger()
        report = debugger.analyse("IndexError: list index out of range")
        assert "index" in report.root_cause.lower()

    def test_none_attribute_root_cause(self):
        debugger = PortmanDebugger()
        report = debugger.analyse("AttributeError: 'NoneType' object has no attribute 'x'")
        assert "none" in report.root_cause.lower() or "NoneType" in report.root_cause

    def test_batch_analyse(self):
        debugger = PortmanDebugger()
        reports = debugger.batch_analyse([
            "IndexError: out of range",
            "SyntaxError: invalid syntax",
        ])
        assert len(reports) == 2

    def test_history_accumulates(self):
        debugger = PortmanDebugger()
        debugger.analyse("error 1")
        debugger.analyse("error 2")
        assert len(debugger.history()) == 2

    def test_clear_history(self):
        debugger = PortmanDebugger()
        debugger.analyse("error")
        debugger.clear_history()
        assert len(debugger.history()) == 0

    def test_to_dict(self):
        debugger = PortmanDebugger()
        report = debugger.analyse("some error")
        d = report.to_dict()
        for key in ("issue", "severity", "root_cause", "suggested_fix", "confidence"):
            assert key in d


# ---------------------------------------------------------------------------
# PortmanToolchain
# ---------------------------------------------------------------------------

class TestPortmanToolchain:
    def test_validate_disallowed_tool(self):
        tc = PortmanToolchain()
        with pytest.raises(ValueError, match="allow-list"):
            tc.run("rm", ["-rf", "/"])

    def test_validate_disallowed_in_describe(self):
        tc = PortmanToolchain()
        with pytest.raises(ValueError, match="allow-list"):
            tc.describe("evil_tool")

    def test_is_available_returns_bool(self):
        tc = PortmanToolchain()
        result = tc.is_available("python3")
        assert isinstance(result, bool)

    def test_available_tools_is_subset_of_allowed(self):
        tc = PortmanToolchain()
        for tool in tc.available_tools():
            assert tool in PortmanToolchain.ALLOWED_TOOLS

    def test_describe_known_tool(self):
        tc = PortmanToolchain()
        info = tc.describe("python3")
        assert info["tool"] == "python3"
        assert info["allowed"] is True

    def test_run_not_found(self):
        tc = PortmanToolchain()
        result = tc.run("strings", ["nonexistent_binary_xyz"])
        # strings may or may not be installed; either way we get a ToolResult
        assert result.tool == "strings"
        assert isinstance(result.returncode, int)
