"""Tests for the AicodeX module."""

import pytest

from aicodex import AicodeXAssistant, Mode, ModeEngine


# ---------------------------------------------------------------------------
# ModeEngine tests
# ---------------------------------------------------------------------------

class TestModeEngine:
    def test_default_mode_is_coding(self):
        engine = ModeEngine()
        assert engine.current == Mode.CODING

    def test_switch_by_enum(self):
        engine = ModeEngine()
        engine.switch(Mode.DEBUG)
        assert engine.current == Mode.DEBUG

    def test_switch_by_string(self):
        engine = ModeEngine()
        engine.switch("design")
        assert engine.current == Mode.DESIGN

    def test_switch_invalid_string_raises(self):
        engine = ModeEngine()
        with pytest.raises(ValueError, match="Unknown mode"):
            engine.switch("invalid_mode")

    def test_process_coding(self):
        engine = ModeEngine()
        engine.switch(Mode.CODING)
        result = engine.process("sort a list")
        assert "Coding" in result

    def test_process_design(self):
        engine = ModeEngine()
        engine.switch(Mode.DESIGN)
        result = engine.process("microservice architecture")
        assert "Design" in result

    def test_process_debug(self):
        engine = ModeEngine()
        engine.switch(Mode.DEBUG)
        result = engine.process("IndexError on line 42")
        assert "Debug" in result

    def test_process_no_handler_raises(self):
        engine = ModeEngine()
        engine._handlers.clear()
        with pytest.raises(RuntimeError, match="No handler registered"):
            engine.process("anything")

    def test_custom_handler(self):
        engine = ModeEngine()
        engine.register_handler(Mode.CODING, lambda p: f"custom: {p}")
        engine.switch(Mode.CODING)
        assert engine.process("hello") == "custom: hello"


# ---------------------------------------------------------------------------
# AicodeXAssistant tests
# ---------------------------------------------------------------------------

class TestAicodeXAssistant:
    def test_default_mode(self):
        assistant = AicodeXAssistant()
        assert assistant.mode == Mode.CODING

    def test_set_mode_string(self):
        assistant = AicodeXAssistant()
        assistant.set_mode("debug")
        assert assistant.mode == Mode.DEBUG

    def test_toggle_mode_cycles(self):
        assistant = AicodeXAssistant()
        assert assistant.mode == Mode.CODING
        next_mode = assistant.toggle_mode()
        assert next_mode == Mode.DESIGN
        next_mode = assistant.toggle_mode()
        assert next_mode == Mode.DEBUG
        next_mode = assistant.toggle_mode()
        assert next_mode == Mode.CODING

    def test_run_returns_string(self):
        assistant = AicodeXAssistant()
        result = assistant.run("write a hello world function")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_empty_prompt_raises(self):
        assistant = AicodeXAssistant()
        with pytest.raises(ValueError, match="empty"):
            assistant.run("")

    def test_generate_code_restores_mode(self):
        assistant = AicodeXAssistant()
        assistant.set_mode("debug")
        assistant.generate_code("sort a list")
        assert assistant.mode == Mode.DEBUG

    def test_debug_restores_mode(self):
        assistant = AicodeXAssistant()
        assistant.set_mode("design")
        assistant.debug("broken code snippet")
        assert assistant.mode == Mode.DESIGN

    def test_design_restores_mode(self):
        assistant = AicodeXAssistant()
        assistant.set_mode("coding")
        assistant.design("event-driven system")
        assert assistant.mode == Mode.CODING

    def test_fine_tune_empty_raises(self):
        assistant = AicodeXAssistant()
        with pytest.raises(ValueError, match="non-empty"):
            assistant.fine_tune([])

    def test_fine_tune_bad_keys_raises(self):
        assistant = AicodeXAssistant()
        with pytest.raises(ValueError, match="'prompt' and 'response'"):
            assistant.fine_tune([{"question": "x", "answer": "y"}])

    def test_fine_tune_valid(self):
        assistant = AicodeXAssistant()
        assistant.fine_tune([{"prompt": "hello", "response": "world"}])
        assert assistant.status()["fine_tune_queue_size"] == 1

    def test_status(self):
        assistant = AicodeXAssistant()
        status = assistant.status()
        assert "mode" in status
        assert "fine_tune_queue_size" in status
