"""AicodeX main assistant — orchestrates coding, design, and debug modes."""

from __future__ import annotations

from typing import Any

from .modes import Mode, ModeEngine


class AicodeXAssistant:
    """
    High-level interface for the AicodeX coding assistant.

    Usage::

        assistant = AicodeXAssistant()
        assistant.set_mode("coding")
        result = assistant.run("Write a Python quicksort implementation")
        print(result)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._engine = ModeEngine()

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        return self._engine.current

    def set_mode(self, mode: str | Mode) -> None:
        """Switch the assistant to *mode*."""
        self._engine.switch(mode)

    def toggle_mode(self) -> Mode:
        """Cycle through CODING → DESIGN → DEBUG → CODING …"""
        cycle = [Mode.CODING, Mode.DESIGN, Mode.DEBUG]
        current_index = cycle.index(self._engine.current)
        next_mode = cycle[(current_index + 1) % len(cycle)]
        self._engine.switch(next_mode)
        return next_mode

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def run(self, prompt: str) -> str:
        """Process *prompt* in the current mode and return the response."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        return self._engine.process(prompt)

    def generate_code(self, description: str, language: str = "python") -> str:
        """
        Generate code for *description* in *language*.

        Switches temporarily to CODING mode if not already in it.
        """
        original_mode = self._engine.current
        self._engine.switch(Mode.CODING)
        result = self._engine.process(
            f"Language: {language}\nTask: {description}"
        )
        self._engine.switch(original_mode)
        return result

    def debug(self, code_snippet: str) -> str:
        """
        Analyse *code_snippet* for bugs and suggest fixes.

        Switches temporarily to DEBUG mode.
        """
        original_mode = self._engine.current
        self._engine.switch(Mode.DEBUG)
        result = self._engine.process(code_snippet)
        self._engine.switch(original_mode)
        return result

    def design(self, specification: str) -> str:
        """
        Generate a system / architecture design from *specification*.

        Switches temporarily to DESIGN mode.
        """
        original_mode = self._engine.current
        self._engine.switch(Mode.DESIGN)
        result = self._engine.process(specification)
        self._engine.switch(original_mode)
        return result

    # ------------------------------------------------------------------
    # Fine-tuning helpers
    # ------------------------------------------------------------------

    def fine_tune(self, examples: list[dict[str, str]]) -> None:
        """
        Ingest *examples* (list of {"prompt": …, "response": …}) for
        fine-tuning the assistant's response quality.

        In a production deployment this would submit the examples to the
        underlying model's fine-tuning endpoint.
        """
        if not examples:
            raise ValueError("examples must be a non-empty list.")
        for item in examples:
            if "prompt" not in item or "response" not in item:
                raise ValueError(
                    "Each example must have 'prompt' and 'response' keys."
                )
        self._config.setdefault("fine_tune_queue", []).extend(examples)

    def status(self) -> dict[str, Any]:
        """Return a summary of the assistant's current state."""
        return {
            "mode": self._engine.current.label(),
            "fine_tune_queue_size": len(
                self._config.get("fine_tune_queue", [])
            ),
        }
