"""AicodeX mode definitions and engine."""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable


class Mode(Enum):
    """Supported AicodeX operational modes."""

    CODING = auto()
    DESIGN = auto()
    DEBUG = auto()

    @classmethod
    def from_string(cls, value: str) -> "Mode":
        """Parse a mode from a human-readable string."""
        mapping = {
            "coding": cls.CODING,
            "design": cls.DESIGN,
            "debug": cls.DEBUG,
        }
        key = value.strip().lower()
        if key not in mapping:
            raise ValueError(
                f"Unknown mode '{value}'. Valid modes: {list(mapping.keys())}"
            )
        return mapping[key]

    def label(self) -> str:
        return self.name.capitalize()


class ModeEngine:
    """Manages mode transitions and dispatches work to the correct handler."""

    def __init__(self) -> None:
        self._current: Mode = Mode.CODING
        self._handlers: dict[Mode, Callable[[str], str]] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current(self) -> Mode:
        return self._current

    def switch(self, mode: Mode | str) -> None:
        """Switch to *mode* (accepts a Mode enum or string name)."""
        if isinstance(mode, str):
            mode = Mode.from_string(mode)
        self._current = mode

    def register_handler(self, mode: Mode, handler: Callable[[str], str]) -> None:
        """Register a callable that handles prompts for *mode*."""
        self._handlers[mode] = handler

    def process(self, prompt: str) -> str:
        """Route *prompt* to the handler registered for the current mode."""
        handler = self._handlers.get(self._current)
        if handler is None:
            raise RuntimeError(
                f"No handler registered for mode '{self._current.label()}'."
            )
        return handler(prompt)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        self._handlers[Mode.CODING] = self._default_coding
        self._handlers[Mode.DESIGN] = self._default_design
        self._handlers[Mode.DEBUG] = self._default_debug

    @staticmethod
    def _default_coding(prompt: str) -> str:
        return f"[AicodeX / Coding] Processing: {prompt}"

    @staticmethod
    def _default_design(prompt: str) -> str:
        return f"[AicodeX / Design] Generating design for: {prompt}"

    @staticmethod
    def _default_debug(prompt: str) -> str:
        return f"[AicodeX / Debug] Analysing for bugs: {prompt}"
