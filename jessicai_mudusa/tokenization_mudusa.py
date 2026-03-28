"""
MudusaTokenizer – thin wrapper around Qwen2Tokenizer for the Mudusa model.

Using Qwen2's tokenizer (tiktoken-based, 151 936-token vocabulary) gives
Mudusa access to the same sub-word vocabulary used by the Qwen backbone, so
pre-trained embeddings can be reused directly.

Example::

    from jessicai_mudusa import MudusaTokenizer

    tok = MudusaTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    ids = tok("Hello, Mudusa!", return_tensors="pt")
    print(ids)
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Special tokens used by the Mudusa chat template.
_SYSTEM_TOKEN = "<|system|>"
_USER_TOKEN = "<|user|>"
_ASSISTANT_TOKEN = "<|assistant|>"
_END_TOKEN = "<|end|>"


class MudusaTokenizer:
    """
    A lightweight tokenizer wrapper for the Mudusa model.

    Internally delegates all tokenisation work to a
    :class:`~transformers.PreTrainedTokenizerBase` (default: Qwen2Tokenizer).
    In addition it exposes a :meth:`apply_chat_template` helper that formats
    multi-turn conversations in Mudusa's preferred markup.

    Attributes:
        _tokenizer: The underlying HuggingFace tokenizer instance.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "Qwen/Qwen2-1.5B",
        **kwargs: Any,
    ) -> "MudusaTokenizer":
        """Load tokenizer from HuggingFace Hub or a local directory.

        Args:
            pretrained_model_name_or_path: Hub repo id or local path.
            **kwargs: Forwarded to :func:`transformers.AutoTokenizer.from_pretrained`.

        Returns:
            A :class:`MudusaTokenizer` instance.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **kwargs,
        )
        return cls(tokenizer)

    # ------------------------------------------------------------------
    # Core tokenisation interface (delegates to inner tokenizer)
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any):
        """Tokenise text – identical call signature to HuggingFace tokenizers."""
        return self._tokenizer(*args, **kwargs)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return self._tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, sequences: list[list[int]], **kwargs: Any) -> list[str]:
        return self._tokenizer.batch_decode(sequences, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs: Any):
        """Persist the tokenizer to *save_directory*."""
        return self._tokenizer.save_pretrained(save_directory, **kwargs)

    # ------------------------------------------------------------------
    # Mudusa chat-template helper
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = (
            "You are JessicaAi Mudusa, an advanced quantum-inspired AI "
            "assistant. You are highly cooperative, creative, and thoughtful."
        ),
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a list of chat messages into Mudusa's prompt format.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts where
                *role* is one of ``"user"`` or ``"assistant"``.
            system_prompt: Optional system instruction prepended to the prompt.
            add_generation_prompt: If ``True``, append the assistant header so
                the model knows to continue generation.

        Returns:
            A formatted prompt string ready to feed into the tokenizer.

        Example::

            messages = [{"role": "user", "content": "What is quantum AI?"}]
            prompt = tok.apply_chat_template(messages)
        """
        parts: list[str] = []

        if system_prompt:
            parts.append(f"{_SYSTEM_TOKEN}\n{system_prompt}{_END_TOKEN}\n")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"{_USER_TOKEN}\n{content}{_END_TOKEN}\n")
            elif role == "assistant":
                parts.append(f"{_ASSISTANT_TOKEN}\n{content}{_END_TOKEN}\n")

        if add_generation_prompt:
            parts.append(f"{_ASSISTANT_TOKEN}\n")

        return "".join(parts)

    # ------------------------------------------------------------------
    # Property pass-throughs
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self._tokenizer.bos_token_id

    def __repr__(self) -> str:  # pragma: no cover
        return f"MudusaTokenizer(vocab_size={self.vocab_size})"
