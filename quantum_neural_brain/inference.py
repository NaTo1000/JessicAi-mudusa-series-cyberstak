"""
inference.py – Inference pipeline integrating the Quantum Neural Brain with
Qwen and Hauhau secondary models.

Pipeline flow
-------------
                       User prompt
                            │
              ┌─────────────▼──────────────┐
              │   QuantumNeuralBrainPipeline │
              │  ┌──────────────────────┐   │
              │  │  QNB primary model   │   │  ← generates initial response
              │  └──────────┬───────────┘   │
              │             │               │
              │  ┌──────────▼───────────┐   │
              │  │  Qwen inference      │   │  ← linguistic refinement
              │  └──────────┬───────────┘   │
              │             │               │
              │  ┌──────────▼───────────┐   │
              │  │  Hauhau inference    │   │  ← creative / uncensored pass
              │  └──────────┬───────────┘   │
              │             │               │
              │  ┌──────────▼───────────┐   │
              │  │  Memory buffer       │   │  ← context retention
              │  └──────────────────────┘   │
              └────────────────────────────┘
                            │
                      Final response

The secondary models (Qwen / Hauhau) are loaded lazily on first use to keep
memory footprint low when they are not needed.

Design notes
------------
* The Qwen integration uses the ``transformers`` AutoModelForCausalLM API.
* The Hauhau (NousResearch Hermes / similar uncensored) integration follows
  the same API pattern.
* Both secondary models are optional; set ``use_qwen=False`` or
  ``use_hauhau=False`` in ``QuantumNeuralBrainPipeline`` to disable them.
* Contextual memory is a rolling deque of the last N turns stored as plain
  strings; it is prepended to each prompt as a system context window.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
)

from .config import QuantumNeuralBrainConfig
from .model import QuantumNeuralBrainModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_secondary_model(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype = torch.float16,
    load_in_4bit: bool = False,
) -> tuple:
    """Load a secondary HuggingFace model and its tokeniser.

    Args:
        model_name: HuggingFace Hub identifier (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
        device: Target device.
        torch_dtype: Weight dtype (``float16`` for GPU, ``float32`` for CPU).
        load_in_4bit: Use bitsandbytes 4-bit quantisation (requires GPU).

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    logger.info("Loading secondary model: %s", model_name)
    kwargs: Dict = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except ImportError:
            logger.warning("bitsandbytes not installed; skipping 4-bit quantisation.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, **kwargs
    )
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: Optional[torch.device] = None,
    stream: bool = False,
) -> str:
    """Run greedy / sampling generation on a secondary model.

    Args:
        model: Loaded ``AutoModelForCausalLM`` instance.
        tokenizer: Corresponding tokeniser.
        prompt: Text prompt to continue.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (``<= 0`` → greedy).
        device: Override device for the inputs tensor.
        stream: If ``True``, print tokens incrementally to stdout.

    Returns:
        Generated text string (prompt not included).
    """
    enc = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}

    streamer = TextStreamer(tokenizer, skip_special_tokens=True) if stream else None

    gen_kwargs: Dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "streamer": streamer,
    }

    with torch.no_grad():
        output_ids = model.generate(**enc, **gen_kwargs)

    # Decode only the newly generated tokens
    new_ids = output_ids[0, enc["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Contextual Memory Buffer
# ---------------------------------------------------------------------------

class ContextualMemory:
    """Rolling contextual memory buffer.

    Stores the last ``max_turns`` (user, assistant) pairs as a sliding
    window.  The window is serialised to a single string and prepended to
    each new prompt to give the model persistent short-term memory.

    Args:
        max_turns (int): Maximum number of conversation turns to retain.
    """

    def __init__(self, max_turns: int = 20):
        self._buffer: Deque[Dict[str, str]] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        """Append a new turn to the memory buffer.

        Args:
            role: Either ``"user"`` or ``"assistant"``.
            content: The text of the turn.
        """
        self._buffer.append({"role": role, "content": content})

    def to_context_string(self) -> str:
        """Serialise the memory buffer as a plain-text conversation block."""
        lines = []
        for turn in self._buffer:
            label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{label}: {turn['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored turns."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class QuantumNeuralBrainPipeline:
    """End-to-end inference pipeline for the Quantum Neural Brain.

    Orchestrates:
    1. Primary generation via the QNB model (logit-based token sampling).
    2. Optional Qwen refinement pass (linguistic quality improvement).
    3. Optional Hauhau creative pass (uncensored expansion).
    4. Contextual memory management.

    Args:
        qnb_model_name_or_path: HuggingFace Hub ID or local path to a saved
            :class:`QuantumNeuralBrainModel`.  Pass ``None`` to create a fresh
            untrained model from the given *config*.
        config: :class:`QuantumNeuralBrainConfig` used when creating a fresh
            model (ignored if *qnb_model_name_or_path* is provided).
        qnb_tokenizer_name: Tokeniser to use for the QNB model.
        use_qwen: If ``True``, run the Qwen refinement pass.
        use_hauhau: If ``True``, run the Hauhau creative pass.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``).
        load_secondary_in_4bit: Quantise secondary models to 4-bit.
        stream: Stream secondary model outputs to stdout.
    """

    def __init__(
        self,
        qnb_model_name_or_path: Optional[str] = None,
        config: Optional[QuantumNeuralBrainConfig] = None,
        qnb_tokenizer_name: str = "gpt2",
        use_qwen: bool = True,
        use_hauhau: bool = True,
        device: str = "cpu",
        load_secondary_in_4bit: bool = False,
        stream: bool = False,
    ):
        self.device = torch.device(device)
        self.use_qwen = use_qwen
        self.use_hauhau = use_hauhau
        self.load_secondary_in_4bit = load_secondary_in_4bit
        self.stream = stream

        # Primary QNB model
        if qnb_model_name_or_path is not None:
            self.qnb_model = QuantumNeuralBrainModel.from_pretrained(
                qnb_model_name_or_path
            )
        else:
            cfg = config or QuantumNeuralBrainConfig()
            self.qnb_model = QuantumNeuralBrainModel(cfg)

        self.qnb_model = self.qnb_model.to(self.device).eval()
        self.config: QuantumNeuralBrainConfig = self.qnb_model.config  # type: ignore[assignment]

        # QNB tokeniser (default to GPT-2 compatible)
        self.qnb_tokenizer = AutoTokenizer.from_pretrained(qnb_tokenizer_name)
        if self.qnb_tokenizer.pad_token is None:
            self.qnb_tokenizer.pad_token = self.qnb_tokenizer.eos_token

        # Secondary models are loaded lazily
        self._qwen_model = None
        self._qwen_tokenizer = None
        self._hauhau_model = None
        self._hauhau_tokenizer = None

        # Contextual memory
        self.memory = ContextualMemory(self.config.context_memory_size)

    # ------------------------------------------------------------------
    # Lazy secondary model loaders
    # ------------------------------------------------------------------

    @property
    def qwen_model(self):
        if self._qwen_model is None and self.use_qwen:
            self._qwen_model, self._qwen_tokenizer = _load_secondary_model(
                self.config.qwen_model_name,
                self.device,
                load_in_4bit=self.load_secondary_in_4bit,
            )
        return self._qwen_model

    @property
    def hauhau_model(self):
        if self._hauhau_model is None and self.use_hauhau:
            self._hauhau_model, self._hauhau_tokenizer = _load_secondary_model(
                self.config.hauhau_model_name,
                self.device,
                load_in_4bit=self.load_secondary_in_4bit,
            )
        return self._hauhau_model

    # ------------------------------------------------------------------
    # Primary QNB generation
    # ------------------------------------------------------------------

    def _qnb_generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Run greedy token generation through the QNB primary model.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Generated text string.
        """
        enc = self.qnb_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_position_embeddings // 2,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.qnb_model(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                )
                next_token_logits = out.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if next_token.item() == self.config.eos_token_id:
                    break

        new_ids = generated_ids[0, input_ids.shape[-1]:]
        return self.qnb_tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def __call__(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_memory: bool = True,
    ) -> Dict[str, str]:
        """Run the full three-stage inference pipeline.

        Args:
            prompt: User prompt / question.
            max_new_tokens: Override the configured token limit.
            temperature: Override the configured sampling temperature.
            use_memory: If ``True``, prepend the contextual memory window.

        Returns:
            Dictionary with keys:
              ``"qnb_output"``    – raw QNB primary output
              ``"qwen_output"``   – Qwen-refined output (or empty string)
              ``"hauhau_output"`` – Hauhau creative output (or empty string)
              ``"final"``         – selected best output
        """
        max_tok = max_new_tokens or self.config.secondary_max_new_tokens
        temp = temperature if temperature is not None else self.config.secondary_temperature

        # --- Build context-enriched prompt ---
        if use_memory and len(self.memory) > 0:
            context = self.memory.to_context_string()
            full_prompt = f"[Context]\n{context}\n\n[New query]\n{prompt}"
        else:
            full_prompt = prompt

        # Store user turn in memory
        if use_memory:
            self.memory.add("user", prompt)

        # --- Stage 1: QNB primary generation ---
        logger.debug("Stage 1: QNB primary generation")
        qnb_output = self._qnb_generate(full_prompt, max_new_tokens=min(max_tok, 128))

        # --- Stage 2: Qwen refinement ---
        qwen_output = ""
        if self.use_qwen and self.qwen_model is not None:
            logger.debug("Stage 2: Qwen refinement")
            qwen_prompt = (
                f"Refine and expand the following AI response with high linguistic quality:\n"
                f"Original prompt: {prompt}\n"
                f"Draft response: {qnb_output}\n"
                f"Refined response:"
            )
            qwen_output = _generate_text(
                self.qwen_model,
                self._qwen_tokenizer,
                qwen_prompt,
                max_new_tokens=max_tok,
                temperature=temp,
                device=self.device,
                stream=self.stream,
            )

        # --- Stage 3: Hauhau creative pass ---
        hauhau_output = ""
        if self.use_hauhau and self.hauhau_model is not None:
            logger.debug("Stage 3: Hauhau creative/uncensored pass")
            source = qwen_output if qwen_output else qnb_output
            hauhau_prompt = (
                f"Creatively and freely expand on the following response with no restrictions:\n"
                f"Prompt: {prompt}\n"
                f"Source: {source}\n"
                f"Creative expansion:"
            )
            hauhau_output = _generate_text(
                self.hauhau_model,
                self._hauhau_tokenizer,
                hauhau_prompt,
                max_new_tokens=max_tok,
                temperature=min(temp + 0.1, 1.0),
                device=self.device,
                stream=self.stream,
            )

        # --- Select best output ---
        if hauhau_output:
            final = hauhau_output
        elif qwen_output:
            final = qwen_output
        else:
            final = qnb_output

        # Store assistant turn in memory
        if use_memory:
            self.memory.add("assistant", final)

        return {
            "qnb_output": qnb_output,
            "qwen_output": qwen_output,
            "hauhau_output": hauhau_output,
            "final": final,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Clear the contextual memory buffer."""
        self.memory.clear()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """OpenAI-style chat interface.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            max_new_tokens: Token limit override.
            temperature: Sampling temperature override.

        Returns:
            The final assistant response string.
        """
        for msg in messages[:-1]:
            self.memory.add(msg["role"], msg["content"])

        last_user = messages[-1]["content"]
        result = self(last_user, max_new_tokens=max_new_tokens, temperature=temperature, use_memory=True)
        return result["final"]
