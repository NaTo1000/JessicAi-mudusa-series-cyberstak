"""
MudusaPipeline – high-level text-generation pipeline for Mudusa.

This pipeline wraps :class:`~jessicai_mudusa.MudusaForCausalLM` and
:class:`~jessicai_mudusa.MudusaTokenizer` with a simple ``__call__`` API
that mirrors :class:`transformers.TextGenerationPipeline`.

Example::

    from jessicai_mudusa import MudusaPipeline

    # Load from the Hub (model weights + tokenizer)
    pipe = MudusaPipeline.from_pretrained("NaTo1000/jessicai-mudusa")

    # Single prompt
    response = pipe("Explain quantum neural networks.")
    print(response)

    # Chat-style (multi-turn)
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    response = pipe.chat(messages)
    print(response)
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from .configuration_mudusa import MudusaConfig
from .modeling_mudusa import MudusaForCausalLM
from .tokenization_mudusa import MudusaTokenizer


class MudusaPipeline:
    """
    High-level generation pipeline for the JessicaAi Mudusa model.

    Args:
        model: A :class:`~jessicai_mudusa.MudusaForCausalLM` instance.
        tokenizer: A :class:`~jessicai_mudusa.MudusaTokenizer` instance.
        device: The torch device to run inference on.  If ``None``, auto-
            selects CUDA > MPS > CPU.
        system_prompt: The system instruction sent at the beginning of every
            chat conversation.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are JessicaAi Mudusa, an advanced quantum-inspired AI assistant "
        "developed by NaTo1000. You are highly cooperative, creative, and "
        "deeply knowledgeable across science, technology, and the arts. "
        "You answer thoughtfully and concisely."
    )

    def __init__(
        self,
        model: MudusaForCausalLM,
        tokenizer: MudusaTokenizer,
        device: Optional[Union[str, torch.device]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **model_kwargs: Any,
    ) -> "MudusaPipeline":
        """Load model and tokenizer from the Hub or a local directory.

        Args:
            pretrained_model_name_or_path: Hub repo id or local path.
            tokenizer_name_or_path: Tokenizer repo id or path.  Defaults to
                ``pretrained_model_name_or_path``.
            device: Inference device.
            torch_dtype: Numeric dtype for model weights (e.g.
                ``torch.float16`` for mixed-precision inference).
            system_prompt: System instruction for the chat template.
            **model_kwargs: Extra kwargs forwarded to
                :meth:`~transformers.PreTrainedModel.from_pretrained`.

        Returns:
            A ready-to-use :class:`MudusaPipeline` instance.
        """
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = MudusaForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_kwargs
        )

        tok_path = tokenizer_name_or_path or pretrained_model_name_or_path
        tokenizer = MudusaTokenizer.from_pretrained(tok_path)

        return cls(model=model, tokenizer=tokenizer, device=device, system_prompt=system_prompt)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """Generate a response to a raw text *prompt*.

        Args:
            prompt: The input text.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more creative).
            top_p: Nucleus sampling probability.
            top_k: Top-k filtering.
            repetition_penalty: Penalty for repeated tokens.
            do_sample: Whether to use sampling (``True``) or greedy decoding.

        Returns:
            The generated text (excluding the prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
        **generation_kwargs: Any,
    ) -> str:
        """Generate a response given a multi-turn chat history.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            system_prompt: Override the default system prompt.
            **generation_kwargs: Forwarded to :meth:`__call__`.

        Returns:
            The assistant's response text.

        Example::

            pipe.chat([
                {"role": "user", "content": "What is the meaning of life?"},
            ])
        """
        sys_prompt = system_prompt or self.system_prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            system_prompt=sys_prompt,
            add_generation_prompt=True,
        )
        return self(prompt, **generation_kwargs)

    # ------------------------------------------------------------------
    # Streaming helper (token-by-token callback)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ):
        """Yield generated tokens one-by-one as they are produced.

        Args:
            prompt: The input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k filtering.
            repetition_penalty: Repetition penalty.

        Yields:
            Decoded string fragments as each token is generated.

        Example::

            for chunk in pipe.stream("Tell me a story."):
                print(chunk, end="", flush=True)
            print()
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        past_key_values = None
        generated = inputs["input_ids"]

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated[0].unique().tolist():
                    logits[0, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                top_k_vals = torch.topk(logits, top_k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < top_k_vals, float("-inf"))

            # Nucleus filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs - torch.softmax(
                    sorted_logits, dim=-1
                ) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(
                    1, sorted_indices, sorted_logits
                )

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            token_str = self.tokenizer.decode(
                next_token[0].tolist(), skip_special_tokens=True
            )
            yield token_str

            eos = self.tokenizer.eos_token_id
            if eos is not None and next_token.item() == eos:
                break

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MudusaPipeline(model={type(self.model).__name__}, "
            f"device={self.device})"
        )
