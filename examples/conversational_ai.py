"""
conversational_ai.py – Multi-turn chatbot example for JessicaAi Mudusa.

This example demonstrates:
- Loading from the Hugging Face Hub
- Applying the Mudusa chat template
- Running a simulated multi-turn conversation

Run::

    python examples/conversational_ai.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from jessicai_mudusa import MudusaTokenizer

# ---------------------------------------------------------------------------
# Demonstrate the chat-template formatter (no GPU required)
# ---------------------------------------------------------------------------
print("=" * 60)
print("JessicaAi Mudusa – Conversational AI Example")
print("=" * 60)

# Simulate a tokenizer wrapper for demonstration (no real weights needed)
class _MockInnerTok:
    vocab_size = 151936
    eos_token_id = 151645
    bos_token_id = 151643
    pad_token_id = None

    def __call__(self, *a, **kw):
        return {}

    def encode(self, text, **kw):
        return list(range(min(len(text), 20)))

    def decode(self, ids, **kw):
        return f"<decoded {len(ids)} tokens>"

    def batch_decode(self, seqs, **kw):
        return [self.decode(s) for s in seqs]

    def save_pretrained(self, *a, **kw):
        pass


tok = MudusaTokenizer(_MockInnerTok())

# ---------------------------------------------------------------------------
# Multi-turn conversation demo
# ---------------------------------------------------------------------------
conversation: list[dict] = []

turns = [
    "Hi! Who are you?",
    "Can you explain quantum neural networks in simple terms?",
    "That's fascinating. How does your synaptic memory work?",
]

print("\nSimulated multi-turn conversation (template formatting demo):\n")

for user_msg in turns:
    conversation.append({"role": "user", "content": user_msg})

    prompt = tok.apply_chat_template(
        conversation,
        system_prompt=(
            "You are JessicaAi Mudusa, an advanced quantum-inspired AI assistant."
        ),
        add_generation_prompt=True,
    )

    print(f"User: {user_msg}")
    print(f"[Formatted prompt ({len(prompt)} chars)]")
    print()

    # In a real scenario:
    # response = pipe.chat(conversation)
    # conversation.append({"role": "assistant", "content": response})
    # print(f"Mudusa: {response}\n")

    # Simulated response for demo purposes
    simulated_response = (
        f"[Mudusa's response to: '{user_msg[:40]}…']"
    )
    conversation.append({"role": "assistant", "content": simulated_response})
    print(f"Mudusa: {simulated_response}\n")

print("\nFull prompt for last turn:")
print("-" * 50)
print(prompt[:800], "…" if len(prompt) > 800 else "")
print("-" * 50)

print("\n✅  conversational_ai.py completed successfully.")
print(
    "\nTo run a real conversation:\n"
    "  from jessicai_mudusa import MudusaPipeline\n"
    '  pipe = MudusaPipeline.from_pretrained("NaTo1000/jessicai-mudusa")\n'
    '  pipe.chat([{"role": "user", "content": "Hello!"}])'
)
