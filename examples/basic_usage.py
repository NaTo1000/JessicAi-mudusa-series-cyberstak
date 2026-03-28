"""
basic_usage.py – Minimal example for loading and running JessicaAi Mudusa.

Run::

    python examples/basic_usage.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from jessicai_mudusa import MudusaConfig, MudusaForCausalLM

# ---------------------------------------------------------------------------
# 1.  Build a small model from scratch (for demonstration / unit tests)
# ---------------------------------------------------------------------------
print("=" * 60)
print("JessicaAi Mudusa – Basic Usage Example")
print("=" * 60)

config = MudusaConfig(
    # Tiny config for quick local testing
    vocab_size=1000,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    num_mesh_channels=3,
    quantum_phase_dim=16,
    synaptic_memory_size=32,
    synaptic_memory_dim=32,
    mesh_fusion_type="gated",
)

print(f"\nConfig: {config.model_type}, hidden_size={config.hidden_size}")

model = MudusaForCausalLM(config)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Parameters: {param_count:,}")

# ---------------------------------------------------------------------------
# 2.  Forward pass with random tokens
# ---------------------------------------------------------------------------
batch_size, seq_len = 2, 16
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
print(f"\nInput shape: {input_ids.shape}")

with torch.no_grad():
    outputs = model(input_ids, return_dict=True)

print(f"Logits shape: {outputs.logits.shape}")
assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size), \
    "Unexpected logits shape!"

# ---------------------------------------------------------------------------
# 3.  KV-cache generation
# ---------------------------------------------------------------------------
print("\n--- Auto-regressive generation (greedy, 8 steps) ---")
prompt_ids = torch.randint(0, config.vocab_size, (1, 4))
generated = model.generate(
    prompt_ids,
    max_new_tokens=8,
    do_sample=False,
    pad_token_id=config.eos_token_id,
)
print(f"Generated token ids: {generated[0].tolist()}")

# ---------------------------------------------------------------------------
# 4.  Load from Hub (requires network and valid Hub weights)
# ---------------------------------------------------------------------------
print("\n--- Hub loading (skipped if offline) ---")
print(
    "To load from the Hub, run:\n"
    "  from jessicai_mudusa import MudusaPipeline\n"
    '  pipe = MudusaPipeline.from_pretrained("NaTo1000/jessicai-mudusa")\n'
    '  print(pipe("Hello, world!"))'
)

print("\n✅  basic_usage.py completed successfully.")
