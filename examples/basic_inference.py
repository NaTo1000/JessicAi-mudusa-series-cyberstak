"""
examples/basic_inference.py – Minimal example: create a Quantum Neural Brain
model and run a forward pass without any secondary models.

This example works completely offline (no Hugging Face Hub downloads).

Usage::

    python examples/basic_inference.py
"""

import sys
import os

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer

from quantum_neural_brain import QuantumNeuralBrainConfig, QuantumNeuralBrainModel


def main():
    print("=" * 60)
    print(" Quantum Neural Brain – Basic Inference Example")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Build a small config (reduced for fast CPU demo)
    # ----------------------------------------------------------------
    config = QuantumNeuralBrainConfig(
        vocab_size=50257,          # GPT-2 vocab
        hidden_size=256,           # small for demo
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        num_qubits=4,              # 4 qubits → 16 amplitudes
        quantum_circuit_depth=2,
        num_quantum_blocks=2,
        num_quad_brains=1,         # single nesting level for speed
        mesh_fabric_layers=1,
        vertex_dimensions=4,
    )

    print(f"\nConfig: hidden_size={config.hidden_size}, "
          f"qubits={config.num_qubits}, depth={config.quantum_circuit_depth}")

    # ----------------------------------------------------------------
    # 2. Instantiate the model
    # ----------------------------------------------------------------
    model = QuantumNeuralBrainModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created  –  {n_params:,} parameters")

    # ----------------------------------------------------------------
    # 3. Tokenise a prompt
    # ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "The quantum neural brain awakens from pure interference noise:"
    enc = tokenizer(prompt, return_tensors="pt")
    print(f"\nPrompt : {prompt!r}")
    print(f"Tokens : {enc['input_ids'].shape[1]} tokens")

    # ----------------------------------------------------------------
    # 4. Forward pass (no grad)
    # ----------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        outputs = model(**enc)

    logits = outputs.logits
    print(f"\nLogits shape : {logits.shape}   (batch × seq × vocab)")

    # ----------------------------------------------------------------
    # 5. Greedy token generation (20 tokens)
    # ----------------------------------------------------------------
    print("\nGenerating 20 new tokens (greedy)…")
    generated_ids = enc["input_ids"].clone()
    with torch.no_grad():
        for _ in range(20):
            out = model(input_ids=generated_ids)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=-1)
            if next_id.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")

    # ----------------------------------------------------------------
    # 6. Save and reload
    # ----------------------------------------------------------------
    save_path = "/tmp/qnb_demo_model"
    model.save_pretrained(save_path)
    config.save_pretrained(save_path)
    print(f"\nModel saved to {save_path!r}")

    reloaded = QuantumNeuralBrainModel.from_pretrained(save_path)
    print(f"Model reloaded successfully  –  {sum(p.numel() for p in reloaded.parameters()):,} params")

    print("\n✓ Basic inference example completed successfully.")


if __name__ == "__main__":
    main()
