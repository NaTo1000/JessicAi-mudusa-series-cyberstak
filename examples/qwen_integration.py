"""
examples/qwen_integration.py – Demonstrates how to run the full pipeline
with Qwen and Hauhau secondary models.

Prerequisites
-------------
* A GPU with ≥24 GB VRAM (or use ``load_secondary_in_4bit=True`` for 8 GB).
* Install the requirements:  ``pip install -r requirements.txt``
* Hugging Face account + ``huggingface-cli login`` for gated models.

Usage::

    # Full pipeline (downloads Qwen + Hauhau models ~15 GB)
    python examples/qwen_integration.py

    # Offline demo – skip secondary models
    python examples/qwen_integration.py --offline
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_neural_brain import QuantumNeuralBrainConfig, QuantumNeuralBrainPipeline


def run_offline_demo():
    """Run a demo that only uses the local QNB model (no downloads)."""
    print("\n[OFFLINE MODE] Using QNB primary model only.\n")

    config = QuantumNeuralBrainConfig(
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        num_qubits=4,
        quantum_circuit_depth=2,
        num_quantum_blocks=2,
        num_quad_brains=1,
        mesh_fabric_layers=1,
    )

    pipeline = QuantumNeuralBrainPipeline(
        config=config,
        use_qwen=False,
        use_hauhau=False,
        device="cpu",
    )

    prompts = [
        "Explain quantum entanglement in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What is the meaning of consciousness?",
    ]

    for prompt in prompts:
        print(f"Prompt  : {prompt}")
        result = pipeline(prompt, max_new_tokens=30)
        print(f"QNB out : {result['qnb_output']!r}")
        print()


def run_full_pipeline(device: str, load_in_4bit: bool):
    """Run the full three-stage pipeline with Qwen and Hauhau models."""
    print("\n[FULL PIPELINE] Loading Qwen + Hauhau secondary models...\n")
    print("Note: Initial download may take several minutes.\n")

    config = QuantumNeuralBrainConfig(
        vocab_size=50257,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        num_qubits=6,
        quantum_circuit_depth=3,
        num_quantum_blocks=4,
        num_quad_brains=1,
        mesh_fabric_layers=2,
        # Secondary model config
        qwen_model_name="Qwen/Qwen2.5-7B-Instruct",
        hauhau_model_name="NousResearch/Hermes-3-Llama-3.1-8B",
        secondary_max_new_tokens=256,
        secondary_temperature=0.7,
        context_memory_size=10,
    )

    pipeline = QuantumNeuralBrainPipeline(
        config=config,
        use_qwen=True,
        use_hauhau=True,
        device=device,
        load_secondary_in_4bit=load_in_4bit,
        stream=True,
    )

    # Multi-turn conversation demo
    print("=" * 60)
    print(" Multi-turn Conversation Demo")
    print("=" * 60)

    conversation = [
        "What are the fundamental principles of quantum computing?",
        "How does this relate to neural networks and AI?",
        "Can you give a creative metaphor for quantum superposition?",
    ]

    for turn, user_input in enumerate(conversation, 1):
        print(f"\n[Turn {turn}] User: {user_input}")
        result = pipeline(user_input, use_memory=True)
        print(f"\n[Turn {turn}] QNB primary:\n  {result['qnb_output'][:200]}")
        if result["qwen_output"]:
            print(f"[Turn {turn}] Qwen refined:\n  {result['qwen_output'][:200]}")
        if result["hauhau_output"]:
            print(f"[Turn {turn}] Hauhau creative:\n  {result['hauhau_output'][:200]}")
        print(f"\n[Turn {turn}] Final response:\n  {result['final'][:400]}")
        print("-" * 60)

    print(f"\nMemory buffer has {len(pipeline.memory)} stored turns.")
    print("\n✓ Full pipeline demo completed.")


def main():
    parser = argparse.ArgumentParser(description="QNB + Qwen + Hauhau pipeline demo")
    parser.add_argument("--offline", action="store_true", help="Skip secondary models")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--4bit", dest="load_4bit", action="store_true", help="Load secondary models in 4-bit")
    args = parser.parse_args()

    if args.offline:
        run_offline_demo()
    else:
        run_full_pipeline(args.device, args.load_4bit)


if __name__ == "__main__":
    main()
