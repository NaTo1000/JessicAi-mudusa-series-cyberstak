"""
examples/fine_tuning.py – Demonstrates how to fine-tune the Quantum Neural
Brain on a custom dataset using the built-in training pipeline.

Usage::

    python examples/fine_tuning.py

The script uses a tiny synthetic dataset so it can run on CPU without
any downloads.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from quantum_neural_brain import (
    QuantumNeuralBrainConfig,
    QuantumNeuralBrainModel,
    OnlineLearner,
    build_training_pipeline,
    QNBTrainingArguments,
)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

TRAIN_TEXTS = [
    "Quantum neural networks process information through superposition and entanglement.",
    "The brain fires neurons in complex patterns to create conscious experience.",
    "Artificial intelligence learns patterns from data to make predictions.",
    "Quantum computing uses qubits that exist in multiple states simultaneously.",
    "The topological structure of neural meshes enables recursive thought processing.",
    "Interference patterns in quantum circuits give rise to emergent intelligence.",
    "Synaptic connections strengthen through repeated activation – Hebb's rule.",
    "The quad-brain architecture divides cognitive load across four parallel processors.",
    "Fabric mesh layers fold the feature space to reveal hidden semantic structures.",
    "Vertex mechanics in four dimensions route information through topological manifolds.",
]

EVAL_TEXTS = [
    "Quantum cognition models the mind using principles from quantum mechanics.",
    "Entangled qubits can transmit information faster than classical bits.",
]


def demo_batch_training():
    """Run a short batch training loop on synthetic data."""
    print("\n" + "=" * 60)
    print(" Quantum Neural Brain – Batch Fine-Tuning Demo")
    print("=" * 60)

    config = QuantumNeuralBrainConfig(
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=64,
        num_qubits=4,
        quantum_circuit_depth=2,
        num_quantum_blocks=2,
        num_quad_brains=1,
        mesh_fabric_layers=1,
    )

    args = QNBTrainingArguments(
        output_dir="/tmp/qnb_finetune",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=3e-4,
        warmup_steps=20,
        logging_steps=5,
        save_strategy="no",
        evaluation_strategy="epoch",
        fp16=False,  # keep float32 for CPU demo
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = build_training_pipeline(
        train_texts=TRAIN_TEXTS,
        eval_texts=EVAL_TEXTS,
        config=config,
        tokenizer_name="gpt2",
        training_args=args,
    )

    print(f"\nTraining on {len(TRAIN_TEXTS)} examples…")
    train_result = trainer.train()
    print(f"\nTraining complete:")
    print(f"  Final loss     : {train_result.training_loss:.4f}")
    print(f"  Runtime (s)    : {train_result.metrics.get('train_runtime', 0):.1f}")

    # Save
    trainer.save_model("/tmp/qnb_finetuned")
    print("\nFine-tuned model saved to /tmp/qnb_finetuned")


def demo_online_learning():
    """Demonstrate adaptive online learning from new QA pairs."""
    print("\n" + "=" * 60)
    print(" Quantum Neural Brain – Online (Continual) Learning Demo")
    print("=" * 60)

    config = QuantumNeuralBrainConfig(
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=64,
        num_qubits=4,
        quantum_circuit_depth=2,
        num_quantum_blocks=2,
        num_quad_brains=1,
        mesh_fabric_layers=1,
        adaptive_learning_rate=3e-4,
    )

    model = QuantumNeuralBrainModel(config)
    learner = OnlineLearner(model, learning_rate=3e-4, device="cpu")

    qa_pairs = [
        ("What is a qubit?", "A qubit is a quantum bit that can be in superposition."),
        ("What is entanglement?", "Entanglement links two qubits so measuring one instantly affects the other."),
        ("What is the neural mesh?", "The neural mesh is a topological processing layer with quad-brain recursion."),
    ]

    print(f"\nRunning {len(qa_pairs)} online learning steps…")
    for prompt, response in qa_pairs:
        loss = learner.step(prompt, response)
        print(f"  [{prompt[:40]:40s}] loss = {loss:.4f}")

    print("\n✓ Online learning demo completed.")


if __name__ == "__main__":
    demo_batch_training()
    demo_online_learning()
    print("\n✓ All fine-tuning demos completed successfully.")
