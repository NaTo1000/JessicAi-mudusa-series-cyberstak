"""
examples/push_to_hub.py – Demonstrates how to publish the Quantum Neural Brain
model to the Hugging Face Hub.

Prerequisites
-------------
1. Create a free account at https://huggingface.co
2. Run:  ``huggingface-cli login``  (or set the ``HF_TOKEN`` env variable)
3. Optionally create a model repo first at
   https://huggingface.co/new

Usage::

    # Push a fresh randomly-initialised model
    python examples/push_to_hub.py --repo YOUR_USERNAME/quantum-neural-brain

    # Push a fine-tuned checkpoint
    python examples/push_to_hub.py --repo YOUR_USERNAME/quantum-neural-brain \
        --checkpoint /tmp/qnb_finetuned
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_neural_brain import QuantumNeuralBrainConfig, QuantumNeuralBrainModel


def main():
    parser = argparse.ArgumentParser(description="Push QNB model to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="Hub repo ID, e.g. username/quantum-neural-brain",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a saved checkpoint (optional; creates fresh model if omitted)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    args = parser.parse_args()

    # Load or create model
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint!r}…")
        model = QuantumNeuralBrainModel.from_pretrained(args.checkpoint)
    else:
        print("Creating fresh QuantumNeuralBrainModel…")
        config = QuantumNeuralBrainConfig()
        model = QuantumNeuralBrainModel(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters.")

    # Push to Hub
    print(f"\nPushing to Hub repo: {args.repo!r} (private={args.private})…")
    model.push_to_hub(args.repo, private=args.private)
    model.config.push_to_hub(args.repo, private=args.private)

    print(f"\n✓ Model published at: https://huggingface.co/{args.repo}")
    print(
        "\nTo load it later:\n"
        f"    from quantum_neural_brain import QuantumNeuralBrainModel\n"
        f"    model = QuantumNeuralBrainModel.from_pretrained('{args.repo}')"
    )


if __name__ == "__main__":
    main()
