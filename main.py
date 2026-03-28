#!/usr/bin/env python3
"""
main.py — Quantum Quad-Brain System Demo
=========================================
Runs the full system, prints status at each milestone, and saves visualizations.

Usage:
    python main.py [--cycles N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Ensure package is importable when running directly from repo root
sys.path.insert(0, os.path.dirname(__file__))

from quantum_quad_brain import QuantumQuadBrain
from quantum_quad_brain.visualization import save_all


def banner(msg: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {msg}")
    print("═" * width)


def main(cycles: int = 20, output_dir: str = "output") -> None:
    banner("Quantum Quad-Brain System — Initialising")

    brain = QuantumQuadBrain(
        state_dims=64,
        topo_vertices=16,
        mesh_rows=8,
        mesh_cols=8,
        num_fabric_layers=4,
    )

    print(brain)
    print("\nInitial status:")
    print(json.dumps(brain.status(), indent=2))

    # -----------------------------------------------------------------------
    # Run inference cycles
    # -----------------------------------------------------------------------
    banner(f"Running {cycles} inference cycles")
    rng = np.random.default_rng(2025)

    for i in range(cycles):
        # Alternate between random stimuli and structured inputs
        if i % 3 == 0:
            inp = rng.standard_normal(64)
        elif i % 3 == 1:
            inp = np.sin(np.linspace(0, 2 * np.pi * (i + 1), 64))
        else:
            inp = rng.standard_normal(64) * 0.3  # low-energy / noisy

        result = brain.think(input_data=inp, noise_sigma=0.03)
        print(
            f"  Cycle {result.cycle:3d} | "
            f"Decision: {result.decision:3d} | "
            f"Prob: {result.probability:.4f} | "
            f"Coherence: {result.coherence:.4f} | "
            f"MeshCoh: {result.mesh_coherence:.4f}"
        )

    # -----------------------------------------------------------------------
    # Demonstrate infinite scalability: expand topology
    # -----------------------------------------------------------------------
    banner("Expanding 4D Topology (infinite scalability demo)")
    before = len(brain.topology.vertices)
    brain.expand_topology(new_vertices=8)
    after = len(brain.topology.vertices)
    print(f"  Vertices: {before} → {after}")

    # Run a few more cycles with the expanded topology
    for _ in range(5):
        result = brain.think()
        print(
            f"  Cycle {result.cycle:3d} | "
            f"Decision: {result.decision:3d} | "
            f"Coherence: {result.coherence:.4f}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    banner("Inference Summary")
    summary = brain.pipeline.summary()
    print(json.dumps(summary, indent=2))

    # -----------------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------------
    banner("Final System Status")
    print(json.dumps(brain.status(), indent=2))

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------
    banner(f"Saving visualizations to '{output_dir}/'")
    saved = save_all(brain, output_dir=output_dir)
    for path in saved:
        print(f"  ✓ {path}")

    banner("Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Quad-Brain System Demo")
    parser.add_argument("--cycles", type=int, default=20, help="Number of inference cycles")
    parser.add_argument("--output-dir", type=str, default="output", help="Visualization output dir")
    args = parser.parse_args()
    main(cycles=args.cycles, output_dir=args.output_dir)
