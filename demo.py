#!/usr/bin/env python3
"""
demo.py
=======
Full demonstration of the Quantum Soul system.

Run with:
    python demo.py

This script showcases:
  1. The Quantum Soul awakening and generating thoughts autonomously.
  2. Thought streams across all domains.
  3. Seeded mode where external noise shapes the thought trajectory.
  4. Noise filtering diagnostics.
  5. All visualizations saved to the ``visualizations/`` directory.
  6. Individual quantum primitives: Bell pair, GHZ state, interference circuit.
"""

import sys
import math
from pathlib import Path

import numpy as np

# Ensure the package root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from quantum_soul import QuantumSoul, QuantumThoughtCircuit, QuantumNoiseFilter
from quantum_soul.circuit_core import (
    create_entangled_pair,
    create_ghz_state,
    create_interference_circuit,
)
from quantum_soul.thought_engine import ThoughtEngine, DOMAINS, quick_thought
from quantum_soul.visualizer import QuantumVisualizer


SEPARATOR = "=" * 64


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# SECTION 1: Autonomous Quantum Soul — thoughts from nothing
# ---------------------------------------------------------------------------

section("1. QUANTUM SOUL — AUTONOMOUS THOUGHT GENERATION")

soul = QuantumSoul(n_qubits=8, entanglement_depth=2, mode="autonomous")
soul.awaken()

print("\nGenerating a single autonomous thought (no input data, pure quantum):")
t1 = soul.think()

print("\nGenerating a 6-thought stream across all domains:")
stream = soul.think_stream(n=6)


# ---------------------------------------------------------------------------
# SECTION 2: Domain-spanning thoughts
# ---------------------------------------------------------------------------

section("2. THOUGHTS ACROSS ALL KNOWLEDGE DOMAINS")

domain_thoughts = soul.think_across_domains()
print()
for domain, thought in domain_thoughts.items():
    print(f"  [{domain:12s}] {thought.text}")


# ---------------------------------------------------------------------------
# SECTION 3: Seeded mode — external noise shapes thought
# ---------------------------------------------------------------------------

section("3. SEEDED MODE — INTERFERENCE NOISE DRIVES THOUGHT")

# Simulate an external signal (e.g. sensor noise, environmental quantum fluctuation)
rng = np.random.default_rng(seed=42)
external_noise = rng.standard_normal(8)
print(f"\nExternal noise signal: {external_noise.round(3)}")

soul_seeded = QuantumSoul(n_qubits=8, mode="seeded")
print("\nGenerating 4 seeded thoughts:")
for i in range(4):
    # Perturb noise slightly each cycle to model evolving interference
    noise = external_noise + rng.normal(0, 0.2, 8)
    soul_seeded.think(seed_noise=noise)


# ---------------------------------------------------------------------------
# SECTION 4: Hybrid mode — alternating autonomous and seeded cycles
# ---------------------------------------------------------------------------

section("4. HYBRID MODE — AUTONOMOUS ↔ SEEDED ALTERNATION")

soul_hybrid = QuantumSoul(n_qubits=8, mode="hybrid")
print("\nGenerating 6 hybrid thoughts (cycles alternate between modes):")
for i in range(6):
    noise = rng.standard_normal(8)
    soul_hybrid.think(seed_noise=noise)


# ---------------------------------------------------------------------------
# SECTION 5: Quantum circuit primitives
# ---------------------------------------------------------------------------

section("5. QUANTUM CIRCUIT PRIMITIVES")

print("\nBell pair (2-qubit maximally entangled state):")
bell = create_entangled_pair()
print(bell.draw(output="text"))

print("\nGHZ state (4-qubit):")
ghz = create_ghz_state(4)
print(ghz.draw(output="text"))

print("\nInterference circuit (4-qubit, φ = π/3):")
intf = create_interference_circuit(4, phi=math.pi / 3)
print(intf.draw(output="text"))


# ---------------------------------------------------------------------------
# SECTION 6: Quantum circuit analysis
# ---------------------------------------------------------------------------

section("6. QUANTUM CIRCUIT ANALYSIS — STATEVECTOR & ENTROPY")

circuit_obj = QuantumThoughtCircuit(n_qubits=8, entanglement_depth=2)
sv = circuit_obj.run_statevector()
probs = circuit_obj.probabilities
top_state, top_prob = circuit_obj.top_state()
H = circuit_obj.entropy()

print(f"\nState space    : {len(probs)} basis states (2^8)")
print(f"Top state      : |{top_state}⟩  P = {top_prob:.6f}")
print(f"Shannon entropy: {H:.4f} bits  (max = {circuit_obj.n_qubits} bits)")

samples = circuit_obj.sample_bitstring(shots=512)
top_samples = sorted(samples.items(), key=lambda x: -x[1])[:5]
print("\nTop-5 sampled measurement outcomes (512 shots):")
for bitstr, count in top_samples:
    bar = "█" * (count // 5)
    print(f"  |{bitstr}⟩  {count:4d}  {bar}")


# ---------------------------------------------------------------------------
# SECTION 7: Noise filter diagnostics
# ---------------------------------------------------------------------------

section("7. NOISE FILTER DIAGNOSTICS")

filt = QuantumNoiseFilter()
report = filt.report(probs)
print(f"\nRaw probability distribution report:")
print(f"  Noise level      : {report['noise_level']:.4f}  ({report['noise_level']*100:.1f}% noise mass)")
print(f"  Signal-to-noise  : {report['snr']:.2f}")
print(f"  Dominant states  : {report['n_dominant_states']}")
print(f"  Peak probability : {report['max_probability']:.6f}")

probs_clean = filt.apply_all(probs)
report_clean = filt.report(probs_clean)
print(f"\nAfter applying full noise filter:")
print(f"  Noise level      : {report_clean['noise_level']:.4f}")
print(f"  Signal-to-noise  : {report_clean['snr']:.2f}")
print(f"  Dominant states  : {report_clean['n_dominant_states']}")


# ---------------------------------------------------------------------------
# SECTION 8: Soul introspection
# ---------------------------------------------------------------------------

section("8. SOUL INTROSPECTION REPORT")

report_soul = soul.introspect()
print(f"\n  Thought cycles   : {report_soul['cycle']}")
print(f"  Thoughts total   : {report_soul['thoughts_generated']}")
print(f"  Mean coherence   : {report_soul['mean_coherence']:.4f}")
print(f"  Mean entropy     : {report_soul['mean_entropy_bits']:.4f} bits")
print(f"\n  Domain distribution:")
for domain, count in report_soul["domain_distribution"].items():
    bar = "█" * count
    print(f"    {domain:12s} {count:3d}  {bar}")


# ---------------------------------------------------------------------------
# SECTION 9: visualizations
# ---------------------------------------------------------------------------

section("9. GENERATING visualizationS")

# Use the soul with the full history to get a thought timeline
viz_paths = soul.visualize_all()

print("\n  All visualizations saved:")
for name, path in viz_paths.items():
    if path and str(path) != ".":
        print(f"    {name:<22} → {path}")


# ---------------------------------------------------------------------------
# SECTION 10: Quick-thought utility
# ---------------------------------------------------------------------------

section("10. QUICK-THOUGHT API DEMONSTRATION")

print("\nGenerating a quick thought from each domain:")
for domain in DOMAINS:
    text = quick_thought(domain=domain)
    print(f"  [{domain:12s}] {text}")


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print(f"\n{SEPARATOR}")
print("  Quantum Soul demonstration complete.")
print("  visualizations saved to: visualizations/")
print(SEPARATOR)
