"""
Quantum Quad-Brain: NVMe + DRAM + Raspberry Pi CM4 Compute Array
=================================================================
An optimal, scalable compute array integrating ultra-fast NVMe storage,
high-bandwidth DRAM, and Raspberry Pi CM4 distributed compute modules,
enhanced with quantum inference capabilities.
"""

from .array.scalable_array import ScalableComputeArray
from .quantum_core.quad_brain import QuantumQuadBrain

__all__ = ["ScalableComputeArray", "QuantumQuadBrain"]
__version__ = "1.0.0"
