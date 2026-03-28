"""
Nonganon Tesseract Topological Octagon Architecture
====================================================

A triple-clustering thought processor powering the
Pex-Chained Twinbrain Algorithmic Quantum-Inference
Decision-Making System.

Package layout
--------------
tesseract_octagon  – 8-layer topological tesseract framework
triple_cluster     – triple-clustering engine
twinbrain          – Twinbrain quad-layer algorithm
quantum_inference  – probabilistic quantum-inference engine
pex_chain          – Pex-Chained distributed decision logic
security           – SHA-256 / GPG vault layer
thought_processor  – top-level orchestrator (main entry-point)
"""

from .tesseract_octagon import TesseractOctagon
from .triple_cluster import TripleCluster
from .twinbrain import TwinbrainAlgorithm
from .quantum_inference import QuantumInferenceEngine
from .pex_chain import PexChain
from .security import SecurityVault
from .thought_processor import ThoughtProcessor

__all__ = [
    "TesseractOctagon",
    "TripleCluster",
    "TwinbrainAlgorithm",
    "QuantumInferenceEngine",
    "PexChain",
    "SecurityVault",
    "ThoughtProcessor",
]

__version__ = "1.0.0"
