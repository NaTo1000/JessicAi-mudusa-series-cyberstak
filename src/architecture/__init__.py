"""
JessicAI AGI Architecture Package

Exports the core subsystems of the 1-billion-superconducting-layer AGI
with trillion-layer topological tesseract configuration.
"""

from .tesseract import Tesseract, TesseractLayer, TesseractConfig
from .superconductor import SuperconductorLayer, SuperconductorArray
from .vault import TripleCodedVault, VaultConfig
from .clustering import ClusterRegulator, SubCluster, ClusterNode
from .agi_processor import AGIProcessor, InferenceResult

__all__ = [
    "Tesseract",
    "TesseractLayer",
    "TesseractConfig",
    "SuperconductorLayer",
    "SuperconductorArray",
    "TripleCodedVault",
    "VaultConfig",
    "ClusterRegulator",
    "SubCluster",
    "ClusterNode",
    "AGIProcessor",
    "InferenceResult",
]
