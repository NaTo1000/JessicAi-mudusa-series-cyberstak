"""Scalable, fault-tolerant modular compute array."""

from .scalable_array import ScalableComputeArray
from .fault_tolerance import FaultToleranceManager

__all__ = ["ScalableComputeArray", "FaultToleranceManager"]
