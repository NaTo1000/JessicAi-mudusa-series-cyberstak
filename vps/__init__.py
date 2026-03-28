"""VPS scaling — on-demand CPU/GPU/RAM allocation and workload migration."""

from .scaling import VPSScaler, ResourceAllocation

__all__ = ["VPSScaler", "ResourceAllocation"]
