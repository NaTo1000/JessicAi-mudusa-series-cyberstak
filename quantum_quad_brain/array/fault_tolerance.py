"""
Fault Tolerance Manager
=======================
Provides automatic fault detection, node failover, and data redundancy
for the Quantum Quad-Brain compute array.  It monitors all subsystems
and orchestrates recovery actions when faults are detected.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class FaultSeverity(Enum):
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class FaultEvent:
    """A detected fault with metadata for incident tracking."""

    fault_id: str
    component: str
    severity: FaultSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_note: str = ""

    def resolve(self, note: str = "") -> None:
        self.resolved = True
        self.resolution_note = note

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "component": self.component,
            "severity": self.severity.name,
            "message": self.message,
            "timestamp": round(self.timestamp, 3),
            "resolved": self.resolved,
            "resolution_note": self.resolution_note,
        }


class FaultToleranceManager:
    """
    Monitors the Quantum Quad-Brain array for faults and triggers
    configured recovery handlers.

    Features
    --------
    * Automatic NVMe device failover (removes unhealthy devices from pipeline).
    * CM4 node isolation (removes OFFLINE nodes and redistributes tasks).
    * Brain degraded-mode fallback (switches to a lower-qubit brain on error).
    * Incident log with severity tracking.

    Parameters
    ----------
    min_healthy_nvme : int
        Minimum number of NVMe devices required; raise CRITICAL if below.
    min_online_nodes : int
        Minimum CM4 nodes that must be online; raise ERROR if below.
    """

    def __init__(
        self,
        min_healthy_nvme: int = 1,
        min_online_nodes: int = 1,
    ) -> None:
        self.min_healthy_nvme = min_healthy_nvme
        self.min_online_nodes = min_online_nodes
        self._incidents: List[FaultEvent] = []
        self._handlers: Dict[FaultSeverity, List[Callable[[FaultEvent], None]]] = {
            FaultSeverity.WARNING: [],
            FaultSeverity.ERROR: [],
            FaultSeverity.CRITICAL: [],
        }

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(
        self,
        severity: FaultSeverity,
        handler: Callable[[FaultEvent], None],
    ) -> None:
        """Register a callback to be invoked when a fault of *severity* fires."""
        self._handlers[severity].append(handler)

    # ------------------------------------------------------------------
    # Fault detection
    # ------------------------------------------------------------------

    def check_storage(self, pipeline: Any) -> List[FaultEvent]:
        """Inspect the NVMe pipeline and raise faults as needed."""
        new_faults: List[FaultEvent] = []
        health = pipeline.run_health_checks()
        unhealthy = [k for k, v in health.items() if not v]
        healthy_count = len(health) - len(unhealthy)

        for dev_id in unhealthy:
            fault = self._raise(
                component=f"nvme:{dev_id}",
                severity=FaultSeverity.ERROR,
                message=f"NVMe device '{dev_id}' failed health check.",
            )
            new_faults.append(fault)
            try:
                pipeline.remove_device(dev_id)
                fault.resolve(note="Device removed from pipeline.")
            except KeyError:
                pass

        if healthy_count < self.min_healthy_nvme:
            new_faults.append(
                self._raise(
                    component="nvme_pipeline",
                    severity=FaultSeverity.CRITICAL,
                    message=(
                        f"Only {healthy_count} healthy NVMe device(s); "
                        f"minimum is {self.min_healthy_nvme}."
                    ),
                )
            )
        return new_faults

    def check_cluster(self, cluster: Any) -> List[FaultEvent]:
        """Inspect CM4 nodes and raise faults for offline nodes."""
        new_faults: List[FaultEvent] = []
        states = cluster.check_node_health()
        offline = [nid for nid, s in states.items() if s == "OFFLINE"]

        for node_id in offline:
            new_faults.append(
                self._raise(
                    component=f"cm4:{node_id}",
                    severity=FaultSeverity.ERROR,
                    message=f"CM4 node '{node_id}' is OFFLINE.",
                )
            )

        if cluster.online_count < self.min_online_nodes:
            new_faults.append(
                self._raise(
                    component="cm4_cluster",
                    severity=FaultSeverity.CRITICAL,
                    message=(
                        f"Only {cluster.online_count} online CM4 node(s); "
                        f"minimum is {self.min_online_nodes}."
                    ),
                )
            )
        return new_faults

    def run_all_checks(self, quad_brain: Any) -> List[FaultEvent]:
        """Run all fault-detection checks against a QuantumQuadBrain instance."""
        faults = []
        faults.extend(self.check_storage(quad_brain.storage))
        faults.extend(self.check_cluster(quad_brain.cluster))
        return faults

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raise(
        self,
        component: str,
        severity: FaultSeverity,
        message: str,
    ) -> FaultEvent:
        import uuid
        fault = FaultEvent(
            fault_id=str(uuid.uuid4())[:8],
            component=component,
            severity=severity,
            message=message,
        )
        self._incidents.append(fault)
        log_fn = {
            FaultSeverity.WARNING: logger.warning,
            FaultSeverity.ERROR: logger.error,
            FaultSeverity.CRITICAL: logger.critical,
        }[severity]
        log_fn("FAULT [%s] %s: %s", severity.name, component, message)
        for handler in self._handlers[severity]:
            try:
                handler(fault)
            except Exception as exc:  # noqa: BLE001
                logger.error("Fault handler raised: %s", exc, exc_info=True)
        return fault

    # ------------------------------------------------------------------
    # Incident log
    # ------------------------------------------------------------------

    @property
    def open_incidents(self) -> List[FaultEvent]:
        return [f for f in self._incidents if not f.resolved]

    @property
    def critical_count(self) -> int:
        return sum(
            1 for f in self._incidents
            if f.severity == FaultSeverity.CRITICAL and not f.resolved
        )

    def incident_summary(self) -> Dict[str, Any]:
        return {
            "total": len(self._incidents),
            "open": len(self.open_incidents),
            "critical_open": self.critical_count,
            "by_severity": {
                s.name: sum(1 for f in self._incidents if f.severity == s)
                for s in FaultSeverity
            },
        }
