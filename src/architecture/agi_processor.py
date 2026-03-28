"""
AGI Processor
==============
Coordinates the three core subsystems of the JessicAI architecture:

  1. Trillion-layer topological ``Tesseract`` – multi-dimensional data fabric.
  2. Billion-layer ``SuperconductorArray`` – accelerated computation.
  3. ``ClusterRegulator`` – self-organising processing node hierarchy.

The ``AGIProcessor`` exposes a high-level inference interface.  Each call to
``infer`` follows this pipeline:

  1. **Route** – the tesseract's quantum-topological routing determines which
     layers and superconductor segments handle the request.
  2. **Compute** – selected superconductor layers execute the workload for the
     specified number of ticks.
  3. **Regulate** – the cluster regulator rebalances load across sub-clusters.
  4. **Vault check** – the result is sealed in the triple-coded vault before
     being returned to the caller.

Quantum-topological inference
------------------------------
Real quantum hardware is not required.  The processor uses the tesseract's
N-dimensional routing table to distribute sub-tasks across spatially-proximate
layers, mimicking locality-of-reference benefits seen in quantum annealing.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .clustering import ClusterRegulator, RegulatorConfig, SubClusterConfig
from .superconductor import SuperconductorArray, ArrayConfig
from .tesseract import Tesseract, TesseractConfig
from .vault import TripleCodedVault, VaultConfig, VaultEntry, AuthResult


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    The outcome of a single AGI inference call.

    Attributes
    ----------
    request_id:
        Unique identifier for this inference request.
    output:
        The processed output (domain-specific dictionary).
    layers_activated:
        Number of tesseract layers activated during processing.
    total_ops:
        Aggregate operations executed across all superconductor layers.
    latency_ms:
        Wall-clock latency in milliseconds.
    vault_entry_id:
        ID of the vault entry where this result is secured.
    success:
        Whether the inference completed without errors.
    """

    request_id: str
    output: Dict[str, Any]
    layers_activated: int
    total_ops: float
    latency_ms: float
    vault_entry_id: Optional[str]
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Processor configuration
# ---------------------------------------------------------------------------

@dataclass
class ProcessorConfig:
    """Aggregate configuration bundle for ``AGIProcessor``."""
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)
    superconductor: ArrayConfig = field(default_factory=ArrayConfig)
    cluster: RegulatorConfig = field(default_factory=RegulatorConfig)
    vault: VaultConfig = field(default_factory=VaultConfig)

    # Inference settings
    layers_per_inference: int = 8     # layers activated per request
    ticks_per_inference: int = 4      # superconductor simulation ticks per request
    initial_sub_clusters: int = 4
    nodes_per_cluster: int = 8


# ---------------------------------------------------------------------------
# AGI Processor
# ---------------------------------------------------------------------------

class AGIProcessor:
    """
    High-level AGI inference engine integrating the four subsystems.

    Example
    -------
    >>> proc = AGIProcessor()
    >>> token = proc.vault.make_auth_token()
    >>> result = proc.infer({"query": "What is the capital of France?"}, token)
    >>> result.success
    True
    """

    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        self.config = config or ProcessorConfig()
        self.processor_id = str(uuid.uuid4())

        # Initialise subsystems
        self.tesseract = Tesseract(config=self.config.tesseract)
        self.superconductor = SuperconductorArray(config=self.config.superconductor)
        self.regulator = ClusterRegulator(config=self.config.cluster)
        self.vault = TripleCodedVault(config=self.config.vault)

        # Bootstrap initial layers and clusters
        self.tesseract.bootstrap(
            count=min(16, self.config.tesseract.max_active_layers),
            spawn_children=True,
        )
        self.superconductor.bootstrap(
            count=min(32, self.config.superconductor.max_active_layers),
        )
        for _ in range(self.config.initial_sub_clusters):
            self.regulator.create_sub_cluster(
                node_count=self.config.nodes_per_cluster,
                sub_config=SubClusterConfig(),
            )

        self._inference_count: int = 0
        self._started_at: float = time.time()

    # ------------------------------------------------------------------
    # Primary inference interface
    # ------------------------------------------------------------------

    def infer(
        self,
        input_data: Dict[str, Any],
        auth_token: str,
    ) -> InferenceResult:
        """
        Run an AGI inference request.

        Parameters
        ----------
        input_data:
            Arbitrary input dictionary representing the task/query.
        auth_token:
            Triple-coded vault authentication token.  Obtain via
            ``self.vault.make_auth_token()``.

        Returns
        -------
        InferenceResult
        """
        start = time.time()
        request_id = str(uuid.uuid4())

        try:
            # 1. Route through tesseract
            activated_layers = self._route(input_data)

            # 2. Execute on superconductor
            total_ops = self._compute(
                activated_layers,
                workload=self._estimate_workload(input_data),
            )

            # 3. Regulate clusters
            self.regulator.tick()

            # 4. Build output
            output = self._build_output(input_data, activated_layers, total_ops)

            # 5. Seal in vault
            vault_entry_id = self._seal_in_vault(
                request_id, output, auth_token
            )

            self._inference_count += 1
            latency_ms = (time.time() - start) * 1_000

            return InferenceResult(
                request_id=request_id,
                output=output,
                layers_activated=len(activated_layers),
                total_ops=total_ops,
                latency_ms=round(latency_ms, 3),
                vault_entry_id=vault_entry_id,
                success=True,
            )

        except Exception as exc:  # pylint: disable=broad-except
            latency_ms = (time.time() - start) * 1_000
            return InferenceResult(
                request_id=request_id,
                output={},
                layers_activated=0,
                total_ops=0.0,
                latency_ms=round(latency_ms, 3),
                vault_entry_id=None,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _route(self, input_data: Dict[str, Any]) -> List[int]:
        """
        Determine which tesseract layer indices to activate for this request.
        Uses the quantum-topological routing table seeded by input hash.
        """
        seed_idx = (
            int(hashlib.sha256(str(input_data).encode()).hexdigest(), 16)
            % self.config.tesseract.logical_layer_count
        )
        # Ensure the seed layer is active
        self.tesseract.create_layer(logical_index=seed_idx % 1_024)

        dest_indices = self.tesseract.route(seed_idx % 1_024, input_data)
        # Also activate up to layers_per_inference additional layers
        n = self.config.layers_per_inference
        activated = list(set([seed_idx % 1_024] + dest_indices[:n - 1]))

        for idx in activated:
            if self.tesseract.get_layer(idx) is None:
                self.tesseract.create_layer(logical_index=idx)

        return activated

    def _compute(self, layer_indices: List[int], workload: float) -> float:
        """Run superconductor ticks for the given layer indices."""
        total_ops = 0.0
        for _ in range(self.config.ticks_per_inference):
            for sc_idx in layer_indices:
                sc_layer = self.superconductor.get_layer(sc_idx % 1_024)
                if sc_layer is None:
                    sc_layer = self.superconductor.allocate_layer(
                        sc_idx % 1_024
                    )
                total_ops += sc_layer.tick(workload)
        return total_ops

    def _estimate_workload(self, input_data: Dict[str, Any]) -> float:
        """Heuristic workload estimate (0.0–1.0) based on input complexity."""
        raw = len(str(input_data))
        return min(1.0, raw / 10_000.0 + 0.3)

    def _build_output(
        self,
        input_data: Dict[str, Any],
        activated_layers: List[int],
        total_ops: float,
    ) -> Dict[str, Any]:
        """Construct the inference output record."""
        return {
            "processor_id": self.processor_id,
            "input_hash": hashlib.sha256(str(input_data).encode()).hexdigest()[:16],
            "activated_tesseract_layers": activated_layers,
            "superconductor_ops": total_ops,
            "cluster_state": self.regulator.status(),
            "tesseract_state": self.tesseract.status(),
            "superconductor_state": self.superconductor.status(),
            "timestamp": time.time(),
        }

    def _seal_in_vault(
        self,
        request_id: str,
        output: Dict[str, Any],
        auth_token: str,
    ) -> Optional[str]:
        """Store the result in the triple-coded vault; return the entry ID."""
        entry = VaultEntry(payload={"request_id": request_id, "output": output})
        result = self.vault.store(entry, auth_token)
        if result == AuthResult.GRANTED:
            return entry.entry_id
        return None

    # ------------------------------------------------------------------
    # Health / status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "processor_id": self.processor_id,
            "inference_count": self._inference_count,
            "uptime_seconds": round(time.time() - self._started_at, 2),
            "tesseract": self.tesseract.status(),
            "superconductor": self.superconductor.status(),
            "cluster": self.regulator.status(),
            "vault": self.vault.status(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AGIProcessor(id={self.processor_id!r}, "
            f"inferences={self._inference_count})"
        )
