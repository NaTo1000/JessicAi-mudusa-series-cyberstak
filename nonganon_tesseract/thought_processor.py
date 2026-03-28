"""
ThoughtProcessor — Top-level Orchestrator
==========================================

Ties together every subsystem of the Nonganon Tesseract Topological
Octagon architecture into a single, easy-to-use entry-point.

Processing pipeline
-------------------
1.  **Ingest** – raw input data is written into the TesseractOctagon's
    input layer (layer 0) and broadcast to adjacent layers.
2.  **Triple-cluster** – the data is partitioned and clustered by the
    TripleCluster engine; the cluster summary is vaulted.
3.  **Quantum inference** – the QuantumInferenceEngine updates its
    probability distribution using the cluster summary.
4.  **Twinbrain decision** – the normalised state vector from the
    inference engine is fed into the TwinbrainAlgorithm, which
    produces a fused decision vector.
5.  **Pex-chain dispatch** – a three-node PexChain propagates the
    decision context through ingest → process → dispatch, applying
    final business logic before returning.
6.  **Vault** – the full result context is sealed inside the
    SecurityVault for integrity.

All intermediate artefacts are also stored in the TesseractOctagon's
layer states so they remain accessible for diagnostics.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from .pex_chain import PexChain
from .quantum_inference import QuantumInferenceEngine
from .security import SecurityVault
from .tesseract_octagon import TesseractOctagon
from .triple_cluster import TripleCluster
from .twinbrain import TwinbrainAlgorithm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ThoughtProcessor
# ---------------------------------------------------------------------------


class ThoughtProcessor:
    """Nonganon Tesseract Topological Octagon thought processor.

    Orchestrates the full pipeline from raw data ingestion through to
    a signed, vaulted decision output.

    Parameters
    ----------
    n_outcomes:
        Number of discrete decision outcomes for the inference engine.
    alpha:
        Adaptive learning rate for the inference engine (0–1).
    vault_passphrase:
        Optional passphrase used to protect the result vault entry.
        If ``None``, the vault entry is sealed but not passphrase-
        protected.
    seed:
        Optional random seed for reproducible triple-clustering.

    Examples
    --------
    >>> tp = ThoughtProcessor(n_outcomes=3, alpha=0.7, seed=42)
    >>> data = [[float(i), float(i*2), float(i*3)] for i in range(12)]
    >>> result = tp.process(data)
    >>> print(result["decision"])
    """

    def __init__(
        self,
        n_outcomes: int = 4,
        alpha: float = 0.5,
        vault_passphrase: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_outcomes = n_outcomes
        self.alpha = alpha
        self._vault_passphrase = vault_passphrase

        self.tesseract = TesseractOctagon()
        self.triple_cluster = TripleCluster(seed=seed)
        self.inference_engine = QuantumInferenceEngine(
            n_outcomes=n_outcomes, alpha=alpha
        )
        self.twinbrain = TwinbrainAlgorithm()
        self.security_vault = SecurityVault(vault_id="nonganon-main-vault")
        self._chain = self._build_pex_chain()

    # ------------------------------------------------------------------

    def process(self, data: List[List[float]]) -> Dict[str, Any]:
        """Run the complete pipeline on *data* and return a result dict.

        Parameters
        ----------
        data:
            List of equal-length numeric vectors (at least 3 rows).

        Returns
        -------
        Dict[str, Any]
            A result dict with keys:
            ``"decision"``, ``"inference_state"``, ``"cluster_summary"``,
            ``"diagnostics"``, ``"vault_manifest"``
        """
        logger.info("ThoughtProcessor: starting pipeline (rows=%d).", len(data))

        # 1. Ingest -------------------------------------------------------
        self.tesseract.push(layer_index=0, key="raw_data", value=data)
        self.tesseract.route(layer_index=0, key="raw_data")

        # 2. Triple-cluster -----------------------------------------------
        self.triple_cluster.fit(data)
        cluster_summary = self.triple_cluster.summary()
        self.tesseract.push(layer_index=1, key="cluster_summary", value=cluster_summary)

        # 3. Quantum inference --------------------------------------------
        inference_state = self.inference_engine.infer_clusters(cluster_summary)
        self.tesseract.push(layer_index=4, key="inference_state", value=inference_state)

        # 4. Twinbrain decision -------------------------------------------
        # Pad / trim inference_state to match the twinbrain input expectation
        tb_input = (inference_state + [0.0] * len(data[0]))[: len(data[0])]
        decision = self.twinbrain.decide(tb_input)
        self.tesseract.push(layer_index=5, key="decision", value=decision)

        # 5. Pex-chain dispatch -------------------------------------------
        ctx = self._chain.execute(
            "ingest",
            {
                "raw_data": data,
                "cluster_summary": cluster_summary,
                "inference_state": inference_state,
                "decision": decision,
            },
        )

        # 6. Vault --------------------------------------------------------
        result: Dict[str, Any] = {
            "decision": ctx.get("decision", decision),
            "inference_state": inference_state,
            "cluster_summary": cluster_summary,
            "diagnostics": self.twinbrain.diagnostics(),
            "tesseract_snapshot": self.tesseract.collect_states(),
            "vault_manifest": {},
        }
        vault_entry = self.security_vault.store(
            "result",
            result,
            passphrase=self._vault_passphrase,
        )
        vault_entry.seal()
        result["vault_manifest"] = self.security_vault.manifest()

        logger.info("ThoughtProcessor: pipeline complete.")
        return result

    # ------------------------------------------------------------------

    def _build_pex_chain(self) -> PexChain:
        """Construct the three-node Pex-Chain for the dispatch stage."""
        chain = PexChain(name="thought-processor-chain")

        def _ingest(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            logger.debug("PexChain[ingest]: context keys=%s.", list(ctx.keys()))
            return {"chain_ingested": True}

        def _process(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            decision: List[float] = ctx.get("decision", [])
            most_probable = (
                decision.index(max(decision)) if decision else -1
            )
            return {"most_probable_outcome": most_probable}

        def _dispatch(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            outcome = ctx.get("most_probable_outcome", -1)
            digest = hashlib.sha256(str(outcome).encode()).hexdigest()
            return {"dispatch_digest": digest}

        chain.register("ingest", _ingest)
        chain.register("process", _process)
        chain.register("dispatch", _dispatch)
        chain.link("ingest", "process")
        chain.link("process", "dispatch")
        return chain

    # ------------------------------------------------------------------

    def topology(self) -> str:
        """Return the TesseractOctagon topology summary."""
        return self.tesseract.topology_summary()

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ThoughtProcessor(n_outcomes={self.n_outcomes}, "
            f"alpha={self.alpha})"
        )
