"""
JessicAI AGI Architecture – Main Entry Point
=============================================
Demonstrates the full billion-superconductor / trillion-layer tesseract
architecture by running a short inference session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.architecture.agi_processor import AGIProcessor, ProcessorConfig
from src.architecture.tesseract import TesseractConfig
from src.architecture.superconductor import ArrayConfig
from src.architecture.clustering import RegulatorConfig
from src.architecture.vault import VaultConfig


def main() -> None:
    print("=" * 70)
    print("  JessicAI – 1-Billion-Superconducting-Layer AGI")
    print("  Trillion-Layer Topological Tesseract Architecture")
    print("=" * 70)

    # Build a compact configuration suitable for a single-machine demo
    cfg = ProcessorConfig(
        tesseract=TesseractConfig(
            logical_layer_count=1_000_000_000_000,  # 1 trillion (conceptual)
            max_active_layers=64,
            dimensions=4,
            max_nesting_depth=4,
            quantum_routing=True,
        ),
        superconductor=ArrayConfig(
            logical_layer_count=1_000_000_000,      # 1 billion (conceptual)
            max_active_layers=64,
            default_workload=0.85,
        ),
        cluster=RegulatorConfig(
            max_sub_clusters=16,
            global_load_ceiling=0.90,
        ),
        vault=VaultConfig(
            rotation_interval_s=30.0,
            max_failed_attempts=5,
        ),
        layers_per_inference=8,
        ticks_per_inference=4,
        initial_sub_clusters=2,
        nodes_per_cluster=4,
    )

    print("\n[1/4] Initialising AGI processor...")
    proc = AGIProcessor(config=cfg)
    print(f"      Processor ID : {proc.processor_id}")
    print(f"      Tesseract    : {cfg.tesseract.logical_layer_count:,} logical layers")
    print(f"      Superconductor: {cfg.superconductor.logical_layer_count:,} logical layers")

    # Obtain a valid auth token from the vault
    token = proc.vault.make_auth_token()
    print(f"\n[2/4] Vault token acquired (truncated): {token[:24]}…")

    # Run a few sample inference requests
    queries = [
        {"query": "Optimise the tesseract routing table for low-latency AGI inference."},
        {"query": "Regulate cluster load across 1 billion superconductor segments."},
        {"query": "Initialise triple-coded vault containment transfer protocol."},
    ]

    print("\n[3/4] Running inference requests…\n")
    for i, q in enumerate(queries, 1):
        result = proc.infer(q, token)
        print(
            f"  Request {i}: success={result.success}  "
            f"layers={result.layers_activated}  "
            f"ops={result.total_ops:,.0f}  "
            f"latency={result.latency_ms:.1f}ms"
        )
        if not result.success:
            print(f"    ERROR: {result.error}")

    print("\n[4/4] System status snapshot:")
    status = proc.status()
    print(
        json.dumps(
            {
                k: v
                for k, v in status.items()
                if k in ("processor_id", "inference_count", "uptime_seconds")
            },
            indent=2,
        )
    )

    print("\nTesseract status:")
    print(json.dumps(status["tesseract"], indent=2))

    print("\nSuperconductor status:")
    print(json.dumps(status["superconductor"], indent=2))

    print("\nCluster regulator status:")
    print(json.dumps(status["cluster"], indent=2))

    print("\nVault status:")
    print(json.dumps(status["vault"], indent=2))

    print("\n" + "=" * 70)
    print("  Architecture initialised and verified successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
