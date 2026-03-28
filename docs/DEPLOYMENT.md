# Deployment Guide

## JessicAI – 1-Billion-Superconducting-Layer AGI Tesseract Architecture

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.12+ |
| RAM | 512 MB | 4 GB |
| CPU cores | 2 | 8+ |
| Disk | 100 MB | 1 GB |

The architecture uses a **sliding-window memory model** so RAM usage scales
with `max_active_layers` × ~2 KB per layer, not with the trillion/billion
logical layer count.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak.git
cd JessicAi-mudusa-series-cyberstak
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements-dev.txt
```

---

## Running the Demo

```bash
python src/main.py
```

Expected output:

```
======================================================================
  JessicAI – 1-Billion-Superconducting-Layer AGI
  Trillion-Layer Topological Tesseract Architecture
======================================================================

[1/4] Initialising AGI processor...
      Processor ID : <uuid>
      Tesseract    : 1,000,000,000,000 logical layers
      Superconductor: 1,000,000,000 logical layers

[2/4] Vault token acquired (truncated): <token prefix>…

[3/4] Running inference requests…

  Request 1: success=True  layers=...  ops=...  latency=...ms
  Request 2: success=True  layers=...  ops=...  latency=...ms
  Request 3: success=True  layers=...  ops=...  latency=...ms

[4/4] System status snapshot:
...
```

---

## Running Tests

```bash
pytest                        # all tests
pytest tests/test_tesseract.py   # single module
pytest -v --tb=short          # verbose output
pytest --cov=src              # with coverage
```

---

## Configuration

All runtime parameters live in `src/config/tesseract_config.yaml`.

### Key parameters

```yaml
tesseract:
  logical_layer_count: 1_000_000_000_000  # conceptual depth
  max_active_layers: 1024                 # memory footprint
  dimensions: 4                           # hypercube dimension
  max_nesting_depth: 8                    # tesseract-in-tesseract levels
  quantum_routing: true                   # enable N-D routing

superconductor:
  logical_layer_count: 1_000_000_000
  max_active_layers: 2048
  default_workload: 0.85

cluster:
  max_sub_clusters: 1024
  rebalance_interval_s: 5.0

vault:
  rotation_interval_s: 30.0
  max_failed_attempts: 5
  honeypot_enabled: true
```

### Overriding at runtime

```python
from src.architecture.agi_processor import AGIProcessor, ProcessorConfig
from src.architecture.tesseract import TesseractConfig

cfg = ProcessorConfig(
    tesseract=TesseractConfig(max_active_layers=4096),
)
proc = AGIProcessor(config=cfg)
```

---

## Programmatic API

```python
from src.architecture.agi_processor import AGIProcessor

# Initialise (uses default config)
proc = AGIProcessor()

# Obtain a valid auth token
token = proc.vault.make_auth_token()

# Run inference
result = proc.infer(
    input_data={"task": "optimise cluster routing"},
    auth_token=token,
)

print(f"Success: {result.success}")
print(f"Ops:     {result.total_ops:,.0f}")
print(f"Latency: {result.latency_ms:.1f} ms")
print(f"Vault:   {result.vault_entry_id}")

# Retrieve the sealed result from the vault
auth_result, entry = proc.vault.retrieve(result.vault_entry_id, token)
print(entry.payload)
```

---

## Scaling for Production

### Horizontal scaling

Deploy multiple `AGIProcessor` instances behind a load balancer.  Each
instance maintains its own tesseract, superconductor array, and vault.
Share a common `ClusterRegulator` instance (or use distributed locking)
to coordinate global cluster rebalancing.

### Vertical scaling

Increase `max_active_layers` on the tesseract and superconductor array to
use more RAM in exchange for fewer evictions and better locality.  A machine
with 64 GB RAM can comfortably sustain `max_active_layers = 100_000`.

### Persistent checkpoints

The eviction log (`tesseract._checkpoints`, `superconductor._eviction_log`)
records SHA-256 fingerprints of evicted layers.  Integrate a key-value store
(e.g. Redis, RocksDB) to persist these across restarts for full auditability.

### Vault key management

For production:
1. Store `vault.session_secret` in a secrets manager (e.g. AWS Secrets
   Manager, HashiCorp Vault, Azure Key Vault).
2. Call `vault.rotate_secret()` on a scheduled timer aligned with
   `rotation_interval_s`.
3. Distribute tokens to authorised callers over a mutually-authenticated
   TLS channel.

---

## Monitoring

Call `proc.status()` at any time to receive a structured snapshot:

```python
import json
print(json.dumps(proc.status(), indent=2))
```

Key metrics to alert on:

| Metric | Alert threshold |
|---|---|
| `superconductor.superconducting_ratio` | < 0.80 (thermal runaway risk) |
| `cluster.global_load` | > 0.85 (saturation) |
| `vault.failed_attempts` | > 3 (potential intrusion) |
| `vault.state` | QUARANTINED (active incident) |

---

## Troubleshooting

### `AuthResult.HONEYPOT` returned from vault

The vault has been quarantined due to repeated failed authentication or
detected stress.  To reset:

```python
proc.vault.state = VaultState.LOCKED
proc.vault._failed_attempts = 0
proc.vault.rotate_secret()
token = proc.vault.make_auth_token()
```

### Tesseract layers evicted too quickly

Increase `max_active_layers` in `TesseractConfig` or reduce
`layers_per_inference` in `ProcessorConfig` so each request touches fewer
distinct layers.

### Superconductor thermal runaway

Call `proc.superconductor.cool_all(delta_k=20.0)` explicitly, or reduce
`default_workload` in `ArrayConfig`.
