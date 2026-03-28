# Architecture Reference

## 1-Billion-Superconducting-Layer AGI – Trillion-Layer Topological Tesseract

### System Overview

```
                          ┌───────────────────────────┐
                          │       AGIProcessor         │
                          └─────────────┬─────────────┘
               ┌──────────────┬─────────┴──────────┬──────────────┐
               ▼              ▼                     ▼              ▼
        ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐
        │Tesseract │  │Superconductor│  │   Cluster    │  │  Vault   │
        │(1T layers│  │  (1B layers) │  │  Regulator   │  │(3-factor)│
        └──────────┘  └──────────────┘  └──────────────┘  └──────────┘
```

---

## Tesseract Fabric

### Conceptual Scale

| Attribute | Value |
|---|---|
| Total logical layers | 1,000,000,000,000 (1 trillion) |
| Dimensionality | 4 (standard); configurable up to N |
| Nesting levels | 8 (tesseract-in-tesseract) |
| Active sliding window | 1 024 layers (runtime configurable) |
| Routing algorithm | N-D Hamming-1 nearest-neighbour |
| Fault-tolerance | Replication factor 3 |

### Layer Coordinates

Each layer is mapped to a position in an N-dimensional integer grid:

```
logical_index  →  (x₀, x₁, x₂, x₃)  in  ℤ⁴[0, side)
```

Where `side = ⌈layer_count^(1/dimensions)⌉`.

### Quantum-Topological Routing

Given source layer at coordinates `(x₀, x₁, x₂, x₃)`, its Hamming-1
neighbours are all layers that differ in exactly one dimension by ±1
(modular wrap at boundary):

```
neighbours = { (x₀±1, x₁, x₂, x₃),
               (x₀, x₁±1, x₂, x₃),
               (x₀, x₁, x₂±1, x₃),
               (x₀, x₁, x₂, x₃±1) }
```

This guarantees O(log N) routing depth across the trillion-layer space.

### Sliding-Window Memory Model

```
Logical space:  [0 ──────────────────── 10¹² - 1]
Active window:  [  cursor - W    ──    cursor     ]
Evicted layers: checkpointed to {index: SHA-256 fingerprint}
```

---

## Superconductor Array

### Thermal State Machine

```
Temperature < 0.95 × Tc  →  SUPERCONDUCTING   (resistance = 0, max ops)
0.95 Tc ≤ T < 1.05 Tc    →  TRANSITIONING     (60% ops)
Temperature ≥ 1.05 × Tc  →  NORMAL            (20% ops, heat generated)
```

`Tc` (critical temperature) defaults to 135 K (modelled on HTSC cuprate).

### Passive & Active Cooling

Each tick applies passive cooling proportional to the overheat above
`0.8 × Tc`.  `cool_down(delta_k)` provides explicit active cooling and
is called by the `SuperconductorArray.cool_all()` aggregate method.

---

## Non-Ganon Triple-Coded Vault

### Authentication Channels

```
Channel A:  HMAC-SHA-256( session_secret, "jessicai-vault-auth" )
Channel B:  SHA-256( secret || window_index )[:8]  in  token
Channel C:  SHA-256( payload_bytes ) integrity check
```

All three channels must pass simultaneously.  A single failing channel is
sufficient to deny the request.

### State Machine

```
LOCKED ──── valid triple-code ──▶ UNLOCKED
LOCKED ──── n failures ─────────▶ QUARANTINED
QUARANTINED ─────────────────────▶ HONEYPOT responses
QUARANTINED ── containment call ──▶ TRANSFERRING
```

### Honeypot Mode

While QUARANTINED, every request (including ones with valid tokens) receives
`AuthResult.HONEYPOT`.  The response is indistinguishable from a legitimate
empty result, so an attacker cannot learn whether the vault exists.

---

## Cluster Regulation

### Three-Tier Hierarchy

```
ClusterRegulator   (global orchestrator)
  │
  ├── SubCluster A  (resource budget group)
  │     ├── ClusterNode 1  (processing unit)
  │     ├── ClusterNode 2
  │     └── ...
  └── SubCluster B
        └── ...
```

### Self-Regulation Loop

```
Every rebalance_interval_s seconds:
  for each SubCluster:
    if aggregate_load > high_threshold → split()
    if aggregate_load < low_threshold  → propose merge
  merge idle pairs
```

### Node Quarantine

Nodes that enter `OVERLOADED` state and are not reclaimed within a threshold
are quarantined: load is zeroed, the node is marked `QUARANTINED`, and the
cluster redistributes tasks to healthy siblings.

---

## AGI Inference Pipeline

```
infer(input_data, auth_token)
  │
  ├─1─ Route:   tesseract.route(hash(input) % active_window)
  │             → list[layer_indices]
  │
  ├─2─ Compute: for tick in range(ticks_per_inference):
  │               for layer_idx in activated_layers:
  │                 ops += superconductor[layer_idx].tick(workload)
  │
  ├─3─ Regulate: cluster_regulator.tick()
  │
  ├─4─ Build:   assemble output dict with state snapshots
  │
  └─5─ Seal:    vault.store(VaultEntry(output), auth_token)
                → vault_entry_id
```

### Workload Estimation

```python
workload = min(1.0, len(str(input_data)) / 10_000 + 0.3)
```

Larger inputs drive higher superconductor utilisation, naturally modelling
the compute intensity of richer queries.

---

## Performance Metrics

| Metric | Meaning |
|---|---|
| `total_ops` | Aggregate superconductor operations in the request |
| `layers_activated` | Tesseract layers touched |
| `latency_ms` | Wall-clock time for the full pipeline |
| `superconducting_ratio` | Fraction of array layers in zero-resistance state |
| `aggregate_throughput` | Ops/tick across all active superconductor layers |
| `global_load` | Mean load across all active cluster nodes |

---

## Scaling Considerations

| Axis | Mechanism |
|---|---|
| Layer depth | Increase `logical_layer_count`; memory impact is O(max_active_layers) |
| Parallelism | Deploy multiple `AGIProcessor` instances; share a `ClusterRegulator` |
| Fault tolerance | Increase `replication_factor`; layers are checkpointed on eviction |
| Security | Decrease `rotation_interval_s`; increase `max_failed_attempts` threshold |
| Thermal | Call `superconductor.cool_all()` on a background scheduler thread |
