# Scalability Guide

## Architecture Overview

The Quantum Quad-Brain compute array follows a layered, modular design that
allows independent scaling of each subsystem:

```
┌─────────────────────────────────────────────────────────┐
│                  ScalableComputeArray                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │             QuantumQuadBrain (4 brains)          │   │
│  │  alpha(6q)  beta(8q)  gamma(10q)  delta(12q)    │   │
│  └────────────────────┬────────────────────────────┘   │
│           ┌───────────┴──────────┐                      │
│  ┌────────▼────────┐  ┌─────────▼──────────────────┐   │
│  │  NVMe Pipeline  │  │     CM4 Cluster Manager    │   │
│  │  (striped SSDs) │  │  node0 node1 node2 node3…  │   │
│  └────────┬────────┘  └─────────┬──────────────────┘   │
│  ┌────────▼────────┐  ┌─────────▼──────────────────┐   │
│  │   DRAM Cache    │  │   Workload Distributor      │   │
│  │   (LRU/LFU)     │  │   (RR / LeastLoaded / Rnd) │   │
│  └─────────────────┘  └────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  PerformanceMonitor + MetricsVisualizer           │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │            FaultToleranceManager                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Horizontal Scaling

### Adding More CM4 Nodes

CM4 nodes can be added live using `add_cm4_node()`.  There is no upper limit
enforced in software; the practical ceiling is the port count on the network
switch and the IP address range in your configuration.

```python
for i in range(4, 16):
    array.add_cm4_node(f"cm4-{i:03d}", f"10.0.0.{100 + i}")
```

### Adding More NVMe Drives

Each additional NVMe device added via `add_nvme_device()` increases the
aggregate read bandwidth proportionally (RAID-0 stripe):

| Devices | Theoretical Peak Read | Example Config |
|---------|-----------------------|----------------|
| 1 | 7 GB/s | Entry level |
| 2 | 14 GB/s | Recommended |
| 4 | 28 GB/s | High throughput |
| 8 | 56 GB/s | Maximum density |

---

## Vertical Scaling

### Increasing DRAM Cache

Pass a larger `dram_capacity_gb` value when constructing the array.  The
cache automatically rebalances on the next `put` operation.

```python
array = ScalableComputeArray(dram_capacity_gb=64.0)
```

### Tuning Quantum Brain Resolution

Higher `n_qubits` values provide finer output class resolution at the cost of
longer inference latency.  Adjust in `array_config.yaml` or via `BrainConfig`:

| Brain | Qubits | Classes | Latency (rel.) |
|-------|--------|---------|----------------|
| alpha | 6 | 64 | 1× |
| beta | 8 | 256 | ~3× |
| gamma | 10 | 1 024 | ~10× |
| delta | 12 | 4,096 | ~40× |

---

## Fault Tolerance

### No Single Point of Failure

- **Storage**: Data is striped across all healthy NVMe devices.  If one device
  fails, remaining devices continue serving reads and writes.  The
  `FaultToleranceManager` removes the failed device automatically.

- **Compute**: If a CM4 node goes OFFLINE (missed heartbeats), the cluster
  manager marks it offline and the workload distributor routes all new tasks
  to available nodes.

- **Cache**: The DRAM cache is a hot-path optimisation.  A total cache failure
  (e.g., out-of-memory) causes transparent fall-through to NVMe reads, with no
  data loss.

- **Brains**: If a brain produces an error during `parallel_infer`, the
  coordinator logs the failure and still returns results from the remaining
  three brains.  The consensus vote adapts to the available set.

### Configuring Fault Thresholds

```yaml
# array_config.yaml
fault_tolerance:
  min_healthy_nvme: 1   # CRITICAL alert if below this
  min_online_nodes: 1   # CRITICAL alert if below this
```

---

## Benchmarking

```python
import time

array = ScalableComputeArray(initial_cm4_nodes=4)

payloads = [bytes(range(256))] * 100
start = time.perf_counter()
for p in payloads:
    array.run_inference(p)
elapsed = time.perf_counter() - start

print(f"{len(payloads) / elapsed:.1f} inferences/s")
print(array.sparklines())
```

---

## Recommended Production Topology

```
                  ┌──────────────────────────────────┐
                  │      Host Machine (x86-64)        │
                  │  • 64 GB RAM (DRAM cache)         │
                  │  • 4× NVMe PCIe Gen 4 (8 TB)      │
                  │  • 1 GbE / 10 GbE NIC             │
                  └──────────────┬───────────────────┘
                                 │ Ethernet
                  ┌──────────────▼───────────────────┐
                  │       Managed 1 GbE Switch        │
                  └──┬──────┬──────┬──────┬──────────┘
              CM4-000 CM4-001 CM4-002 CM4-003 …
              (8 GB)  (8 GB)  (8 GB)  (8 GB)
```
