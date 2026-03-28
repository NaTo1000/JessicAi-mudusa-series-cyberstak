# JessicAi-mudusa-series-cyberstak

# Quantum Quad-Brain · NVMe + DRAM + CM4 Compute Array

An optimal, modular, and scalable compute array that integrates:

- **Quantum Quad-Brain** — four independent quantum-inspired inference engines
  (`alpha` 6q / `beta` 8q / `gamma` 10q / `delta` 12q) with consensus voting.
- **NVMe Storage Pipeline** — striped ultra-fast SSD array (RAID-0 style) for
  quantum data management with automatic failover.
- **DRAM Cache Layer** — LRU/LFU eviction cache that sits in front of NVMe,
  delivering sub-microsecond hot-data access.
- **Raspberry Pi CM4 Cluster** — distributed compute nodes managed by a
  heartbeat-aware cluster manager and a pluggable workload distributor.
- **Real-Time Performance Monitor** — sliding-window metrics with ASCII
  dashboard, sparklines, and energy-efficiency KPIs.
- **Fault Tolerance Manager** — automatic detection and recovery for NVMe
  device failures and CM4 node outages.

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from quantum_quad_brain.array.scalable_array import ScalableComputeArray

array = ScalableComputeArray(dram_capacity_gb=8.0, initial_cm4_nodes=4)

# Single-brain inference
result = array.run_inference(b"input payload")
print(result)

# Consensus across all four brains
consensus = array.run_consensus_inference(b"input payload")
print(consensus)

# ASCII performance dashboard
print(array.dashboard())
```

## Running Tests

```bash
pytest tests/ -v
```

## Documentation

| Document | Description |
|----------|-------------|
| [Hardware Setup](docs/hardware_setup.md) | CM4, NVMe, and network assembly |
| [Software Integration](docs/software_integration.md) | Installation, config, and API usage |
| [Scalability Guide](docs/scalability.md) | Horizontal/vertical scaling, fault tolerance, benchmarking |

## Repository Structure

```
quantum_quad_brain/
  nvme_dram/         NVMe storage pipeline + DRAM cache manager
  cm4_cluster/       CM4 cluster manager + workload distributor
  quantum_core/      Quantum inference engine + quad-brain coordinator
  metrics/           Performance monitor + ASCII visualizer
  array/             Scalable array + fault tolerance manager
  config/            YAML configuration template
tests/               80 unit / integration tests
docs/                Hardware, software, and scalability documentation
```
