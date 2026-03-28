# Software Integration Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak.git
cd JessicAi-mudusa-series-cyberstak

# Install the package (and dev dependencies)
pip install -e ".[dev]"
```

---

## Quick Start

```python
from quantum_quad_brain.array.scalable_array import ScalableComputeArray
from quantum_quad_brain.quantum_core.quad_brain import BrainConfig

# Create the array with default settings (4 CM4 nodes, 8 GB DRAM cache)
array = ScalableComputeArray(dram_capacity_gb=8.0, initial_cm4_nodes=4)

# Run a single-brain inference
result = array.run_inference(b"my input data")
print(result)

# Run consensus inference across all four brains
consensus = array.run_consensus_inference(b"my input data")
print(consensus)

# Print an ASCII performance dashboard
print(array.dashboard())
```

---

## Configuration

Edit `quantum_quad_brain/config/array_config.yaml` to match your hardware:

```yaml
array:
  dram_capacity_gb: 8
  initial_cm4_nodes: 4
  scheduling_strategy: least_loaded

storage:
  nvme_devices:
    - id: nvme0
      path: /dev/nvme0n1
      capacity_gb: 2000
```

Load the config at runtime:

```python
import yaml
from pathlib import Path

config_path = Path("quantum_quad_brain/config/array_config.yaml")
with config_path.open() as f:
    config = yaml.safe_load(f)
```

---

## Hot-Plug Operations

```python
# Add an NVMe device at runtime
array.add_nvme_device("nvme1", "/dev/nvme1n1", capacity_gb=2000.0)

# Remove a failed NVMe device without stopping the array
array.remove_nvme_device("nvme1")

# Add a new CM4 node
array.add_cm4_node("cm4-004", "10.0.0.104", ram_gb=8.0)

# Remove a decommissioned node
array.remove_cm4_node("cm4-004")
```

---

## Workload Scheduling Strategies

| Strategy | Best For |
|----------|----------|
| `least_loaded` | Heterogeneous workloads; routes to the most free node |
| `round_robin` | Uniform workloads; ensures even distribution |
| `random` | Testing / benchmarking; avoids bias |

```python
from quantum_quad_brain.cm4_cluster.workload_distributor import SchedulingStrategy

array = ScalableComputeArray(scheduling_strategy=SchedulingStrategy.ROUND_ROBIN)
```

---

## Cache Tuning

```python
from quantum_quad_brain.nvme_dram.cache_manager import DRAMCacheManager

# Use LFU eviction for inference workloads with skewed access patterns
cache = DRAMCacheManager(capacity_gb=16.0, eviction_policy="lfu")
```

---

## Performance Monitoring

```python
from quantum_quad_brain.metrics.performance_monitor import PerformanceMonitor
from quantum_quad_brain.metrics.visualizer import MetricsVisualizer

monitor = PerformanceMonitor(window_size=300)
viz = MetricsVisualizer()

# Record a sample from the live system
monitor.record_from_system(array.quad_brain)

# Print the dashboard
print(viz.render_dashboard(array.quad_brain, monitor))

# Print sparkline trends
print(viz.render_sparklines(monitor))
```

---

## Fault Tolerance

```python
from quantum_quad_brain.array.fault_tolerance import FaultToleranceManager, FaultSeverity

ft = FaultToleranceManager(min_healthy_nvme=1, min_online_nodes=1)

# Register a custom alert handler
ft.register_handler(FaultSeverity.CRITICAL, lambda fault: print(f"ALERT: {fault.message}"))

# Run all checks
faults = ft.run_all_checks(array.quad_brain)
print(ft.incident_summary())
```

---

## Running Tests

```bash
pytest tests/ -v
```
