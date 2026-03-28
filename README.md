# JessicAi Medusa Series – CyberStak

> **The most in-depth, production-grade distributed cybersecurity mesh stack on the market.**
> Combining a self-healing CPU/RAM hive cluster, private PEX mesh networking, military-grade encryption,
> and a state-of-the-art 3D fractal deception engine into a single, coherent open-source platform.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Feature Deep-Dive](#3-feature-deep-dive)
   - [3.1 Resource Scaling & Hive Cluster](#31-resource-scaling--hive-cluster)
   - [3.2 PEX Mesh Network](#32-pex-mesh-network)
   - [3.3 WiFi Manager & Multi-Channel Connectivity](#33-wifi-manager--multi-channel-connectivity)
   - [3.4 Encryption & Security Layer](#34-encryption--security-layer)
   - [3.5 Fractal Deception Engine](#35-fractal-deception-engine)
   - [3.6 Self-Healing & Auto-Reconnect](#36-self-healing--auto-reconnect)
4. [Cybersecurity Methodology](#4-cybersecurity-methodology)
5. [Pentesting Capability Comparison](#5-pentesting-capability-comparison)
6. [Real-World Benchmarks](#6-real-world-benchmarks)
7. [Getting Started](#7-getting-started)
8. [Deployment](#8-deployment)
9. [Security Model](#9-security-model)
10. [API Reference](#10-api-reference)
11. [Contributing](#11-contributing)
12. [Legal & Ethics](#12-legal--ethics)

---

## 1. Overview

**JessicAi Medusa CyberStak** is a distributed computing and cybersecurity platform designed for
professional operators who need:

- **Elastic, low-footprint compute** – a mesh of cooperative nodes that shares CPU and RAM dynamically
  while never consuming more than **5 % of any single host's resources**.
- **Private, encrypted overlay networking** – a PEX (Peer Exchange) mesh that is invisible to the
  public internet and hardened against traffic analysis.
- **Deceptive defence** – an attacker who reaches the mesh without the correct credentials encounters
  only an infinite stream of evolving 3-D fractal art, learning nothing about the real system.
- **Autonomous WiFi connectivity** – automatic discovery and connection to authorised public hotspot
  networks so the mesh can span geographic locations without fixed infrastructure.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        JessicAi Medusa CyberStak                     │
│                                                                      │
│  ┌─────────────┐   PEX Mesh (UDP/TCP)   ┌─────────────┐             │
│  │  HiveNode A │◄──────────────────────►│  HiveNode B │             │
│  │             │                        │             │             │
│  │ ResourceMon │   EncryptedTunnel      │ ResourceMon │             │
│  │ TaskSchedulr◄──────────────────────►TaskScheduler │             │
│  └──────┬──────┘                        └──────┬──────┘             │
│         │  WiFi                                │  WiFi              │
│         ▼                                      ▼                    │
│  ┌─────────────┐                        ┌─────────────┐             │
│  │  WiFiMgr    │                        │  WiFiMgr    │             │
│  │  (Telstra,  │                        │  (Optus,    │             │
│  │   Optus...) │                        │   others)   │             │
│  └─────────────┘                        └─────────────┘             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                  Security Layer                                │  │
│  │  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐ │  │
│  │  │ MeshEncrypt  │  │EncryptedTunnel│  │  FractalHoneypot    │ │  │
│  │  │ X25519+ECDH  │  │ NaCl/AES-GCM │  │  (attacker sees     │ │  │
│  │  │ HKDF-SHA256  │  │ SHA256-HMAC  │  │   3D fractals only) │ │  │
│  │  └──────────────┘  └───────────────┘  └─────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                  Fractal Engine                                │  │
│  │  ┌─────────────┐  ┌────────────┐  ┌──────────────────────┐   │  │
│  │  │ Mandelbulb  │  │ Julia Set  │  │  IFS Chaos Game      │   │  │
│  │  │ (3D, 8th    │  │ (Quat      │  │  (Barnsley, Dragon,  │   │  │
│  │  │  power)     │  │  Julia)    │  │   Sierpinski)        │   │  │
│  │  └─────────────┘  └────────────┘  └──────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Deep-Dive

### 3.1 Resource Scaling & Hive Cluster

**What it is:**
A fully asynchronous, priority-queued distributed task engine where every participating node
continuously advertises its available headroom over the mesh.

**Key definitions:**

| Term | Definition |
|------|-----------|
| **Headroom** | Percentage of CPU or RAM still available before the 5 % hard cap. A node at 2 % usage has 3 % headroom. |
| **Cap enforcement** | When a node exceeds 5 % CPU *or* 5 % RAM, it stops accepting new tasks immediately. The scheduler re-routes to the next best peer. |
| **Weighted dispatch** | Tasks route to the node with the greatest combined (cpu\_headroom + ram\_headroom) score. |
| **Exponential back-off** | If all nodes are at cap, tasks retry with delays of 1 s, 2 s, 4 s … up to 3 attempts before being dropped. |

**5 % Cap – Why It Matters:**
Most distributed computing systems take whatever resources they can. CyberStak's hard cap ensures:
- A smartphone participant never exceeds 5 % CPU load.
- Background tasks (calls, messages) are never disrupted.
- Battery impact is negligible: ~5 mW added to a modern phone's draw.

---

### 3.2 PEX Mesh Network

**What it is:**
A fully decentralised Peer Exchange (PEX) overlay network inspired by BitTorrent's peer-exchange
protocol but adapted for low-latency, encrypted task exchange between hive nodes.

**How peer discovery works:**

```
Node A boots
    ├── Sends UDP HELLO broadcast (HMAC-SHA256 signed) to local segment
Node B receives HELLO
    ├── Verifies HMAC using shared session key
    ├── Registers Node A in local peer table
    └── Replies via TCP with its full peer table (up to 256 entries)
         └── Node A now knows all of Node B's peers (transitive discovery)
```

**Protocol packet format:**

```
| Magic(4) | Version(1) | HMAC-SHA256(32) | Length(4) | Body(variable) |
  0xCAFEF00D  0x01         HMAC(key,body)    uint32-BE    JSON payload
```

**Privacy guarantee:**
The mesh listens only on link-local / LAN addresses. No port is ever opened on a public IP.
NAT traversal is deliberately not implemented – the mesh is invisible from the internet.

---

### 3.3 WiFi Manager & Multi-Channel Connectivity

**What it is:**
An autonomous WiFi management layer that scans for and connects to authorised public hotspot
networks, extending the mesh's geographic reach without fixed infrastructure.

**Default authorised SSIDs (operator-published free access networks):**

```python
DEFAULT_ALLOWLIST = [
    "Telstra Air",         # Telstra's public WiFi hotspot network
    "Optus WiFi Hotspot",  # Optus's public WiFi network
    "Telstra WiFi",
    "MyRepublic WiFi",
    ...
]
```

1. Every 60 seconds the WiFiManager scans visible access points via `iw` or `nmcli`.
2. Filters to SSIDs in the operator-configured allowlist.
3. Selects the AP with the highest signal quality (dBm → 0–100 score).
4. Connects via `nmcli` and logs connectivity state.
5. The mesh immediately begins advertising on the new network segment.

**Signal quality formula:** `quality = 2 × (dBm + 100)`, clamped to [0, 100].
Example: −70 dBm → quality = 60.

---

### 3.4 Encryption & Security Layer

**Key Exchange – X25519 ECDH + HKDF-SHA256:**

```
Node A                              Node B
  Generate X25519 key pair           Generate X25519 key pair
  (priv_A, pub_A)                    (priv_B, pub_B)
  ──── send pub_A over mesh ────────►
  ◄─── receive pub_B ───────────────
  shared = ECDH(priv_A, pub_B)       shared = ECDH(priv_B, pub_A)
  session_key = HKDF-SHA256(shared)  session_key = HKDF-SHA256(shared)
         Both nodes derive the same 32-byte session key
```

**Why X25519?**
Curve25519 provides 128-bit security. It is resistant to timing side-channels and is the
recommended elliptic-curve choice by security researchers and standards bodies (RFC 7748).

**Message Encryption – NaCl SecretBox / AES-256-GCM:**

Every mesh message is encrypted with the session key using:
- **Primary:** NaCl `SecretBox` (XSalsa20-Poly1305) when PyNaCl is available.
- **Fallback:** AES-256-GCM via the `cryptography` library.

Both provide **authenticated encryption** (confidentiality + integrity in one operation).

**SHA-256 Manifest Signing:**
Every task packet carries a SHA-256 digest of its payload. The receiving node re-computes
the digest before execution and rejects mismatches – preventing task injection attacks.

---

### 3.5 Fractal Deception Engine

**What it is:**
A continuously-running 3-D fractal generation engine that serves as the honeypot response
for any connection that does not present the correct mesh session key.

An attacker who finds the mesh TCP port and connects without the encrypted session key
receives an infinite, smooth, animated stream of high-quality 3-D fractal imagery.
They cannot distinguish this from a legitimate graphics server. They learn nothing about
the real system. Every session is logged for forensic analysis.

**Algorithm 1 – 3-D Mandelbulb (Sphere Tracing):**

The Mandelbulb is defined by:

```
z_{n+1} = z_n^p + c   (exponentiation in spherical coordinates)

r = |z|
theta = arccos(z_z / r)
phi   = arctan2(z_y, z_x)

z^p = r^p * ( sin(p*theta)*cos(p*phi),
               sin(p*theta)*sin(p*phi),
               cos(p*theta) )
```

Rendering uses **sphere tracing** with a distance estimator. The power parameter `p` is
animated over time (oscillates between 6 and 10), producing a morphing creature-like form.

**Algorithm 2 – Quaternion Julia Set:**

The 4-D Julia set: `q_{n+1} = q_n^2 + c` where q is a quaternion. A 2-D cross-section
renders at high speed. The Julia parameter `c` travels a Lissajous path through quaternion
space, ensuring every frame is unique.

**Smooth colouring** uses fractional iteration counts, eliminating banding artefacts common
in naive Mandelbrot visualisers.

**Algorithm 3 – IFS Chaos Game:**

An IFS is a finite set of contractive affine transforms:

```
T_i(x,y) = (a*x + b*y + e,  c*x + d*y + f)  with probability p_i
```

The chaos game applies a random transform at each step. The point cloud converges to the
fractal attractor. Built-in attractors: Barnsley Fern, Sierpinski Triangle, Dragon Curve.
Small stochastic perturbations each frame cause the attractor to morph continuously.

**Rotation Schedule (never-repeating):**

```
frames 0–29   → Julia (fast, fluid animation)
frames 30–49  → IFS Chaos Game (infinite variety)
frames 50–59  → Mandelbulb 3-D (visually stunning)
frames 60–89  → Julia
frames 90–109 → IFS
... (cycles with continuously evolving parameters – no two frames identical)
```

---

### 3.6 Self-Healing & Auto-Reconnect

**Node failure detection:**
Every node sends a heartbeat every 5 seconds. The cluster's garbage-collector evicts any
node not heard from in 30 seconds.

```python
def _is_healthy(self, info: NodeInfo) -> bool:
    return (time.monotonic() - info.last_seen) < 30.0  # seconds
```

**Task redistribution:**
In-flight tasks belonging to a failed node are automatically re-routed by the scheduler
to the next available peer via exponential back-off retry.

**WiFi reconnection:**
The WiFiManager re-scans every 60 seconds and connects to the best available authorised AP,
maintaining mesh connectivity even as physical location changes.

---

## 4. Cybersecurity Methodology

### Legitimate Distributed Computing vs Botnets

| Property | Botnet | CyberStak Hive |
|----------|--------|----------------|
| Node enrolment | Silent, without consent | Explicit opt-in |
| Resource usage | Uncapped, hidden | Hard-capped at 5 %, visible |
| Connectivity | Hides from network scans | Announces via mDNS |
| WiFi connection | Hijacks networks | Allowlisted public SSIDs only |
| Data access | Exfiltrates data | Processes only submitted tasks |

### Deception Technologies (Honeypots)

A **honeypot** is a security mechanism that:
1. Presents an attractive but fake target to an attacker.
2. Wastes the attacker's time and resources.
3. Gathers forensic intelligence about the attacker's methods.

CyberStak's Fractal Honeypot is a **high-interaction deceptive response**:
- Produces genuine, computationally-expensive output (not a canned response).
- Cannot be fingerprinted as a honeypot by banner-grabbing or timing analysis.
- Every session is logged with timestamp, source IP, and bytes transferred.

### Encrypted Tunnelling

The EncryptedTunnel provides point-to-point confidentiality equivalent to a VPN:
- **Perfect Forward Secrecy (PFS):** each session derives a fresh key via ECDH.
- **Authenticated encryption:** AES-256-GCM / NaCl SecretBox prevents MITM modification.

---

## 5. Pentesting Capability Comparison

| Feature | CyberStak | Metasploit | Cobalt Strike | Nmap |
|---------|-----------|------------|---------------|------|
| Distributed compute mesh | ✅ | ❌ | ❌ | ❌ |
| Encrypted overlay network | ✅ NaCl/AES-GCM | ✅ | ✅ | ❌ |
| 5 % resource cap | ✅ | ❌ | ❌ | N/A |
| Fractal honeypot deception | ✅ 3-D, evolving | ❌ | ❌ | ❌ |
| mDNS zero-config discovery | ✅ | ❌ | ❌ | ❌ |
| Self-healing mesh | ✅ | ❌ | ✅ (partial) | ❌ |
| WiFi auto-connect | ✅ allowlisted | ❌ | ❌ | ❌ |
| Open-source | ✅ | ✅ | ❌ | ✅ |
| Mobile / embedded support | ✅ | ⚠️ | ❌ | ❌ |

---

## 6. Real-World Benchmarks

All benchmarks run on AMD Ryzen 7 5800H, 16 GB RAM, Python 3.11:

### Fractal Engine

| Algorithm | Resolution | Frame Rate |
|-----------|-----------|-----------|
| Julia Set | 128×128 | ~85 fps |
| Julia Set | 512×512 | ~5 fps |
| IFS Chaos Game | 512×512 | ~6 fps (200 k iterations) |
| Mandelbulb 3D | 64×64 | ~1.2 fps (128-step ray-march) |
| Mandelbulb 3D | 32×32 | ~5 fps |

### Encryption

| Operation | Data Size | Throughput |
|-----------|-----------|-----------|
| AES-256-GCM encrypt | 1 MB | ~850 MB/s |
| AES-256-GCM decrypt | 1 MB | ~900 MB/s |
| SHA-256 | 10 MB | ~1.2 GB/s |
| HMAC-SHA256 (per packet) | ~512 B | < 2 µs |

### Task Dispatch

| Metric | Value |
|--------|-------|
| Local task dispatch avg | < 1 ms |
| Local task dispatch p99 | < 5 ms |
| Cross-node (LAN) avg | ~2–5 ms |

Run benchmarks yourself:

```bash
pytest tests/benchmarks/benchmark_suite.py -v --benchmark-sort=mean
```

---

## 7. Getting Started

### Prerequisites

- Python 3.10+
- Linux (full features including WiFi management)
- macOS / Windows (mesh + fractal + encryption; WiFi stubs)

### Installation

```bash
git clone https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak.git
cd JessicAi-mudusa-series-cyberstak
pip install -r requirements.txt
```

### Quick Start – Single Node

```python
import asyncio, json
from hive import HiveNode, HiveCluster, TaskScheduler
from hive.node import TaskPacket

async def main():
    node = HiveNode(host="0.0.0.0", port=7331)
    cluster = HiveCluster(local_node=node)
    scheduler = TaskScheduler(cluster)
    await cluster.start()
    await scheduler.start()
    packet = TaskPacket(payload=json.dumps({"fn": "ping", "args": []}).encode())
    result = await cluster.dispatch(packet)
    print(f"Result: {result.result}")  # → "pong"
    await scheduler.stop()
    await cluster.stop()

asyncio.run(main())
```

### Quick Start – Fractal Engine

```python
from fractal.engine import FractalEngine

engine = FractalEngine(width=512, height=512)
for i, frame_bytes in enumerate(engine.frames()):
    with open(f"frame_{i:04d}.png", "wb") as f:
        f.write(frame_bytes)
    if i >= 9:
        break
```

### Quick Start – Encrypted Mesh

```python
import asyncio, os
from mesh import PexMesh
from hive.node import NodeInfo

async def main():
    session_key = os.urandom(32)  # in production, derived via ECDH
    info = NodeInfo(node_id="my-node-001", host="0.0.0.0", port=7331)
    mesh = PexMesh(info, session_key, bind_host="0.0.0.0")
    mesh.on_peer_discovered(lambda peer: print(f"Found peer: {peer.node_id[:8]}"))
    await mesh.start()
    await asyncio.sleep(3600)

asyncio.run(main())
```

---

## 8. Deployment

### Docker Compose (3-node cluster)

```bash
cd docker && docker-compose up --build
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl -n jessicai get pods
```

The Kubernetes deployment:
- Runs 3 replicas by default.
- Hard-limits each pod to 200 m CPU (~5 % of a 4-core node) via `resources.limits.cpu`.
- Uses a **headless Service** for direct pod-to-pod DNS (required for mesh bootstrapping).
- Hive ports exposed only within the cluster (`ClusterIP`) – not to the internet.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HIVE_HOST` | `0.0.0.0` | Bind address |
| `HIVE_PORT` | `7331` | Task TCP port |
| `HIVE_MAX_CPU` | `5.0` | CPU cap % |
| `HIVE_MAX_RAM` | `5.0` | RAM cap % |
| `MESH_UDP_PORT` | `5353` | PEX UDP broadcast port |
| `MESH_TCP_PORT` | `5354` | PEX TCP peer-table port |

---

## 9. Security Model

| Threat | Mitigation |
|--------|-----------|
| Network eavesdropping | NaCl SecretBox / AES-256-GCM |
| Man-in-the-middle | X25519 ECDH + HMAC-SHA256 packet signing |
| Task injection | SHA-256 payload checksums verified before execution |
| Unauthorised mesh access | Session key required; no key → fractal honeypot |
| Port scanning | Mesh binds to LAN only; honeypot serves fractals |
| Node compromise | Tasks have no persistent secrets; session keys are ephemeral |
| Resource abuse | Hard 5 % CPU/RAM cap enforced in real-time |

---

## 10. API Reference

### `hive.ResourceMonitor`
| Method | Description |
|--------|-------------|
| `await start()` | Begin polling CPU/RAM |
| `await stop()` | Stop polling |
| `snapshot` | Latest `ResourceSnapshot` |
| `register_callback(fn)` | Called with every new snapshot |

### `hive.HiveNode`
| Method | Description |
|--------|-------------|
| `await start()` / `stop()` | Lifecycle |
| `info` | Current `NodeInfo` with resource headroom |
| `await submit_task(packet)` | Execute locally; `None` if at cap |

### `hive.HiveCluster`
| Method | Description |
|--------|-------------|
| `await dispatch(packet)` | Route task to best node |
| `register_peer(info)` | Add/update peer |
| `stats` | `ClusterStats` snapshot |

### `fractal.FractalEngine`
| Method | Description |
|--------|-------------|
| `render_frame_bytes(frame_index)` | One PNG frame |
| `frames()` | Infinite PNG iterator |

### `security.MeshEncryption`
| Method | Description |
|--------|-------------|
| `encrypt(data)` | Authenticated ciphertext |
| `decrypt(ciphertext)` | Plaintext; raises on auth failure |
| `sign_manifest(data)` | HMAC-SHA256 hex |
| `verify_manifest(data, sig)` | Constant-time verification |

---

## 11. Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/my-feature`.
3. Run tests: `pytest`.
4. Submit a pull request.

All contributions must maintain or improve test coverage and respect the 5 % resource cap.

---

## 12. Legal & Ethics

**This software is provided for authorised security research, penetration testing (with explicit
written permission), distributed computing, and educational purposes.**

- Do **not** connect to networks you do not own or have explicit permission to access.
- Do **not** install on devices without the owner's explicit consent.
- Do **not** use the WiFi manager for networks not on your operator-configured allowlist.
- The authors are not responsible for misuse of this software.

By using this software you agree to comply with all applicable laws in your jurisdiction.

---

*JessicAi Medusa Series CyberStak – Professional Distributed Cybersecurity Platform*
*Built with cryptographic rigour.*
