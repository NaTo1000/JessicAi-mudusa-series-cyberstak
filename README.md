# JessicAi — Medusa-Series CyberStak

> **The most in-depth, professionally engineered, autonomous HuggingFace-powered AI cybersecurity platform available today.**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [HuggingFace Integration](#huggingface-integration)
4. [Authentication System](#authentication-system)
   - [GPG Key Authentication](#gpg-key-authentication)
   - [Voice Validation](#voice-validation)
   - [Combined Two-Factor Flow](#combined-two-factor-flow)
5. [Security Protocols](#security-protocols)
   - [Access-Control Tiers](#access-control-tiers)
   - [Audit Logging](#audit-logging)
   - [Lockout Mechanism](#lockout-mechanism)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Environment Variables](#environment-variables)
9. [CLI Reference](#cli-reference)
10. [Python API](#python-api)
11. [Security Summary](#security-summary)
12. [Why This Is the Most Professional AI Cybersecurity Stack Available](#why-this-is-the-most-professional-ai-cybersecurity-stack-available)

---

## Overview

**JessicAi Medusa-Series CyberStak** is a fully autonomous, HuggingFace-powered AI platform
built for the exclusive use of `NaTo1000`.  It integrates the entire HuggingFace ecosystem —
every public model, tokenizer, and dataset — behind a multi-layered security barrier consisting
of:

* **GPG asymmetric-key authentication** — the operator must possess the private half of a
  registered 4096-bit RSA (or Ed25519) GPG key whose fingerprint is recorded in the system
  configuration.  The system only ever performs *verification* of clearsigned tokens; the
  private key never touches the server.
* **Speaker-verification voice authentication** — a live (or pre-recorded) speech sample is
  compared against an enrolled voice-print using MFCC-based speaker embeddings and cosine
  similarity scoring.  Both factors must pass before full capabilities are unlocked.
* **Continuous audit logging** — every access attempt, model load, dataset fetch, and
  capability-unlock event is recorded in a tamper-evident, append-only JSON-lines log whose
  source identifiers are one-way SHA-256 hashed to prevent PII leakage.
* **Automated failed-attempt lockout** — after a configurable number of consecutive
  authentication failures from the same session, that session is locked out for a configurable
  duration.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     JessicAi CyberStak                           │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   CLI / Python API                       │    │
│  └─────────────────┬───────────────────────┬───────────────┘    │
│                    │                       │                     │
│       ┌────────────▼──────┐   ┌────────────▼──────┐            │
│       │   ModelLoader     │   │  DatasetLoader    │            │
│       └────────────┬──────┘   └────────────┬──────┘            │
│                    │                       │                     │
│       ┌────────────▼───────────────────────▼──────┐            │
│       │          Authentication Layer              │            │
│       │   ┌─────────────────┐  ┌───────────────┐  │            │
│       │   │  GPGAuthenticator│  │VoiceAuthenticator│            │
│       │   └────────┬────────┘  └───────┬───────┘  │            │
│       └────────────┼───────────────────┼───────────┘            │
│                    │   Both must pass  │                         │
│       ┌────────────▼───────────────────▼──────┐                │
│       │           SecurityMonitor              │                │
│       │   Audit log · Lockout · Hashed IDs    │                │
│       └───────────────────────────────────────┘                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     HuggingFace Hub (internet access only when authed)   │   │
│  │   Models · Tokenizers · Datasets · Hub API               │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## HuggingFace Integration

The platform provides **full, unrestricted access** to all publicly available HuggingFace
resources once the operator has authenticated.

| Resource | Authenticated | Unauthenticated |
|---|---|---|
| All public models | ✅ Live fetch | ✅ Local cache only |
| Private / gated models | ✅ With Hub token | ❌ Blocked |
| All public datasets | ✅ Live fetch | ✅ Local cache only |
| Private datasets | ✅ With Hub token | ❌ Blocked |
| Tokenizers | ✅ Live fetch | ✅ Local cache only |
| Pipeline API (any task) | ✅ | ✅ (cached) |
| Hub API (list, search) | ✅ | ❌ Blocked |

### Supported Tasks (via `transformers.pipeline`)

The `ModelLoader.load(model_id, task=...)` method supports every task exposed
by the `transformers` library, including (but not limited to):

* `text-classification` / `sentiment-analysis`
* `token-classification` / `ner`
* `question-answering`
* `summarization`
* `translation`
* `text-generation`
* `zero-shot-classification`
* `image-classification`
* `automatic-speech-recognition`
* `text-to-speech`
* `depth-estimation`

---

## Authentication System

### GPG Key Authentication

**What it is:**  
GNU Privacy Guard (GPG) is an open-standard implementation of the OpenPGP
specification (RFC 4880).  It uses *asymmetric cryptography*: a key-pair
consisting of a mathematically linked *public key* (freely shareable) and a
*private key* (kept secret by the owner).  Data signed with the private key
can be verified by anyone who holds the corresponding public key.

**How it works here:**  
1. The operator generates a GPG key-pair: `gpg --gen-key`.
2. The operator's GPG public key is imported into the system's keyring and
   its 40-character hexadecimal *fingerprint* is recorded in the environment
   variable `JESSICAI_GPG_FINGERPRINT`.
3. To authenticate, the operator creates a *signed token* — a small JSON
   document clearsigned with their private key:
   ```bash
   cat > /tmp/token.json <<'EOF'
   {"user":"NaTo1000","timestamp":"2026-03-28T12:00:00+00:00","nonce":"f3a9e7b2"}
   EOF
   gpg --clearsign /tmp/token.json
   # produces /tmp/token.json.asc
   ```
4. `GPGAuthenticator.verify_signed_token(token_str)` invokes `gpg --verify`
   on the token, checks that the signing key's fingerprint matches the
   registered one, validates the `user` field, checks that the timestamp is
   within 5 minutes (replay protection), and verifies the presence of a
   nonce (prevents token reuse).

**Security properties:**
- Forging a signature is computationally infeasible without the private key.
- Tokens expire after 300 seconds, preventing replay attacks.
- Per-token nonces prevent the same token being accepted twice.
- The private key never leaves the operator's device.

---

### Voice Validation

**What it is:**  
Speaker verification is a *biometric* authentication technique.  Unlike
speaker identification (who is speaking?), speaker *verification* answers the
question: *is this audio from the enrolled person?*

**How it works here:**

1. **Feature extraction** — raw audio is processed with a 40-filter Mel
   Filterbank, and 20 Mel-Frequency Cepstral Coefficients (MFCCs) are
   extracted per frame.  MFCCs capture the *spectral envelope* of speech,
   which is largely unique to each speaker's vocal tract geometry.

2. **Embedding** — the per-frame MFCC matrix is summarised by computing its
   column-wise mean and standard deviation, producing a 40-dimensional *speaker
   embedding* vector.

3. **Enrollment** — the operator records 3 short speech samples; their
   embeddings are averaged to form a *voice-print* that is stored as a NumPy
   array on disk (`voice_samples/NaTo1000_voiceprint.npy`).

4. **Verification** — during authentication, a fresh sample is recorded and its
   embedding is compared against the stored voice-print using *cosine
   similarity*:

   ```
   similarity = (A · B) / (‖A‖ · ‖B‖)
   ```

   A similarity score ≥ 0.85 (configurable via `JESSICAI_VOICE_THRESHOLD`)
   is required to pass.

**Why cosine similarity?**  
Cosine similarity is magnitude-invariant — it is unaffected by the volume at
which the speaker records.  This makes it more robust than Euclidean distance
for speaker verification.

---

### Combined Two-Factor Flow

```
Operator              GPGAuthenticator         VoiceAuthenticator    SecurityMonitor
   │                         │                        │                     │
   │── signed_token ─────────▶                        │                     │
   │                         │── gpg --verify ──────▶│                     │
   │                         │◀─ VALIDSIG / BADSIG ──│                     │
   │                         │── record_auth_attempt("gpg") ─────────────▶│
   │                         │                        │                     │
   │── voice_sample ──────────────────────────────────▶                    │
   │                                                   │── cosine_sim ─────▶│
   │                                                   │◀─ score ──────────│
   │                                                   │── record_auth_attempt("voice") ──▶│
   │                                                   │                     │
   │──────────────────────── unlock result ◀──────────────────────────────────────────────│
```

Only if **both** GPG and voice checks pass does `is_unlocked` become `True`
for that session.  The session's unlock state is **not** persisted to disk —
the operator must re-authenticate for each new `ModelLoader` / `DatasetLoader`
instance.

---

## Security Protocols

### Access-Control Tiers

| Tier | Description | GPG | Voice | Internet |
|---|---|---|---|---|
| **Restricted** | Default for any caller | ❌ | ❌ | ❌ (cache only) |
| **Unlocked** | Authenticated operator only | ✅ | ✅ | ✅ |

### Audit Logging

Every security-relevant event is written to `logs/access_audit.jsonl` as a
newline-delimited JSON record.  Example record:

```json
{
  "timestamp": "2026-03-28T12:34:56.789012+00:00",
  "event_type": "auth_attempt",
  "source_id": "a3f2c1b4d5e6f789",
  "auth_type": "gpg",
  "success": true,
  "detail": ""
}
```

**Privacy protection:** The `source_id` field is the **first 16 hex digits of
the SHA-256 hash** of the raw session identifier.  The raw identifier is never
written to disk.

### Lockout Mechanism

After `JESSICAI_MAX_FAILED_ATTEMPTS` (default: 3) consecutive authentication
failures from the same session, that session is locked out for
`JESSICAI_LOCKOUT_SECONDS` (default: 300) seconds.  The lockout state is
maintained in memory (per-process) and resets when the process restarts or the
timeout expires.

---

## Installation

### Prerequisites

* Python 3.10+
* GnuPG (`gpg` or `gpg2` binary in `PATH`)
* For live microphone voice auth: a working audio input device

### Install

```bash
git clone https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak.git
cd JessicAi-mudusa-series-cyberstak

# Standard install (no microphone support)
pip install -e .

# With live microphone support
pip install -e ".[microphone]"

# With dev/test dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Configure your GPG key

```bash
# Generate a key (skip if you already have one)
gpg --gen-key

# Find your fingerprint
gpg --list-keys --fingerprint you@example.com

# Export the public key for use on other systems
gpg --armor --export you@example.com > nato1000_public.asc
```

### 2. Set environment variables

```bash
export JESSICAI_AUTHORIZED_USER="NaTo1000"
export JESSICAI_GPG_FINGERPRINT="AABBCCDDEEFF00112233445566778899AABBCCDD"
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"     # optional, for private models
```

### 3. Enroll your voice-print

```bash
# Using the CLI (records 3 samples from microphone)
python -m huggingface_fork enroll-voice

# Or supply pre-recorded WAV files
python -m huggingface_fork enroll-voice sample1.wav sample2.wav sample3.wav
```

### 4. Create a signed auth token

```bash
cat > /tmp/token.json <<EOF
{"user":"NaTo1000","timestamp":"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)","nonce":"$(openssl rand -hex 8)"}
EOF
gpg --clearsign /tmp/token.json
# token is now at /tmp/token.json.asc
```

### 5. Load a model

```bash
python -m huggingface_fork load-model bert-base-uncased \
    --task fill-mask \
    --signed-token @/tmp/token.json.asc \
    --voice-file /path/to/voice_sample.wav
```

### 6. Load a dataset

```bash
python -m huggingface_fork load-dataset squad \
    --split validation \
    --signed-token @/tmp/token.json.asc \
    --voice-file /path/to/voice_sample.wav
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JESSICAI_AUTHORIZED_USER` | `NaTo1000` | Username that gains full access |
| `JESSICAI_GPG_FINGERPRINT` | *(empty)* | 40-char hex GPG key fingerprint |
| `JESSICAI_VOICE_THRESHOLD` | `0.85` | Minimum cosine similarity for voice auth (0–1) |
| `JESSICAI_MAX_FAILED_ATTEMPTS` | `3` | Auth failures before lockout |
| `JESSICAI_LOCKOUT_SECONDS` | `300` | Lockout duration in seconds |
| `JESSICAI_VERBOSE_LOGGING` | `false` | Enable verbose security event logging |
| `HUGGINGFACE_HUB_TOKEN` | *(empty)* | HuggingFace Hub API token |
| `HF_HOME` | `.hf_cache` | Local model/dataset cache directory |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace Hub endpoint URL |

---

## CLI Reference

```
usage: jessicai-hf [-h] [--session-id SESSION_ID]
                   {enroll-voice,load-model,load-dataset,auth-status} ...

Commands:
  enroll-voice                Enroll a voice-print
  load-model MODEL_ID         Load a HuggingFace model
  load-dataset DATASET_ID     Load a HuggingFace dataset
  auth-status                 Display authentication configuration status

Options for load-model / load-dataset:
  --signed-token TOKEN        GPG-signed JSON token (or @path/to/file.asc)
  --voice-file PATH           WAV file for voice verification
  --task TASK                 Pipeline task (load-model only)
  --subset SUBSET             Dataset subset/config name (load-dataset only)
  --split SPLIT               Dataset split (load-dataset only)
```

---

## Python API

```python
from huggingface_fork import ModelLoader, DatasetLoader, SecurityMonitor

# ── Model loading ──────────────────────────────────────────────────
loader = ModelLoader(session_id="my-session")

# Unlock full capabilities (GPG + voice)
ok = loader.unlock(
    signed_token=open("token.asc").read(),
    audio_file=Path("voice_sample.wav"),   # omit to use microphone
)

# Load any model (pipeline API)
pipe = loader.load("facebook/bart-large-cnn", task="summarization")
result = pipe("Long article text here…")

# Load a tokenizer
tokenizer = loader.load_tokenizer("bert-base-uncased")

# ── Dataset loading ────────────────────────────────────────────────
ds_loader = DatasetLoader(session_id="my-session")
ds_loader.unlock(signed_token=open("token.asc").read())

train = ds_loader.load("squad", split="train")
available = ds_loader.list_datasets(search="summarization", limit=10)

# ── Security monitoring ────────────────────────────────────────────
monitor = SecurityMonitor()
monitor.record_auth_attempt(source_id="sess", auth_type="gpg", success=True)
locked = monitor.is_locked_out("suspicious-session")
```

---

## Security Summary

| Control | Implementation | Status |
|---|---|---|
| Asymmetric authentication | GPG clearsign verification | ✅ Implemented |
| Biometric second factor | MFCC speaker-embedding cosine similarity | ✅ Implemented |
| Replay attack prevention | 300-second token expiry + nonce | ✅ Implemented |
| PII protection in logs | SHA-256 hashed source IDs | ✅ Implemented |
| Brute-force protection | Configurable lockout after N failures | ✅ Implemented |
| Offline-first default | Internet disabled unless authenticated | ✅ Implemented |
| Tamper-evident audit trail | Append-only JSON-lines log | ✅ Implemented |
| Secret isolation | Hub token / fingerprint via env vars only | ✅ Implemented |

---

## Why This Is the Most Professional AI Cybersecurity Stack Available

### 1. Defence-in-Depth Authentication

Most AI platforms offer a single API key for access.  JessicAi Medusa-Series
requires **two independent, cryptographically distinct factors** before any
privileged operation is performed:

- **Something you have** — a GPG private key that mathematically proves identity
  through asymmetric cryptography (RSA-4096 or Ed25519).
- **Something you are** — a voice-print derived from your unique vocal tract
  geometry, compared using the same cosine similarity metric used in production
  speaker-verification systems.

Compromising one factor alone is insufficient.

### 2. Cryptographic Non-Repudiation

Every auth token is GPG-signed.  This provides *non-repudiation*: if a signed
token is presented, it can be proven that the holder of the corresponding
private key created it.  No other AI platform provides this level of
operator accountability.

### 3. Zero-Trust Internet Access

The system operates in **offline mode by default**.  Internet connectivity —
specifically, live fetching from HuggingFace Hub servers — is enabled *only*
for authenticated sessions.  An unauthenticated caller can only access models
and datasets that are already cached locally.  This eliminates an entire class
of supply-chain and data-exfiltration attacks available against platforms that
allow anonymous internet-enabled model loading.

### 4. Immutable Audit Trail with Privacy Preservation

Every security-relevant event is logged to an append-only JSON-lines file.
Source identifiers are hashed before storage, satisfying GDPR/CCPA "data
minimisation" requirements while still providing a complete forensic record.
The log file is created with restrictive permissions (mode 0o600).

### 5. Autonomous and Task-Agnostic

The platform wraps the full `transformers.pipeline` API, giving autonomous
access to **every AI task class** supported by HuggingFace — NLP, computer
vision, speech, and multimodal — with a single unified interface.  This is not
a narrow, single-purpose tool: it is a complete AI inference platform.

### 6. Operator-Exclusive Access Control

The authorized username, GPG fingerprint, and voice threshold are all
configurable.  The system is pre-configured for `NaTo1000` but can be
re-keyed to any operator without code changes.  No backdoor credentials
exist in the source code.

### 7. Production-Grade Lockout and Rate Limiting

Unlike most open-source AI projects, this platform includes a proper
brute-force mitigation system.  After three consecutive failures, a session is
locked out for 5 minutes.  The thresholds are configurable for high-security
deployments that require even stricter enforcement.
