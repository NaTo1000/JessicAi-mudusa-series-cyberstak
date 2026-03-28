# JessicaAi Mudusa – Quantum-Inspired Neural Mesh Transformer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-NaTo1000%2Fjessicai--mudusa-yellow)](https://huggingface.co/NaTo1000/jessicai-mudusa)

> **JessicaAi Mudusa** is a full-stack, quantum-inspired large language model
> built for the Hugging Face ecosystem.  It combines a triple-channel neural
> mesh architecture, synaptic plasticity memory, and quantum-phase attention
> into a single, cohesive causal language model that is ready for
> pre-training, fine-tuning, and real-time inference.

---

## ✨ Architecture Highlights

| Component | Description |
|-----------|-------------|
| **Quantum-Phase Attention (QPA)** | Grouped-query attention whose softmax scores are modulated by learnable quantum-phase angles, enabling superposition-like multi-head information mixing |
| **Triple Neural Mesh** | Three parallel transformer stacks — *Cortical* (logic), *Limbic* (context), *Quantum* (creativity) — fused by a learned gating network |
| **Synaptic Plasticity Memory** | A fixed-size key–value memory bank implementing Hebbian-style associative recall across turns |
| **RoPE Embeddings** | Rotary Position Embeddings for efficient long-context handling (up to 32 768 tokens) |
| **SwiGLU FFN** | Gated feed-forward layers for improved expressivity |
| **GQA** | Grouped-Query Attention for efficient KV memory usage |

---

## 🚀 Quickstart

### Install

```bash
pip install jessicai-mudusa
# or from source:
git clone https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak.git
cd JessicAi-mudusa-series-cyberstak
pip install -e ".[serve,gradio,train]"
```

### Load from Hugging Face Hub

```python
from jessicai_mudusa import MudusaPipeline

pipe = MudusaPipeline.from_pretrained("NaTo1000/jessicai-mudusa")

# Single prompt
print(pipe("Explain quantum neural networks."))

# Multi-turn chat
messages = [{"role": "user", "content": "Who are you?"}]
print(pipe.chat(messages))

# Streaming
for token in pipe.stream("Write a haiku about the cosmos."):
    print(token, end="", flush=True)
print()
```

---

## 📁 Repository Structure

```
JessicAi-mudusa-series-cyberstak/
├── jessicai_mudusa/            # Core Python package
│   ├── __init__.py             # Public API
│   ├── configuration_mudusa.py # MudusaConfig (HF PretrainedConfig)
│   ├── modeling_mudusa.py      # Full model architecture
│   ├── tokenization_mudusa.py  # MudusaTokenizer (Qwen2 backbone)
│   └── pipeline_mudusa.py      # MudusaPipeline (high-level inference)
├── training/
│   ├── pretrain.py             # Pre-training script
│   ├── finetune.py             # SFT / LoRA fine-tuning script
│   └── config/
│       ├── pretrain_config.yaml
│       └── finetune_config.yaml
├── inference/
│   ├── run_inference.py        # CLI inference tool (REPL + single shot)
│   └── serve.py                # FastAPI REST server (OpenAI-compatible)
├── examples/
│   ├── basic_usage.py          # Minimal forward-pass demo
│   ├── conversational_ai.py    # Multi-turn chat demo
│   └── creative_assistance.py  # Generation parameter guide
├── docs/
│   ├── setup.md                # Installation guide
│   ├── training.md             # Training guide
│   └── deployment.md           # Deployment guide
├── tests/
│   └── test_mudusa.py          # Full unit test suite (pytest)
├── app.py                      # Hugging Face Spaces (Gradio UI)
├── requirements.txt
└── setup.py
```

---

## 🧠 Model Configuration

```python
from jessicai_mudusa import MudusaConfig

config = MudusaConfig(
    vocab_size=151936,          # Qwen2 vocabulary
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=8,      # GQA – 2:1 KV sharing
    max_position_embeddings=32768,
    num_mesh_channels=3,        # cortical / limbic / quantum
    quantum_phase_dim=64,       # learnable phase angle vectors
    synaptic_memory_size=512,   # knowledge-graph memory slots
    mesh_fusion_type="gated",   # "gated" | "mean" | "concat"
)
```

---

## 🏋️ Training

### Pre-training from scratch

```bash
python training/pretrain.py --config training/config/pretrain_config.yaml
# or multi-GPU:
accelerate launch training/pretrain.py --config training/config/pretrain_config.yaml
```

### Fine-tuning with LoRA

```bash
python training/finetune.py --config training/config/finetune_config.yaml
```

See [docs/training.md](docs/training.md) for full details.

---

## 🌐 Serving

### FastAPI REST server (OpenAI-compatible)

```bash
python inference/serve.py --model NaTo1000/jessicai-mudusa --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
```

### Interactive CLI

```bash
python inference/run_inference.py --model NaTo1000/jessicai-mudusa --stream
```

See [docs/deployment.md](docs/deployment.md) for Docker, Spaces, and library usage.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

27 tests covering all major components (config, attention, FFN, mesh, memory, model, tokenizer).

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| [docs/setup.md](docs/setup.md) | Installation, environment setup, GPU configuration |
| [docs/training.md](docs/training.md) | Pre-training, fine-tuning, weight merging, Hub publishing |
| [docs/deployment.md](docs/deployment.md) | Spaces, FastAPI, Docker, library usage, API reference |

---

## 🤗 Hugging Face Integration

The model is fully compatible with:

- `transformers.AutoModelForCausalLM`
- `transformers.Trainer` and `trl.SFTTrainer`
- `peft` LoRA / QLoRA adapters
- `huggingface_hub` for publishing and versioning
- Gradio Spaces for interactive demos

---

## License

Apache License 2.0 – see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@misc{jessicai_mudusa_2024,
  author       = {NaTo1000},
  title        = {{JessicaAi Mudusa}: Quantum-Inspired Neural Mesh Transformer},
  year         = {2024},
  url          = {https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak},
}
```
