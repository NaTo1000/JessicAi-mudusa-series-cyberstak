# JessicAi-mudusa-series-cyberstak

# Quantum Neural Brain 🧠⚛️

A Hugging Face-compatible AI model that fuses **quantum circuit simulation**, a
**topological neural mesh** (quad-brain, 4-D vertex mechanics, triple-fold
fabric layers), and **Qwen / Hauhau secondary inference** for advanced language
understanding and creative generation.

---

## Architecture

```
User prompt
    │
    ▼
Token + Positional Embeddings
+ Quantum Noise Seed (trainable)   ← "thoughts from pure interference noise"
    │
    ▼
Transformer Encoder Stack (N layers)
    │
    ▼
Quantum Interference Layer
│   (superposition, entanglement, interference readout)
    │
    ▼
NeuralMeshNetwork
│   ├─ QuadBrainBlock (recursive)   quad-brain inside quad-brain
│   ├─ VertexMechanicsLayer (4-D)   4D vertex routing
│   └─ FabricMeshLayer × 3         triple-folding fabric mesh
    │
    ▼
LM Head → logits
    │
    ▼  (optional secondary inference)
Qwen2.5 refinement pass
    │
    ▼
Hauhau / Hermes creative pass
    │
    ▼
Final response  +  Contextual memory update
```

---

## Installation

```bash
git clone https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak.git
cd JessicAi-mudusa-series-cyberstak

pip install -r requirements.txt
pip install -e .                      # editable install
pip install bitsandbytes>=0.43.0      # optional: 4-bit quantisation
```

---

## Quick Start

### 1. Basic inference (offline, no downloads)

```python
import torch
from transformers import AutoTokenizer
from quantum_neural_brain import QuantumNeuralBrainConfig, QuantumNeuralBrainModel

config = QuantumNeuralBrainConfig(
    vocab_size=50257,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
    num_qubits=4,
    quantum_circuit_depth=2,
    num_quantum_blocks=2,
    num_quad_brains=1,
    mesh_fabric_layers=1,
)

model = QuantumNeuralBrainModel(config).eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("The quantum neural brain awakens:", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.logits.shape)  # (1, seq_len, 50257)
```

```bash
python examples/basic_inference.py
```

---

### 2. Full pipeline with Qwen + Hauhau

```python
from quantum_neural_brain import QuantumNeuralBrainConfig, QuantumNeuralBrainPipeline

config = QuantumNeuralBrainConfig(
    qwen_model_name="Qwen/Qwen2.5-7B-Instruct",
    hauhau_model_name="NousResearch/Hermes-3-Llama-3.1-8B",
    secondary_max_new_tokens=256,
    secondary_temperature=0.7,
    context_memory_size=20,
)

pipeline = QuantumNeuralBrainPipeline(
    config=config,
    use_qwen=True,
    use_hauhau=True,
    device="cuda",
    load_secondary_in_4bit=True,
)

result = pipeline("Explain quantum superposition and its role in AI.")
print(result["qnb_output"])     # QNB primary
print(result["qwen_output"])    # Qwen linguistic refinement
print(result["hauhau_output"])  # Hauhau creative expansion
print(result["final"])          # Best output
```

```bash
python examples/qwen_integration.py            # full (GPU + ~15 GB)
python examples/qwen_integration.py --offline  # demo mode, CPU only
```

---

### 3. Multi-turn conversation

```python
response = pipeline.chat([
    {"role": "user",      "content": "What are qubits?"},
    {"role": "assistant", "content": "Qubits are quantum bits..."},
    {"role": "user",      "content": "How do they enable superposition?"},
])
print(response)
```

---

### 4. Fine-tuning

```python
from quantum_neural_brain import build_training_pipeline, QNBTrainingArguments

trainer = build_training_pipeline(
    train_texts=["Custom training text 1", "Training text 2"],
    eval_texts=["Evaluation text 1"],
    config=config,
    tokenizer_name="gpt2",
    training_args=QNBTrainingArguments(
        output_dir="./my_qnb_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
    ),
)
trainer.train()
trainer.save_model("./my_qnb_model")
```

```bash
python examples/fine_tuning.py
```

---

### 5. Online / continual learning

```python
from quantum_neural_brain import QuantumNeuralBrainModel, OnlineLearner

model = QuantumNeuralBrainModel.from_pretrained("./my_qnb_model")
learner = OnlineLearner(model, learning_rate=1e-5)
loss = learner.step("What is the neural mesh?",
                    "A topological structure with quad-brain recursion.")
print(f"Loss: {loss:.4f}")
```

---

### 6. Push to Hugging Face Hub

```bash
huggingface-cli login
python examples/push_to_hub.py --repo YOUR_USERNAME/quantum-neural-brain
```

Or from Python:

```python
model.push_to_hub("YOUR_USERNAME/quantum-neural-brain")
config.push_to_hub("YOUR_USERNAME/quantum-neural-brain")
```

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | `1024` | Core feature dimension |
| `num_hidden_layers` | `12` | Transformer encoder layers |
| `num_attention_heads` | `16` | Attention heads per layer |
| `num_qubits` | `8` | Simulated qubits per circuit block |
| `quantum_circuit_depth` | `4` | Rotation+entanglement layers per block |
| `num_quantum_blocks` | `4` | Parallel quantum circuit blocks |
| `num_quad_brains` | `2` | Quad-brain recursion depth |
| `mesh_fabric_layers` | `3` | Triple-folding fabric mesh layers |
| `vertex_dimensions` | `4` | 4-D vertex mechanics axes |
| `qwen_model_name` | `"Qwen/Qwen2.5-7B-Instruct"` | Qwen secondary model |
| `hauhau_model_name` | `"NousResearch/Hermes-3-Llama-3.1-8B"` | Hauhau secondary model |
| `secondary_max_new_tokens` | `512` | Max tokens from secondary models |
| `secondary_temperature` | `0.7` | Sampling temperature |
| `context_memory_size` | `20` | Conversation turns retained in memory |

---

## Hardware Requirements

| Mode | Minimum |
|------|---------|
| QNB only (CPU) | 4 GB RAM |
| QNB only (GPU) | 2 GB VRAM |
| + Qwen 7B (fp16) | 16 GB VRAM |
| + Hauhau 8B (fp16) | 20 GB VRAM |
| All models (4-bit) | 10 GB VRAM |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
quantum_neural_brain/
├── __init__.py          # Public API + HF Auto-registration
├── config.py            # QuantumNeuralBrainConfig
├── model.py             # QuantumNeuralBrainModel (PreTrainedModel)
├── quantum_circuits.py  # Quantum circuit simulation
├── neural_mesh.py       # Topological neural mesh
├── inference.py         # Pipeline + ContextualMemory
└── training.py          # Trainer, OnlineLearner, build_training_pipeline

examples/
├── basic_inference.py   # Offline quick-start demo
├── qwen_integration.py  # Full pipeline with Qwen + Hauhau
├── fine_tuning.py       # Batch and online learning demos
└── push_to_hub.py       # Publish to Hugging Face Hub

tests/
├── test_model.py        # Model unit tests
├── test_circuits.py     # Quantum circuit + neural mesh tests
└── test_training.py     # Training utility tests
```

---

## License

Apache 2.0
