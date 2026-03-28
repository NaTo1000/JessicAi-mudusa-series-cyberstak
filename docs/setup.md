# Setup Guide – JessicaAi Mudusa

This guide covers all the steps required to set up the JessicaAi Mudusa model
on your local machine or a cloud instance.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python    | 3.10    | 3.11 / 3.12 |
| RAM       | 16 GB   | 32 GB+      |
| VRAM      | 8 GB    | 24 GB+      |
| Storage   | 20 GB   | 50 GB+      |
| CUDA      | 11.8    | 12.1+       |

---

## 1. Clone the Repository

```bash
git clone https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak.git
cd JessicAi-mudusa-series-cyberstak
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows
```

---

## 3. Install Dependencies

### Full install (training + serving + UI):

```bash
pip install -r requirements.txt
```

### Package-only install (inference only):

```bash
pip install -e ".[serve,gradio]"
```

---

## 4. GPU Support

### NVIDIA CUDA

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (MPS)

```bash
pip install torch torchvision torchaudio
```

### CPU-only

PyTorch is included in `requirements.txt` and defaults to CPU if no GPU is
detected.

---

## 5. Hugging Face Authentication

To push model weights to the Hub or access private models:

```bash
huggingface-cli login
# Enter your HF token when prompted
```

---

## 6. Quick Smoke Test

Run the basic usage example to verify the install:

```bash
python examples/basic_usage.py
```

Expected output ends with:

```
✅  basic_usage.py completed successfully.
```

---

## 7. Hardware Optimisation Tips

### 8-bit / 4-bit quantisation (bitsandbytes)

```python
from jessicai_mudusa import MudusaForCausalLM
import torch

model = MudusaForCausalLM.from_pretrained(
    "NaTo1000/jessicai-mudusa",
    load_in_8bit=True,     # or load_in_4bit=True
    device_map="auto",
)
```

### Flash Attention 2 (NVIDIA Ampere+)

```bash
pip install flash-attn --no-build-isolation
```

Then pass `attn_implementation="flash_attention_2"` to `from_pretrained`.

### DeepSpeed ZeRO (multi-GPU)

```bash
pip install deepspeed
accelerate config  # choose DeepSpeed backend
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `per_device_train_batch_size` or use 4-bit quant |
| `ModuleNotFoundError: jessicai_mudusa` | Run `pip install -e .` from repo root |
| `trust_remote_code=True` required | Pass `trust_remote_code=True` to `from_pretrained` |
| Slow inference on CPU | Use `torch.compile(model)` or switch to GPU |
