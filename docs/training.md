# Training Guide – JessicaAi Mudusa

---

## Overview

The Mudusa training pipeline consists of two stages:

1. **Pre-training** – train on a large text corpus to learn language.
2. **Fine-tuning (SFT)** – align the model to follow instructions and chat.

Both stages use [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
and support multi-GPU / multi-node setups via
[Accelerate](https://huggingface.co/docs/accelerate).

---

## Stage 1 – Pre-training

### Configuration

Edit `training/config/pretrain_config.yaml`:

```yaml
model:
  hidden_size: 2048
  num_hidden_layers: 24
  # … see file for full options

training:
  output_dir: ./outputs/pretrain
  per_device_train_batch_size: 4
  learning_rate: 2.0e-4
  # …

data:
  dataset_name: HuggingFaceFW/fineweb
  dataset_config_name: sample-10BT
```

### Single GPU

```bash
python training/pretrain.py --config training/config/pretrain_config.yaml
```

### Multi-GPU (DDP)

```bash
accelerate launch --num_processes 4 \
    training/pretrain.py --config training/config/pretrain_config.yaml
```

### DeepSpeed ZeRO-3

```bash
accelerate launch --config_file accelerate_ds_z3.yaml \
    training/pretrain.py --config training/config/pretrain_config.yaml
```

---

## Stage 2 – Supervised Fine-Tuning (SFT)

### Configuration

Edit `training/config/finetune_config.yaml`:

```yaml
base_model: ./outputs/pretrain   # or NaTo1000/jessicai-mudusa

peft:
  use_peft: true
  lora_r: 64
  lora_alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj]

data:
  dataset_name: HuggingFaceH4/ultrachat_200k
```

### Run fine-tuning

```bash
python training/finetune.py --config training/config/finetune_config.yaml
```

---

## Merging LoRA Weights

After LoRA fine-tuning, merge the adapter back into the base model:

```python
from peft import PeftModel
from jessicai_mudusa import MudusaForCausalLM

base = MudusaForCausalLM.from_pretrained("./outputs/pretrain")
model = PeftModel.from_pretrained(base, "./outputs/finetune")
merged = model.merge_and_unload()
merged.save_pretrained("./outputs/merged")
```

---

## Publishing to Hugging Face Hub

```bash
huggingface-cli login

python - <<'EOF'
from jessicai_mudusa import MudusaForCausalLM, MudusaTokenizer

model = MudusaForCausalLM.from_pretrained("./outputs/merged")
tok   = MudusaTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

model.push_to_hub("NaTo1000/jessicai-mudusa")
tok.save_pretrained("./tokenizer_upload")
# Upload tokenizer files manually or via huggingface_hub
EOF
```

---

## Monitoring with Weights & Biases

Set `report_to: wandb` in your training config and ensure `wandb` is
installed:

```bash
pip install wandb
wandb login
```

---

## Tips for Best Results

- Use **bf16** on Ampere+ GPUs (A100, 4090) for stability.
- Start with **LoRA** (rank 64) before full fine-tuning.
- Use `gradient_checkpointing=True` if VRAM is limited.
- Set `dataloader_pin_memory=True` for faster data loading.
- Validate on a held-out split every 500–1000 steps to detect overfitting.
