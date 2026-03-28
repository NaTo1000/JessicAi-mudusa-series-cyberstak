"""
finetune.py – Supervised fine-tuning (SFT) script for JessicaAi Mudusa.

Supports full fine-tuning and LoRA / QLoRA via PEFT.

Usage::

    # Full fine-tuning
    python training/finetune.py --config training/config/finetune_config.yaml

    # LoRA fine-tuning (set peft.use_peft=true in the config)
    python training/finetune.py --config training/config/finetune_config.yaml

    # Multi-GPU
    accelerate launch training/finetune.py \\
        --config training/config/finetune_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from jessicai_mudusa import MudusaForCausalLM, MudusaTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the JessicaAi Mudusa model.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/finetune_config.yaml",
        help="Path to the YAML training configuration file.",
    )
    return parser.parse_args()


def format_chat_sample(sample: dict, tokenizer) -> str:
    """Convert a chat sample to the Mudusa prompt format."""
    from jessicai_mudusa import MudusaTokenizer as MT
    messages = sample.get("messages", [])
    if not messages:
        return ""
    # Build a MudusaTokenizer-compatible wrapper just for template formatting
    wrapper = MT(tokenizer)
    return wrapper.apply_chat_template(messages, add_generation_prompt=False)


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"].get("seed", 42))

    # ------------------------------------------------------------------ #
    # 1. Load model                                                       #
    # ------------------------------------------------------------------ #
    base_model_path = cfg["base_model"]
    logger.info("Loading base model from %s …", base_model_path)

    torch_dtype = torch.bfloat16 if cfg["training"].get("bf16") else torch.float32
    model = MudusaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
    )

    # ------------------------------------------------------------------ #
    # 2. Optional: wrap with LoRA                                         #
    # ------------------------------------------------------------------ #
    peft_cfg = cfg.get("peft", {})
    if peft_cfg.get("use_peft", False):
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=peft_cfg.get("lora_r", 64),
                lora_alpha=peft_cfg.get("lora_alpha", 128),
                lora_dropout=peft_cfg.get("lora_dropout", 0.05),
                target_modules=peft_cfg.get("target_modules", ["q_proj", "v_proj"]),
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            logger.warning("peft not installed; running full fine-tuning.")

    # ------------------------------------------------------------------ #
    # 3. Tokenizer                                                        #
    # ------------------------------------------------------------------ #
    tokenizer = MudusaTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

    # ------------------------------------------------------------------ #
    # 4. Dataset                                                          #
    # ------------------------------------------------------------------ #
    data_cfg = cfg["data"]
    logger.info("Loading dataset: %s …", data_cfg["dataset_name"])
    raw_dataset = load_dataset(data_cfg["dataset_name"], streaming=False)

    def formatting_func(sample):
        return format_chat_sample(sample, tokenizer._tokenizer)

    # ------------------------------------------------------------------ #
    # 5. Training                                                         #
    # ------------------------------------------------------------------ #
    train_cfg = {k: v for k, v in cfg["training"].items() if k != "seed"}
    training_args = TrainingArguments(**train_cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset.get("test"),
        formatting_func=formatting_func,
        max_seq_length=data_cfg.get("max_seq_length", 4096),
    )

    logger.info("Starting fine-tuning …")
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    logger.info("Fine-tuning complete.  Model saved to %s", train_cfg["output_dir"])


if __name__ == "__main__":
    main()
