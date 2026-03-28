"""
pretrain.py – Pre-training script for JessicaAi Mudusa.

Usage::

    python training/pretrain.py --config training/config/pretrain_config.yaml

    # Multi-GPU with accelerate:
    accelerate launch training/pretrain.py \\
        --config training/config/pretrain_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from jessicai_mudusa import MudusaConfig, MudusaForCausalLM, MudusaTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train the JessicaAi Mudusa model.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/pretrain_config.yaml",
        help="Path to the YAML training configuration file.",
    )
    return parser.parse_args()


def tokenize_function(examples, tokenizer, text_column: str, max_seq_length: int):
    return tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True,
    )


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"].get("seed", 42))

    # ------------------------------------------------------------------ #
    # 1. Build model from scratch                                         #
    # ------------------------------------------------------------------ #
    logger.info("Initialising model from config …")
    model_cfg = MudusaConfig(**cfg["model"])
    model = MudusaForCausalLM(model_cfg)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{param_count:,}")

    # ------------------------------------------------------------------ #
    # 2. Tokenizer                                                        #
    # ------------------------------------------------------------------ #
    logger.info("Loading tokenizer …")
    tokenizer = MudusaTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

    # ------------------------------------------------------------------ #
    # 3. Dataset                                                          #
    # ------------------------------------------------------------------ #
    data_cfg = cfg["data"]
    logger.info("Loading dataset: %s …", data_cfg["dataset_name"])
    raw_dataset = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config_name"),
        streaming=False,
    )

    tokenized = raw_dataset.map(
        lambda ex: tokenize_function(
            ex,
            tokenizer._tokenizer,
            data_cfg["text_column"],
            data_cfg["max_seq_length"],
        ),
        batched=True,
        num_proc=data_cfg.get("preprocessing_num_workers", 4),
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenising",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer._tokenizer,
        mlm=False,
    )

    # ------------------------------------------------------------------ #
    # 4. Training                                                         #
    # ------------------------------------------------------------------ #
    train_cfg = cfg["training"]
    training_args = TrainingArguments(**{k: v for k, v in train_cfg.items() if k != "seed"})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
    )

    logger.info("Starting pre-training …")
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    logger.info("Pre-training complete.  Model saved to %s", train_cfg["output_dir"])


if __name__ == "__main__":
    main()
