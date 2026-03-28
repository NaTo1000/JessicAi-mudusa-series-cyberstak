"""
training.py – Configurable training and fine-tuning utilities for the
Quantum Neural Brain model.

Features
--------
* ``QuantumNeuralBrainTrainer`` – a thin wrapper around the HuggingFace
  ``Trainer`` that adds QNB-specific defaults (cosine LR schedule, gradient
  clipping, mixed-precision training).
* ``build_training_pipeline`` – convenience factory that sets up a full
  training run from a dataset, tokeniser, and config.
* ``OnlineLearner`` – lightweight continual/online fine-tuning utility for
  adaptive learning from new conversational data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import QuantumNeuralBrainConfig
from .model import QuantumNeuralBrainModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Minimal text dataset that tokenises strings at construction time.

    Args:
        texts: List of raw text strings.
        tokenizer: HuggingFace tokeniser.
        max_length: Maximum tokenised sequence length.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.examples = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.examples["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.examples.items()}


# ---------------------------------------------------------------------------
# QNB-specific TrainingArguments defaults
# ---------------------------------------------------------------------------

@dataclass
class QNBTrainingArguments(TrainingArguments):
    """Training arguments with sensible defaults for Quantum Neural Brain.

    Inherits all fields from :class:`transformers.TrainingArguments`.  The
    overridden defaults are optimised for the QNB architecture size.
    """

    output_dir: str = field(default="./qnb_checkpoints")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=8)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=100)
    lr_scheduler_type: str = field(default="cosine")
    gradient_accumulation_steps: int = field(default=4)
    fp16: bool = field(default=torch.cuda.is_available())
    save_strategy: str = field(default="epoch")
    evaluation_strategy: str = field(default="epoch")
    logging_steps: int = field(default=50)
    load_best_model_at_end: bool = field(default=True)
    save_total_limit: int = field(default=3)
    dataloader_num_workers: int = field(default=2)
    report_to: str = field(default="none")


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------

class QuantumNeuralBrainTrainer(Trainer):
    """Trainer subclass that adds QNB-specific logging and loss handling.

    Overrides ``compute_loss`` to support labelled batches where *labels*
    is set to the shifted *input_ids* (standard causal LM objective).
    """

    def compute_loss(
        self,
        model: QuantumNeuralBrainModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        # Use the `labels` key that the DataCollator already placed in `inputs`
        # (it shifts input_ids → labels internally).  Only fall back to using
        # `input_ids` as labels if `labels` is not present.
        inputs = dict(inputs)
        if "labels" not in inputs:
            inputs["labels"] = inputs.get("input_ids")
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_training_pipeline(
    train_texts: List[str],
    eval_texts: Optional[List[str]] = None,
    config: Optional[QuantumNeuralBrainConfig] = None,
    tokenizer_name: str = "gpt2",
    training_args: Optional[QNBTrainingArguments] = None,
    model: Optional[QuantumNeuralBrainModel] = None,
) -> QuantumNeuralBrainTrainer:
    """Build a ready-to-run training pipeline.

    Args:
        train_texts: List of training text strings.
        eval_texts: Optional list of evaluation text strings.
        config: QNB config used when creating a new model.
        tokenizer_name: HuggingFace tokeniser identifier.
        training_args: Custom training arguments.
        model: Pre-constructed model (skips model creation if provided).

    Returns:
        A configured :class:`QuantumNeuralBrainTrainer` instance.  Call
        ``.train()`` to start training.

    Example::

        trainer = build_training_pipeline(
            train_texts=["Hello, world!", "Quantum AI is amazing."],
            eval_texts=["Test sentence."],
        )
        trainer.train()
        trainer.save_model("./my_qnb_model")
    """
    cfg = config or QuantumNeuralBrainConfig()
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if model is None:
        model = QuantumNeuralBrainModel(cfg)
        logger.info(
            "Created fresh QuantumNeuralBrainModel with %d parameters.",
            sum(p.numel() for p in model.parameters()),
        )

    args = training_args or QNBTrainingArguments()

    train_ds = TextDataset(train_texts, tok, max_length=cfg.max_position_embeddings)
    eval_ds = (
        TextDataset(eval_texts, tok, max_length=cfg.max_position_embeddings)
        if eval_texts
        else None
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    return QuantumNeuralBrainTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tok,
    )


# ---------------------------------------------------------------------------
# Online / Continual learner
# ---------------------------------------------------------------------------

class OnlineLearner:
    """Lightweight continual learning utility.

    Allows the QNB model to adapt incrementally from new (prompt, response)
    pairs without running a full training loop.  Uses a small gradient step
    per example (online SGD with a cosine-annealed micro-LR).

    Args:
        model: :class:`QuantumNeuralBrainModel` to fine-tune.
        tokenizer_name: Tokeniser identifier.
        learning_rate: Per-step learning rate.
        device: Target device string.

    Example::

        learner = OnlineLearner(model, learning_rate=1e-5)
        learner.step("What is quantum computing?", "It uses qubits to ...")
    """

    def __init__(
        self,
        model: QuantumNeuralBrainModel,
        tokenizer_name: str = "gpt2",
        learning_rate: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(torch.device(device))
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self._step_count = 0

    def step(self, prompt: str, response: str) -> float:
        """Perform a single online gradient step on one (prompt, response) pair.

        Args:
            prompt: Input prompt string.
            response: Target response string.

        Returns:
            The scalar loss value for this step.
        """
        text = f"{prompt} {response}"
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.model.eval()

        self._step_count += 1
        return loss.item()
