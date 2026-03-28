"""
tests/test_training.py – Unit tests for the training utilities.

Run with:
    pytest tests/test_training.py -v
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from quantum_neural_brain import QuantumNeuralBrainConfig, QuantumNeuralBrainModel
from quantum_neural_brain.training import (
    OnlineLearner,
    QNBTrainingArguments,
    TextDataset,
    build_training_pipeline,
)


@pytest.fixture(scope="module")
def tiny_config():
    return QuantumNeuralBrainConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
        dropout_prob=0.0,
        num_qubits=2,
        quantum_circuit_depth=1,
        num_quantum_blocks=2,
        num_quad_brains=1,
        mesh_fabric_layers=1,
    )


def _local_tok(path):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(path)


class TestTextDataset:
    def test_length(self, local_tokenizer):
        texts = ["Hello world", "Quantum AI", "Neural mesh"]
        ds = TextDataset(texts, local_tokenizer, max_length=16)
        assert len(ds) == 3

    def test_item_keys(self, local_tokenizer):
        texts = ["Hello world"]
        ds = TextDataset(texts, local_tokenizer, max_length=16)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item

    def test_max_length(self, local_tokenizer):
        texts = ["Hello world"]
        ds = TextDataset(texts, local_tokenizer, max_length=8)
        item = ds[0]
        assert item["input_ids"].shape[0] == 8


class TestOnlineLearner:
    def test_single_step(self, tiny_config, local_tokenizer_path):
        model = QuantumNeuralBrainModel(tiny_config)
        with patch("quantum_neural_brain.training.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = _local_tok(local_tokenizer_path)
            learner = OnlineLearner(model, tokenizer_name=local_tokenizer_path, learning_rate=1e-3)
        loss = learner.step("What is AI?", "AI is artificial intelligence.")
        assert isinstance(loss, float)
        assert loss > 0
        assert not (loss != loss)  # not NaN

    def test_multiple_steps(self, tiny_config, local_tokenizer_path):
        model = QuantumNeuralBrainModel(tiny_config)
        with patch("quantum_neural_brain.training.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = _local_tok(local_tokenizer_path)
            learner = OnlineLearner(model, tokenizer_name=local_tokenizer_path, learning_rate=1e-3)
        text = "Quantum computing uses superposition."
        losses = [learner.step(text, text) for _ in range(3)]
        # All losses should be positive finite floats
        assert all(l > 0 for l in losses)

    def test_step_count(self, tiny_config, local_tokenizer_path):
        model = QuantumNeuralBrainModel(tiny_config)
        with patch("quantum_neural_brain.training.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = _local_tok(local_tokenizer_path)
            learner = OnlineLearner(model, tokenizer_name=local_tokenizer_path)
        assert learner._step_count == 0
        learner.step("test", "response")
        assert learner._step_count == 1


class TestBuildTrainingPipeline:
    def test_creates_trainer(self, tiny_config, local_tokenizer_path):
        train_texts = ["Hello world", "Quantum AI"]
        with patch("quantum_neural_brain.training.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = _local_tok(local_tokenizer_path)
            trainer = build_training_pipeline(
                train_texts=train_texts,
                config=tiny_config,
                tokenizer_name=local_tokenizer_path,
                training_args=QNBTrainingArguments(
                    output_dir="/tmp/test_qnb",
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    save_strategy="no",
                    use_cpu=True,
                    report_to="none",
                    evaluation_strategy="no",
                ),
            )
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.train_dataset is not None

    def test_with_eval_dataset(self, tiny_config, local_tokenizer_path):
        with patch("quantum_neural_brain.training.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = _local_tok(local_tokenizer_path)
            trainer = build_training_pipeline(
                train_texts=["Train text 1", "Train text 2"],
                eval_texts=["Eval text 1"],
                config=tiny_config,
                tokenizer_name=local_tokenizer_path,
                training_args=QNBTrainingArguments(
                    output_dir="/tmp/test_qnb_eval",
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    save_strategy="no",
                    use_cpu=True,
                    report_to="none",
                    evaluation_strategy="epoch",
                ),
            )
        assert trainer.eval_dataset is not None

    def test_one_epoch_runs(self, tiny_config, local_tokenizer_path):
        train_texts = [
            "Quantum neural brain processes information.",
            "Synaptic connections enable learning.",
        ]
        args = QNBTrainingArguments(
            output_dir="/tmp/test_qnb_train",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            save_strategy="no",
            use_cpu=True,
            report_to="none",
            evaluation_strategy="no",
            logging_steps=1000,
        )
        with patch("quantum_neural_brain.training.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = _local_tok(local_tokenizer_path)
            trainer = build_training_pipeline(
                train_texts=train_texts,
                config=tiny_config,
                tokenizer_name=local_tokenizer_path,
                training_args=args,
            )
        result = trainer.train()
        assert result.training_loss > 0
