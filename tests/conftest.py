"""
tests/conftest.py – Shared fixtures for the test suite.

Creates a minimal local tokenizer so tests can run fully offline without
downloading the gpt2 or any other HuggingFace model.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Build a minimal offline tokenizer once per session
# ---------------------------------------------------------------------------

_TOKENIZER_DIR = "/tmp/qnb_test_tokenizer"

_TRAINING_CORPUS = [
    "hello world quantum neural brain test data processing information",
    "artificial intelligence language model training evaluation validation",
    "synaptic connections enable adaptive learning through gradient descent",
    "the quick brown fox jumps over the lazy dog pack my box with five dozen",
    "quantum computing superposition entanglement interference circuit qubit",
]


def _build_local_tokenizer(path: str, vocab_size: int = 512) -> None:
    """Build and save a small BPE tokenizer trained on ASCII characters."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=1,
    )
    tokenizer.train_from_iterator(_TRAINING_CORPUS * 10, trainer=trainer)
    os.makedirs(path, exist_ok=True)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast_tok.save_pretrained(path)


@pytest.fixture(scope="session")
def local_tokenizer_path() -> str:
    """Return path to a tiny offline tokenizer, building it if needed."""
    if not os.path.exists(os.path.join(_TOKENIZER_DIR, "tokenizer.json")):
        _build_local_tokenizer(_TOKENIZER_DIR, vocab_size=512)
    return _TOKENIZER_DIR


@pytest.fixture(scope="session")
def local_tokenizer(local_tokenizer_path):
    """Return a loaded :class:`PreTrainedTokenizerFast` for offline testing."""
    tok = PreTrainedTokenizerFast.from_pretrained(local_tokenizer_path)
    return tok
