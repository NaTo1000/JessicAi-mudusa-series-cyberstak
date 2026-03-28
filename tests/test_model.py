"""
tests/test_model.py – Unit tests for the Quantum Neural Brain model.

Run with:
    pytest tests/test_model.py -v
"""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from quantum_neural_brain import QuantumNeuralBrainConfig, QuantumNeuralBrainModel
from quantum_neural_brain.inference import ContextualMemory


# ---------------------------------------------------------------------------
# Shared tiny config for all tests (fast CPU execution)
# ---------------------------------------------------------------------------

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
        num_qubits=3,
        quantum_circuit_depth=1,
        num_quantum_blocks=2,
        num_quad_brains=1,
        mesh_fabric_layers=1,
        vertex_dimensions=4,
    )


@pytest.fixture(scope="module")
def tiny_model(tiny_config):
    model = QuantumNeuralBrainModel(tiny_config)
    model.eval()
    return model


def _make_mock_tokenizer(vocab_size: int = 512, eos_id: int = 2):
    """Create a minimal mock tokenizer that does not require network access."""

    class _FakeEncoding(dict):
        """dict subclass that also has a .to() method (like BatchEncoding)."""
        def to(self, device):
            return self

    class _MockTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token = "</s>"
        eos_token_id = eos_id
        _vocab_size = vocab_size

        def __call__(self, text, return_tensors="pt", truncation=False, max_length=512, **kwargs):
            ids = torch.randint(3, self._vocab_size, (1, min(len(text.split()) + 2, 10)))
            mask = torch.ones_like(ids)
            return _FakeEncoding({"input_ids": ids, "attention_mask": mask})

        def decode(self, ids, skip_special_tokens=True):
            return "generated text"

    return _MockTokenizer()




class TestConfig:
    def test_defaults(self):
        cfg = QuantumNeuralBrainConfig()
        assert cfg.hidden_size == 1024
        assert cfg.num_qubits == 8
        assert cfg.num_quad_brains == 2
        assert cfg.mesh_fabric_layers == 3
        assert cfg.vertex_dimensions == 4
        assert cfg.model_type == "quantum_neural_brain"

    def test_custom_values(self, tiny_config):
        assert tiny_config.hidden_size == 64
        assert tiny_config.num_qubits == 3
        assert tiny_config.num_quad_brains == 1

    def test_serialisation(self, tmp_path, tiny_config):
        tiny_config.save_pretrained(str(tmp_path))
        loaded = QuantumNeuralBrainConfig.from_pretrained(str(tmp_path))
        assert loaded.hidden_size == tiny_config.hidden_size
        assert loaded.num_qubits == tiny_config.num_qubits
        assert loaded.model_type == "quantum_neural_brain"


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_instantiation(self, tiny_model):
        assert tiny_model is not None
        n_params = sum(p.numel() for p in tiny_model.parameters())
        assert n_params > 0

    def test_parameter_count(self, tiny_model):
        # Just verify we can count parameters without error
        n_params = sum(p.numel() for p in tiny_model.parameters())
        assert n_params > 1000  # at minimum some meaningful parameters

    def test_forward_pass_shape(self, tiny_config, tiny_model):
        batch, seq = 2, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids)
        assert out.logits.shape == (batch, seq, tiny_config.vocab_size)

    def test_forward_with_attention_mask(self, tiny_config, tiny_model):
        batch, seq = 2, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        attention_mask = torch.ones(batch, seq)
        attention_mask[0, -2:] = 0  # Mask last 2 tokens of first example
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids, attention_mask=attention_mask)
        assert out.logits.shape == (batch, seq, tiny_config.vocab_size)

    def test_loss_computation(self, tiny_config, tiny_model):
        batch, seq = 2, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids, labels=input_ids)
        assert out.loss is not None
        assert out.loss.item() > 0
        assert not torch.isnan(out.loss)

    def test_no_loss_without_labels(self, tiny_config, tiny_model):
        batch, seq = 2, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids)
        assert out.loss is None

    def test_hidden_states_output(self, tiny_config, tiny_model):
        batch, seq = 1, 6
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids, output_hidden_states=True)
        assert out.hidden_states is not None
        assert len(out.hidden_states) > 0

    def test_save_and_reload(self, tiny_config, tiny_model, tmp_path):
        tiny_model.save_pretrained(str(tmp_path))
        tiny_config.save_pretrained(str(tmp_path))

        reloaded = QuantumNeuralBrainModel.from_pretrained(str(tmp_path))
        reloaded.eval()

        batch, seq = 1, 4
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out_orig = tiny_model(input_ids=input_ids)
            out_reload = reloaded(input_ids=input_ids)

        assert torch.allclose(out_orig.logits, out_reload.logits, atol=1e-5)

    def test_gradient_flow(self, tiny_config):
        model = QuantumNeuralBrainModel(tiny_config)
        model.train()
        batch, seq = 1, 4
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        # Check that at least some gradients flowed
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients flowed during backward pass"

    def test_weight_tying(self, tiny_model):
        # LM head weights should be tied to token embeddings
        assert tiny_model.lm_head.weight is tiny_model.token_embedding.weight

    def test_input_ids_or_embeds_required(self, tiny_model):
        with pytest.raises(ValueError):
            tiny_model()

    def test_inputs_embeds(self, tiny_config, tiny_model):
        batch, seq = 1, 4
        embeds = torch.randn(batch, seq, tiny_config.hidden_size)
        with torch.no_grad():
            out = tiny_model(inputs_embeds=embeds)
        assert out.logits.shape == (batch, seq, tiny_config.vocab_size)

    def test_return_dict_false(self, tiny_config, tiny_model):
        batch, seq = 1, 4
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        with torch.no_grad():
            out = tiny_model(input_ids=input_ids, return_dict=False)
        assert isinstance(out, tuple)
        logits = out[0]
        assert logits.shape == (batch, seq, tiny_config.vocab_size)


# ---------------------------------------------------------------------------
# Contextual Memory tests
# ---------------------------------------------------------------------------

class TestContextualMemory:
    def test_add_and_retrieve(self):
        mem = ContextualMemory(max_turns=5)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        assert len(mem) == 2
        ctx = mem.to_context_string()
        assert "Hello" in ctx
        assert "Hi there!" in ctx

    def test_rolling_window(self):
        mem = ContextualMemory(max_turns=3)
        for i in range(5):
            mem.add("user", f"msg {i}")
        assert len(mem) == 3
        ctx = mem.to_context_string()
        # Only last 3 messages should be present
        assert "msg 2" in ctx
        assert "msg 3" in ctx
        assert "msg 4" in ctx
        assert "msg 0" not in ctx

    def test_clear(self):
        mem = ContextualMemory(max_turns=5)
        mem.add("user", "test")
        mem.clear()
        assert len(mem) == 0
        assert mem.to_context_string() == ""

    def test_empty_context_string(self):
        mem = ContextualMemory(max_turns=5)
        assert mem.to_context_string() == ""


# ---------------------------------------------------------------------------
# Pipeline tests (offline – no secondary models, mock tokenizer)
# ---------------------------------------------------------------------------

class TestPipelineOffline:
    @pytest.fixture
    def pipeline(self, tiny_config):
        from quantum_neural_brain.inference import QuantumNeuralBrainPipeline
        with patch("quantum_neural_brain.inference.AutoTokenizer") as mock_tok_cls:
            mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer(
                vocab_size=tiny_config.vocab_size
            )
            p = QuantumNeuralBrainPipeline(
                config=tiny_config,
                qnb_tokenizer_name="mock-tokenizer",
                use_qwen=False,
                use_hauhau=False,
                device="cpu",
            )
        return p

    def test_basic_call(self, pipeline):
        result = pipeline("Hello quantum brain!", max_new_tokens=5)
        assert "qnb_output" in result
        assert "qwen_output" in result
        assert "hauhau_output" in result
        assert "final" in result
        # Secondary outputs should be empty (disabled)
        assert result["qwen_output"] == ""
        assert result["hauhau_output"] == ""
        # Final should equal QNB output when secondaries disabled
        assert result["final"] == result["qnb_output"]

    def test_memory_accumulation(self, pipeline):
        pipeline.reset_memory()
        pipeline("First message", max_new_tokens=5)
        pipeline("Second message", max_new_tokens=5)
        assert len(pipeline.memory) == 4  # 2 user + 2 assistant turns

    def test_memory_in_prompt(self, pipeline):
        pipeline.reset_memory()
        pipeline("Tell me about qubits.", max_new_tokens=5)
        result = pipeline("And what about entanglement?", max_new_tokens=5, use_memory=True)
        # Memory context should be included – just check it runs without error
        assert result["final"] is not None

    def test_reset_memory(self, pipeline):
        pipeline("some message", max_new_tokens=3)
        pipeline.reset_memory()
        assert len(pipeline.memory) == 0

    def test_no_memory_mode(self, pipeline):
        pipeline.reset_memory()
        result = pipeline("Test prompt", max_new_tokens=3, use_memory=False)
        # When use_memory=False nothing is stored
        assert len(pipeline.memory) == 0
        assert result["final"] is not None

    def test_chat_interface(self, pipeline):
        pipeline.reset_memory()
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "What about quantum AI?"},
        ]
        response = pipeline.chat(messages, max_new_tokens=5)
        assert isinstance(response, str)
