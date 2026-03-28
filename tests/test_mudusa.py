"""
Tests for the JessicaAi Mudusa model package.

These tests run without network access and without large GPU memory by using
a tiny configuration (hidden_size=128, 2 layers, etc.).

Run::

    pytest tests/
"""

from __future__ import annotations

import pytest
import torch

from jessicai_mudusa import MudusaConfig, MudusaForCausalLM, MudusaModel
from jessicai_mudusa.modeling_mudusa import (
    MudusaRMSNorm,
    QuantumPhaseAttention,
    MudusaMLP,
    MudusaDecoderLayer,
    TripleNeuralMesh,
    SynapticPlasticityMemory,
)


# ---------------------------------------------------------------------------
# Shared tiny config for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_config() -> MudusaConfig:
    return MudusaConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        num_mesh_channels=3,
        quantum_phase_dim=16,
        synaptic_memory_size=16,
        synaptic_memory_dim=16,
        mesh_fusion_type="gated",
    )


@pytest.fixture(scope="module")
def tiny_model(tiny_config: MudusaConfig) -> MudusaForCausalLM:
    model = MudusaForCausalLM(tiny_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestMudusaConfig:
    def test_model_type(self, tiny_config):
        assert tiny_config.model_type == "mudusa"

    def test_defaults(self):
        cfg = MudusaConfig()
        assert cfg.vocab_size == 151936
        assert cfg.hidden_size == 2048
        assert cfg.num_mesh_channels == 3
        assert cfg.mesh_fusion_type == "gated"

    def test_custom_params(self, tiny_config):
        assert tiny_config.hidden_size == 64
        assert tiny_config.num_hidden_layers == 2
        assert tiny_config.synaptic_memory_size == 16

    def test_serialisation_round_trip(self, tmp_path, tiny_config):
        tiny_config.save_pretrained(tmp_path)
        loaded = MudusaConfig.from_pretrained(tmp_path)
        assert loaded.hidden_size == tiny_config.hidden_size
        assert loaded.num_mesh_channels == tiny_config.num_mesh_channels
        assert loaded.synaptic_memory_size == tiny_config.synaptic_memory_size


# ---------------------------------------------------------------------------
# Component tests
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self, tiny_config):
        norm = MudusaRMSNorm(tiny_config.hidden_size)
        x = torch.randn(2, 8, tiny_config.hidden_size)
        out = norm(x)
        assert out.shape == x.shape

    def test_scale_param(self, tiny_config):
        norm = MudusaRMSNorm(tiny_config.hidden_size)
        assert norm.weight.shape == (tiny_config.hidden_size,)


class TestQuantumPhaseAttention:
    def test_output_shape(self, tiny_config):
        attn = QuantumPhaseAttention(tiny_config)
        x = torch.randn(1, 6, tiny_config.hidden_size)
        out, present = attn(x, use_cache=True)
        assert out.shape == x.shape
        assert present is not None
        assert len(present) == 2  # (keys, values)

    def test_phase_param_shape(self, tiny_config):
        attn = QuantumPhaseAttention(tiny_config)
        assert attn.quantum_phase.shape == (tiny_config.num_attention_heads,)

    def test_causal_mask_respected(self, tiny_config):
        attn = QuantumPhaseAttention(tiny_config)
        x = torch.randn(1, 4, tiny_config.hidden_size)
        mask = torch.full((1, 1, 4, 4), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        out_masked, _ = attn(x, attention_mask=mask)
        assert out_masked.shape == x.shape


class TestMudusaMLP:
    def test_output_shape(self, tiny_config):
        mlp = MudusaMLP(tiny_config)
        x = torch.randn(2, 5, tiny_config.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape


class TestDecoderLayer:
    def test_forward(self, tiny_config):
        layer = MudusaDecoderLayer(tiny_config)
        x = torch.randn(1, 4, tiny_config.hidden_size)
        out, present = layer(x, use_cache=True)
        assert out.shape == x.shape

    def test_no_cache(self, tiny_config):
        layer = MudusaDecoderLayer(tiny_config)
        x = torch.randn(1, 4, tiny_config.hidden_size)
        out, present = layer(x, use_cache=False)
        assert present is None


class TestTripleNeuralMesh:
    @pytest.mark.parametrize("fusion", ["gated", "mean", "concat"])
    def test_fusion_modes(self, fusion):
        cfg = MudusaConfig(
            vocab_size=512,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            num_mesh_channels=3,
            synaptic_memory_size=8,
            synaptic_memory_dim=8,
            mesh_fusion_type=fusion,
        )
        mesh = TripleNeuralMesh(cfg, layer_idx=0)
        x = torch.randn(1, 4, cfg.hidden_size)
        out, pasts = mesh(x, use_cache=True)
        assert out.shape == x.shape
        assert len(pasts) == cfg.num_mesh_channels


class TestSynapticPlasticityMemory:
    def test_output_shape(self, tiny_config):
        mem = SynapticPlasticityMemory(tiny_config)
        x = torch.randn(2, 8, tiny_config.hidden_size)
        out = mem(x)
        assert out.shape == x.shape

    def test_differentiable(self, tiny_config):
        mem = SynapticPlasticityMemory(tiny_config)
        x = torch.randn(1, 4, tiny_config.hidden_size, requires_grad=True)
        out = mem(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestMudusaModel:
    def test_forward_shape(self, tiny_config, tiny_model):
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        with torch.no_grad():
            out = tiny_model(input_ids, return_dict=True)
        assert out.logits.shape == (2, 8, tiny_config.vocab_size)

    def test_loss_computation(self, tiny_config, tiny_model):
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 6))
        labels = input_ids.clone()
        with torch.no_grad():
            out = tiny_model(input_ids, labels=labels, return_dict=True)
        assert out.loss is not None
        assert out.loss.item() > 0

    def test_kv_cache(self, tiny_config, tiny_model):
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            out = tiny_model(input_ids, use_cache=True, return_dict=True)
        assert out.past_key_values is not None
        assert len(out.past_key_values) == tiny_config.num_hidden_layers

    def test_generate(self, tiny_config, tiny_model):
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 3))
        with torch.no_grad():
            generated = tiny_model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tiny_config.eos_token_id,
            )
        assert generated.shape[1] > input_ids.shape[1]
        assert generated.shape[0] == 1

    def test_attention_mask(self, tiny_config, tiny_model):
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
        mask = torch.ones(2, 6)
        mask[1, -2:] = 0  # padding on second sample
        with torch.no_grad():
            out = tiny_model(input_ids, attention_mask=mask, return_dict=True)
        assert out.logits.shape == (2, 6, tiny_config.vocab_size)

    def test_parameter_count_nonzero(self, tiny_model):
        total = sum(p.numel() for p in tiny_model.parameters())
        assert total > 0

    def test_save_load_round_trip(self, tmp_path, tiny_config, tiny_model):
        tiny_model.save_pretrained(tmp_path)
        loaded = MudusaForCausalLM.from_pretrained(tmp_path)
        # Verify configs match
        assert loaded.config.hidden_size == tiny_config.hidden_size
        assert loaded.config.num_mesh_channels == tiny_config.num_mesh_channels

    def test_tuple_output(self, tiny_config, tiny_model):
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            out = tiny_model(input_ids, return_dict=False)
        assert isinstance(out, tuple)
        assert isinstance(out[0], torch.Tensor)


# ---------------------------------------------------------------------------
# Tokenizer tests (no network)
# ---------------------------------------------------------------------------

class TestMudusaTokenizer:
    def test_apply_chat_template(self):
        from jessicai_mudusa import MudusaTokenizer

        class _MockTok:
            vocab_size = 100
            eos_token_id = 2
            bos_token_id = 1
            pad_token_id = 0

            def __call__(self, *a, **kw):
                return {}

            def encode(self, t, **kw):
                return [0]

            def decode(self, ids, **kw):
                return ""

            def batch_decode(self, seqs, **kw):
                return [""] * len(seqs)

            def save_pretrained(self, *a, **kw):
                pass

        tok = MudusaTokenizer(_MockTok())
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "Hello" in prompt
        assert "Hi!" in prompt

    def test_vocab_size_property(self):
        from jessicai_mudusa import MudusaTokenizer

        class _MockTok:
            vocab_size = 42
            eos_token_id = None
            bos_token_id = None
            pad_token_id = None

            def __call__(self, *a, **kw):
                return {}

            def encode(self, t, **kw):
                return []

            def decode(self, ids, **kw):
                return ""

            def batch_decode(self, seqs, **kw):
                return []

            def save_pretrained(self, *a, **kw):
                pass

        tok = MudusaTokenizer(_MockTok())
        assert tok.vocab_size == 42
