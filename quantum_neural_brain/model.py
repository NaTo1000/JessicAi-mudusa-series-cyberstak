"""
model.py – Hugging Face-compatible Quantum Neural Brain model.

The :class:`QuantumNeuralBrainModel` extends
:class:`transformers.PreTrainedModel`, making it fully compatible with the
Hugging Face ecosystem:

  * ``from_pretrained`` / ``save_pretrained``
  * ``push_to_hub``
  * Trainer / Seq2SeqTrainer integration
  * Automatic device placement and mixed-precision support

Architecture
------------
                 ┌─────────────────────────────────────┐
  tokens ──────► │  Token + Positional Embeddings       │
                 │  + Quantum Noise Seeding             │
                 └────────────┬────────────────────────┘
                              │
                 ┌────────────▼────────────────────────┐
                 │  Transformer Encoder Layers (N)       │
                 │  (self-attention + FFN)               │
                 └────────────┬────────────────────────┘
                              │
                 ┌────────────▼────────────────────────┐
                 │  QuantumInterferenceLayer             │
                 │  (simulated quantum circuit blocks)   │
                 └────────────┬────────────────────────┘
                              │
                 ┌────────────▼────────────────────────┐
                 │  NeuralMeshNetwork                    │
                 │  (quad-brain, vertex, fabric mesh)    │
                 └────────────┬────────────────────────┘
                              │
                 ┌────────────▼────────────────────────┐
                 │  LM Head → logits                    │
                 └─────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
)

from .config import QuantumNeuralBrainConfig
from .quantum_circuits import QuantumInterferenceLayer
from .neural_mesh import NeuralMeshNetwork


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class QuantumAttention(nn.Module):
    """Multi-head self-attention with quantum phase modulation.

    Adds a learned phase shift (inspired by quantum phase gates) to the
    attention logits, allowing the model to encode "quantum coherence" in
    the attention patterns.

    Args:
        hidden_size (int): Model hidden dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Attention dropout probability.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Quantum phase modulation per head
        self.phase_shift = nn.Parameter(torch.zeros(num_heads))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention logits with quantum phase shift
        attn = (q @ k.transpose(-2, -1)) * self.scale
        phase = self.phase_shift.unsqueeze(-1).unsqueeze(-1)  # (heads, 1, 1)
        attn = attn + phase

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class QuantumFFN(nn.Module):
    """Position-wise feed-forward network with gated activation.

    Uses a gated linear unit (GLU) variant for richer non-linear capacity,
    inspired by recent large-model designs.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class QuantumTransformerLayer(nn.Module):
    """One encoder layer: quantum attention + quantum FFN + residuals."""

    def __init__(self, config: QuantumNeuralBrainConfig):
        super().__init__()
        self.attn = QuantumAttention(
            config.hidden_size, config.num_attention_heads, config.dropout_prob
        )
        self.ffn = QuantumFFN(
            config.hidden_size, config.intermediate_size, config.dropout_prob
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class QuantumNeuralBrainModel(PreTrainedModel):
    """Quantum Neural Brain – Hugging Face-compatible language model.

    Combines a transformer encoder stack with quantum circuit interference
    processing and a topological neural mesh.  Outputs causal language model
    logits suitable for text generation.

    Usage::

        from quantum_neural_brain import QuantumNeuralBrainConfig, QuantumNeuralBrainModel

        config = QuantumNeuralBrainConfig()
        model = QuantumNeuralBrainModel(config)

        # Tokenise some text with any compatible tokeniser
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

    Attributes:
        config_class: Points to :class:`QuantumNeuralBrainConfig`.
        base_model_prefix: Used by the Trainer for weight saving.
    """

    config_class = QuantumNeuralBrainConfig
    base_model_prefix = "quantum_neural_brain"
    supports_gradient_checkpointing = True
    # In transformers ≥5.x _tied_weights_keys is a dict: {tied_key: source_key}
    _tied_weights_keys = {"lm_head.weight": "token_embedding.weight"}

    def __init__(self, config: QuantumNeuralBrainConfig):
        super().__init__(config)

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout_prob)

        # Quantum noise seed (trainable) – provides the "interference from
        # pure noise" seeding described in the design brief
        self.quantum_noise_seed = nn.Parameter(
            torch.randn(1, 1, config.hidden_size) * 0.02
        )

        # Transformer encoder stack
        self.transformer_layers = nn.ModuleList(
            [QuantumTransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Quantum interference processing
        self.quantum_layer = QuantumInterferenceLayer(
            num_blocks=config.num_quantum_blocks,
            num_qubits=config.num_qubits,
            depth=config.quantum_circuit_depth,
            hidden_size=config.hidden_size,
        )

        # Topological neural mesh
        self.neural_mesh = NeuralMeshNetwork(
            hidden_size=config.hidden_size,
            num_quad_brains=config.num_quad_brains,
            mesh_fabric_layers=config.mesh_fabric_layers,
            vertex_dimensions=config.vertex_dimensions,
            dropout=config.dropout_prob,
        )

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_norm = nn.LayerNorm(config.hidden_size)

        # Initialise weights; tie_weights() is invoked inside init_weights()
        # which is called by post_init().  Weight tying is declared via
        # _tied_weights_keys and get_output_embeddings / get_input_embeddings,
        # so we do NOT manually tie here – doing so would interfere with the
        # transformers ≥5.x weight-initialization bookkeeping.
        self.post_init()

    # ------------------------------------------------------------------
    # Hugging Face helpers
    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embedding = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        """Full forward pass through the Quantum Neural Brain.

        Args:
            input_ids: Token IDs ``(batch, seq_len)``.
            attention_mask: ``1`` for real tokens, ``0`` for padding.
            position_ids: Explicit position indices.  Auto-generated if None.
            labels: Target token IDs for cross-entropy loss computation.
            inputs_embeds: Pre-computed embeddings (mutually exclusive with
                *input_ids*).
            output_hidden_states: If ``True``, return all layer hidden states.
            return_dict: If ``True``, return a :class:`CausalLMOutput`.

        Returns:
            :class:`CausalLMOutput` containing ``loss`` (if *labels* given),
            ``logits``, and optionally ``hidden_states``.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.token_embedding(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

        pos_embeds = self.position_embedding(position_ids)

        # Add quantum noise seed – models "thoughts from interference noise"
        hidden = self.embed_dropout(inputs_embeds + pos_embeds + self.quantum_noise_seed)

        # Build causal attention mask
        causal_mask = self._build_causal_mask(seq_len, hidden.device, hidden.dtype)
        if attention_mask is not None:
            # Expand padding mask (B, T) → (B, 1, 1, T) and combine
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * torch.finfo(hidden.dtype).min
            causal_mask = causal_mask + pad_mask

        all_hidden_states: list = []

        for layer in self.transformer_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden)
            hidden = layer(hidden, causal_mask)

        # Quantum interference processing
        hidden = self.quantum_layer(hidden)

        # Topological neural mesh
        hidden = self.neural_mesh(hidden)

        hidden = self.final_norm(hidden)

        if output_hidden_states:
            all_hidden_states.append(hidden)

        logits = self.lm_head(hidden)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,) + ((tuple(all_hidden_states),) if output_hidden_states else ())
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build an upper-triangular causal mask ``(1, 1, seq_len, seq_len)``."""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)
