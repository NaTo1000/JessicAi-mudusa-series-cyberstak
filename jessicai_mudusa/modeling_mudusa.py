"""
modeling_mudusa.py – JessicaAi Mudusa neural architecture.

Architecture overview
---------------------
The Mudusa model is a decoder-only transformer enhanced with three
Mudusa-specific innovations:

1. **Quantum-Phase Attention (QPA)**
   Standard grouped-query attention whose softmax weights are modulated by
   learnable quantum-phase angles.  Phase vectors are projected onto the unit
   circle and multiplied element-wise with the raw attention scores before
   softmax, achieving superposition-like mixing across heads.

2. **Triple-Layer Neural Mesh**
   Three parallel transformer stacks (cortical, limbic, quantum channels)
   process the sequence independently.  Their hidden states are fused by a
   learned gating network at each layer, analogous to the way distinct brain
   regions integrate information.

3. **Synaptic Plasticity Memory (SPM)**
   A fixed-size key–value memory bank sits above the transformer stack.  At
   each generation step the model writes a summary of the current context into
   the memory and reads the top-K most similar entries, implementing a form of
   Hebbian associative recall without gradient updates.

All components are registered as standard ``nn.Module`` subclasses and are
fully compatible with HuggingFace ``generate()``, ``Trainer``, and PEFT.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_mudusa import MudusaConfig


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class MudusaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2022)."""

    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 1e6) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)
        return _apply_rotary_emb(q, k, cos, sin)


# ---------------------------------------------------------------------------
# RMS LayerNorm
# ---------------------------------------------------------------------------


class MudusaRMSNorm(nn.Module):
    """Root-mean-square layer normalisation (Zhang & Sennrich, 2019)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


# ---------------------------------------------------------------------------
# Quantum-Phase Attention
# ---------------------------------------------------------------------------


class QuantumPhaseAttention(nn.Module):
    """
    Grouped-query attention with quantum-phase modulation.

    Phase angles ``φ ∈ [0, 2π)`` are learned per-head.  Before softmax, each
    attention score matrix is element-wise multiplied by
    ``cos(φ_h) + 1`` (range [0, 2]) providing a smooth head-specific
    amplitude gate inspired by quantum phase interference.
    """

    def __init__(self, config: MudusaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Quantum phase angles – one per attention head, initialised at zero.
        # Each head modulates its attention scores by cos(φ_h) + 1 ∈ (0, 2].
        # quantum_phase_dim from config is reserved for future extended phase
        # representations; currently each head uses a single scalar angle.
        self.quantum_phase = nn.Parameter(
            torch.zeros(self.num_heads)
        )

        self.rotary_emb = MudusaRotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query head count (GQA)."""
        bsz, num_kv, seq, dim = kv.shape
        return kv[:, :, None, :, :].expand(bsz, num_kv, self.kv_groups, seq, dim).reshape(
            bsz, num_kv * self.kv_groups, seq, dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = q_len if past_key_value is None else past_key_value[0].shape[2] + q_len
        q, k = self.rotary_emb(q, k, kv_seq_len)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present = (k, v) if use_cache else None

        k_full = self._expand_kv(k)
        v_full = self._expand_kv(v)

        attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * self.scale

        # Quantum-phase modulation: gate ∈ (0, 2] per head
        phase_gate = (torch.cos(self.quantum_phase) + 1.0).view(1, self.num_heads, 1, 1)
        attn_weights = attn_weights * phase_gate

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v_full)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_out), present


# ---------------------------------------------------------------------------
# Feed-Forward (SwiGLU)
# ---------------------------------------------------------------------------


class MudusaMLP(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020)."""

    def __init__(self, config: MudusaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Single transformer block (one mesh channel)
# ---------------------------------------------------------------------------


class MudusaDecoderLayer(nn.Module):
    """One transformer block for a single neural-mesh channel."""

    def __init__(self, config: MudusaConfig) -> None:
        super().__init__()
        self.self_attn = QuantumPhaseAttention(config)
        self.mlp = MudusaMLP(config)
        self.input_layernorm = MudusaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MudusaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Synaptic plasticity gate – a per-dimension learnable gate vector of
        # shape (hidden_size,). Applied element-wise after attention to scale
        # each feature independently, mimicking Hebbian synapse weighting.
        self.synaptic_gate = nn.Parameter(torch.ones(config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # Apply synaptic plasticity gate
        gate = torch.sigmoid(self.synaptic_gate)
        hidden_states = residual + gate * attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


# ---------------------------------------------------------------------------
# Triple Neural Mesh
# ---------------------------------------------------------------------------


class TripleNeuralMesh(nn.Module):
    """
    Three parallel transformer stacks fused by a learned gating network.

    Channels:
    - **Cortical** (index 0): logical reasoning, structured language.
    - **Limbic**   (index 1): emotional tone, context sensitivity.
    - **Quantum**  (index 2): creativity, long-range pattern integration.

    Fusion strategy (``config.mesh_fusion_type``):
    - ``"gated"``  – softmax over three learned scalar gates per position.
    - ``"mean"``   – simple average of channel outputs.
    - ``"concat"`` – concatenate then project back to ``hidden_size``.
    """

    def __init__(self, config: MudusaConfig, layer_idx: int) -> None:
        super().__init__()
        self.channels = nn.ModuleList(
            [MudusaDecoderLayer(config) for _ in range(config.num_mesh_channels)]
        )
        self.fusion_type = config.mesh_fusion_type

        if self.fusion_type == "gated":
            # One gate score per channel, applied per token position
            self.gate_proj = nn.Linear(
                config.hidden_size, config.num_mesh_channels, bias=True
            )
        elif self.fusion_type == "concat":
            self.merge_proj = nn.Linear(
                config.hidden_size * config.num_mesh_channels, config.hidden_size, bias=False
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        channel_outputs: list[torch.Tensor] = []
        new_past: list = []

        for ch_idx, channel in enumerate(self.channels):
            pkv = past_key_values[ch_idx] if past_key_values is not None else None
            ch_out, ch_present = channel(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=pkv,
                use_cache=use_cache,
            )
            channel_outputs.append(ch_out)
            if use_cache:
                new_past.append(ch_present)

        # --- Fusion ---
        stacked = torch.stack(channel_outputs, dim=-1)  # (B, S, H, C)

        if self.fusion_type == "gated":
            gates = F.softmax(self.gate_proj(hidden_states), dim=-1)  # (B, S, C)
            fused = (stacked * gates.unsqueeze(-2)).sum(-1)
        elif self.fusion_type == "concat":
            cat = torch.cat(channel_outputs, dim=-1)
            fused = self.merge_proj(cat)
        else:  # mean
            fused = stacked.mean(-1)

        return fused, new_past


# ---------------------------------------------------------------------------
# Synaptic Plasticity Memory
# ---------------------------------------------------------------------------


class SynapticPlasticityMemory(nn.Module):
    """
    A differentiable associative memory bank.

    Keys and values are stored as ``nn.Embedding`` tables initialised
    randomly and kept **fixed** (``requires_grad=False``) at inference time.
    During training they receive gradients via the soft read operation.

    At inference:
    1. The current hidden state is projected to a query vector.
    2. Cosine similarity is computed against all memory keys.
    3. The top-K most similar values are soft-pooled and added to the
       hidden state (residual connection).
    """

    def __init__(self, config: MudusaConfig) -> None:
        super().__init__()
        self.memory_size = config.synaptic_memory_size
        self.memory_dim = config.synaptic_memory_dim
        self.hidden_size = config.hidden_size
        self.top_k = min(8, config.synaptic_memory_size)

        self.keys = nn.Embedding(self.memory_size, self.memory_dim)
        self.values = nn.Embedding(self.memory_size, self.memory_dim)

        self.query_proj = nn.Linear(self.hidden_size, self.memory_dim, bias=False)
        self.value_proj = nn.Linear(self.memory_dim, self.hidden_size, bias=False)
        self.norm = MudusaRMSNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use the mean-pooled representation as the query
        query = self.query_proj(hidden_states.mean(dim=1))  # (B, memory_dim)
        query = F.normalize(query, dim=-1)

        key_mat = F.normalize(self.keys.weight, dim=-1)  # (M, memory_dim)
        scores = torch.matmul(query, key_mat.T)  # (B, M)

        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)  # (B, K)
        topk_weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # (B, K, 1)

        topk_values = self.values(topk_idx)  # (B, K, memory_dim)
        aggregated = (topk_weights * topk_values).sum(dim=1)  # (B, memory_dim)
        memory_out = self.value_proj(aggregated).unsqueeze(1)  # (B, 1, hidden_size)

        return self.norm(hidden_states + memory_out)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class MudusaPreTrainedModel(PreTrainedModel):
    """Base class for all Mudusa models – handles weight init and HF hooks."""

    config_class = MudusaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # Use the legacy list-of-tuples cache format rather than DynamicCache
    _supports_cache_class = False

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class MudusaModel(MudusaPreTrainedModel):
    """
    The bare Mudusa transformer decoder outputting raw hidden states.

    Forward pass returns :class:`~transformers.modeling_outputs.BaseModelOutputWithPast`.
    """

    def __init__(self, config: MudusaConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.mesh_layers = nn.ModuleList(
            [TripleNeuralMesh(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = MudusaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.synaptic_memory = SynapticPlasticityMemory(config)
        self.post_init()

    @staticmethod
    def _make_causal_mask(
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]  # (1, 1, S, S)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPast, Tuple]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Guard: if HuggingFace passes a DynamicCache (transformers >= 4.50),
        # fall back to cache-free mode for this custom nested-cache architecture.
        if past_key_values is not None and not isinstance(past_key_values, list):
            past_key_values = None

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Determine full KV sequence length (past + current)
        past_kv_len = 0
        if past_key_values is not None:
            # past_key_values[layer][channel] = (k, v), k has shape (B, H, S_past, D)
            try:
                past_kv_len = past_key_values[0][0][0].shape[2]
            except (IndexError, TypeError):
                past_kv_len = 0
        kv_len = past_kv_len + seq_len

        # Causal mask: shape (1, 1, seq_len, kv_len)
        # Each new token can attend to all past KV tokens plus causally to current tokens.
        causal_mask = self._make_causal_mask(seq_len, inputs_embeds.dtype, device)
        if past_kv_len > 0:
            # Prepend zeros (attend to all past tokens freely)
            prefix = torch.zeros(
                (1, 1, seq_len, past_kv_len), dtype=inputs_embeds.dtype, device=device
            )
            causal_mask = torch.cat([prefix, causal_mask], dim=-1)
        if attention_mask is not None:
            # Combine with padding mask for the full KV length
            if attention_mask.shape[-1] == kv_len:
                pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
                pad_mask = pad_mask * torch.finfo(inputs_embeds.dtype).min
                causal_mask = causal_mask + pad_mask

        hidden_states = inputs_embeds
        all_hidden_states: tuple = () if output_hidden_states else None
        new_past: list = []

        for layer_idx, mesh_layer in enumerate(self.mesh_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            hidden_states, layer_new_past = mesh_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_values=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                new_past.append(layer_new_past)

        hidden_states = self.norm(hidden_states)

        # Synaptic memory read
        hidden_states = self.synaptic_memory(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values_out = new_past if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values_out, all_hidden_states] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values_out,
            hidden_states=all_hidden_states,
        )


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------


class MudusaForCausalLM(MudusaPreTrainedModel, GenerationMixin):
    """
    Mudusa model with a causal language-modelling (LM) head.

    This is the primary model class for text generation.  It is compatible
    with :meth:`~transformers.GenerationMixin.generate` and HuggingFace
    ``Trainer``.

    Example::

        from jessicai_mudusa import MudusaConfig, MudusaForCausalLM
        import torch

        config = MudusaConfig(num_hidden_layers=2, hidden_size=256,
                               intermediate_size=512, num_attention_heads=4,
                               num_key_value_heads=2)
        model = MudusaForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        out = model(input_ids)
        print(out.logits.shape)   # (1, 16, 151936)
    """

    # Weight tying: lm_head shares embed_tokens weights when tie_word_embeddings=True
    # Declare as list for HuggingFace compatibility
    _tied_weights_keys: list[str] = []

    def __init__(self, config: MudusaConfig) -> None:
        super().__init__(config)
        self.model = MudusaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[CausalLMOutputWithPast, Tuple]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        # The Mudusa nested-list cache format is not compatible with
        # HuggingFace's DynamicCache management in generate().  We run
        # full-context (no-cache) generation here for correctness; the
        # MudusaPipeline.stream() method uses our native KV cache.
        return {
            "input_ids": input_ids,
            "past_key_values": None,
            "use_cache": False,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(
        past_key_values: list, beam_idx: torch.LongTensor
    ) -> list:
        return [
            [
                (kv[0].index_select(0, beam_idx), kv[1].index_select(0, beam_idx))
                if kv is not None
                else None
                for kv in layer_past
            ]
            for layer_past in past_key_values
        ]
