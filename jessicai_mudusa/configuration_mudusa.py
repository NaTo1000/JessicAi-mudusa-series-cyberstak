"""
MudusaConfig – configuration class for the JessicaAi Mudusa model.

The Mudusa model is a quantum-inspired, synapse-driven transformer whose
architecture features:

* **Quantum-Phase Multi-Head Attention** – attention weights are modulated by
  learnable quantum-phase angles, enabling superposition-like information
  mixing across heads.
* **Triple-Layer Neural Mesh** – three parallel transformer sub-stacks
  (cortical, limbic, and quantum channels) whose outputs are fused via an
  adaptive gating mechanism.
* **Synaptic Plasticity Module** – per-layer gating vectors that mimic
  Hebbian-style weight updates during inference, giving the model dynamic
  context retention.
* **Knowledge Graph Memory** – a fixed-size key–value memory bank that
  stores and retrieves factual associations across turns.
"""

from transformers import PretrainedConfig


class MudusaConfig(PretrainedConfig):
    r"""
    Configuration for :class:`~jessicai_mudusa.MudusaForCausalLM`.

    Args:
        vocab_size (:obj:`int`, *optional*, defaults to 151936):
            Vocabulary size (matches Qwen2 tokenizer).
        hidden_size (:obj:`int`, *optional*, defaults to 2048):
            Dimensionality of the model's hidden representations.
        intermediate_size (:obj:`int`, *optional*, defaults to 8192):
            Dimensionality of the feed-forward intermediate layer.
        num_hidden_layers (:obj:`int`, *optional*, defaults to 24):
            Number of transformer blocks in **each** neural mesh channel.
        num_attention_heads (:obj:`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_key_value_heads (:obj:`int`, *optional*, defaults to 8):
            Number of key/value heads for grouped-query attention (GQA).
        max_position_embeddings (:obj:`int`, *optional*, defaults to 32768):
            Maximum supported sequence length.
        rope_theta (:obj:`float`, *optional*, defaults to 1000000.0):
            Base period for Rotary Position Embeddings (RoPE).
        rms_norm_eps (:obj:`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS LayerNorm.
        num_mesh_channels (:obj:`int`, *optional*, defaults to 3):
            Number of parallel neural mesh channels
            (cortical / limbic / quantum).
        quantum_phase_dim (:obj:`int`, *optional*, defaults to 64):
            Dimension of the learnable quantum-phase angle vectors.
        synaptic_memory_size (:obj:`int`, *optional*, defaults to 512):
            Number of slots in the knowledge-graph memory bank.
        synaptic_memory_dim (:obj:`int`, *optional*, defaults to 256):
            Embedding dimension for each memory slot.
        mesh_fusion_type (:obj:`str`, *optional*, defaults to ``"gated"``):
            Strategy to fuse mesh channels:
            ``"gated"`` (learnable), ``"mean"``, or ``"concat"``.
        tie_word_embeddings (:obj:`bool`, *optional*, defaults to ``False``):
            Whether to tie input / output embedding weights.
        initializer_range (:obj:`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialisation.
        use_cache (:obj:`bool`, *optional*, defaults to ``True``):
            Whether to return KV-cache for fast auto-regressive decoding.
        pad_token_id (:obj:`int`, *optional*, defaults to ``None``):
            Index of the padding token.
        bos_token_id (:obj:`int`, *optional*, defaults to 151643):
            Index of the beginning-of-sequence token.
        eos_token_id (:obj:`int`, *optional*, defaults to 151645):
            Index of the end-of-sequence token.
    """

    model_type = "mudusa"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        # Mudusa-specific
        num_mesh_channels: int = 3,
        quantum_phase_dim: int = 64,
        synaptic_memory_size: int = 512,
        synaptic_memory_dim: int = 256,
        mesh_fusion_type: str = "gated",
        # Standard HF fields
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        # Mudusa-specific
        self.num_mesh_channels = num_mesh_channels
        self.quantum_phase_dim = quantum_phase_dim
        self.synaptic_memory_size = synaptic_memory_size
        self.synaptic_memory_dim = synaptic_memory_dim
        self.mesh_fusion_type = mesh_fusion_type

        self.initializer_range = initializer_range
        self.use_cache = use_cache

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
