"""
QuantumNeuralBrainConfig – Hugging Face PretrainedConfig subclass.

Stores all hyper-parameters that define the Quantum Neural Brain (QNB):
topology, quantum circuit depth, neural-mesh geometry, and the secondary
inference model identifiers (Qwen / Hauhau).
"""

from transformers import PretrainedConfig


class QuantumNeuralBrainConfig(PretrainedConfig):
    """Configuration class for the Quantum Neural Brain model.

    Inherits from :class:`transformers.PretrainedConfig` so the model can be
    loaded/saved with ``from_pretrained`` / ``save_pretrained``.

    Args:
        vocab_size (int): Vocabulary size for the token embeddings.
        hidden_size (int): Dimensionality of the hidden representation.
        num_hidden_layers (int): Number of stacked transformer-style layers.
        num_attention_heads (int): Number of attention heads per layer.
        intermediate_size (int): FFN expansion width inside each layer.
        max_position_embeddings (int): Maximum sequence length.
        dropout_prob (float): Dropout probability applied in each sub-layer.

        # --- Quantum circuit parameters ---
        num_qubits (int): Number of simulated qubits per circuit block.
        quantum_circuit_depth (int): Number of entanglement + rotation layers
            in each quantum circuit block.
        num_quantum_blocks (int): How many quantum circuit blocks are wired
            into the neural mesh.

        # --- Topological neural-mesh parameters ---
        num_quad_brains (int): Recursion depth of the "quad-brain inside a
            quad-brain" topology (each level nests 4 sub-brains).
        mesh_fabric_layers (int): Number of triple-folding fabric mesh layers.
        vertex_dimensions (int): Dimension of the 4-D vertex mechanics space
            used for topological routing.

        # --- Secondary model identifiers ---
        qwen_model_name (str): Hugging Face Hub identifier for the Qwen
            secondary inference model.
        hauhau_model_name (str): Hugging Face Hub identifier for the Hauhau
            (uncensored) secondary inference model.
        secondary_max_new_tokens (int): Max tokens to generate when calling
            either secondary model.
        secondary_temperature (float): Sampling temperature for secondary
            models.

        # --- Memory and learning ---
        context_memory_size (int): Number of past interaction turns retained
            in the contextual memory buffer.
        adaptive_learning_rate (float): Base learning rate used during online
            / continual fine-tuning.
    """

    model_type = "quantum_neural_brain"

    def __init__(
        self,
        # Core transformer geometry
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 2048,
        dropout_prob: float = 0.1,
        # Quantum circuit parameters
        num_qubits: int = 8,
        quantum_circuit_depth: int = 4,
        num_quantum_blocks: int = 4,
        # Topological neural-mesh parameters
        num_quad_brains: int = 2,
        mesh_fabric_layers: int = 3,
        vertex_dimensions: int = 4,
        # Secondary model identifiers
        qwen_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        hauhau_model_name: str = "NousResearch/Hermes-3-Llama-3.1-8B",
        secondary_max_new_tokens: int = 512,
        secondary_temperature: float = 0.7,
        # Memory & learning
        context_memory_size: int = 20,
        adaptive_learning_rate: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        # Ensure word embeddings are tied (needed for proper save/load in transformers ≥5.x)
        kwargs.setdefault("tie_word_embeddings", True)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob

        self.num_qubits = num_qubits
        self.quantum_circuit_depth = quantum_circuit_depth
        self.num_quantum_blocks = num_quantum_blocks

        self.num_quad_brains = num_quad_brains
        self.mesh_fabric_layers = mesh_fabric_layers
        self.vertex_dimensions = vertex_dimensions

        self.qwen_model_name = qwen_model_name
        self.hauhau_model_name = hauhau_model_name
        self.secondary_max_new_tokens = secondary_max_new_tokens
        self.secondary_temperature = secondary_temperature

        self.context_memory_size = context_memory_size
        self.adaptive_learning_rate = adaptive_learning_rate
