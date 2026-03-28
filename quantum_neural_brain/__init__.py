"""
quantum_neural_brain – Quantum Neural Brain Hugging Face model package.

Public API::

    from quantum_neural_brain import (
        QuantumNeuralBrainConfig,
        QuantumNeuralBrainModel,
        QuantumNeuralBrainPipeline,
        build_training_pipeline,
        OnlineLearner,
    )
"""

from .config import QuantumNeuralBrainConfig
from .model import QuantumNeuralBrainModel
from .inference import QuantumNeuralBrainPipeline, ContextualMemory
from .training import (
    build_training_pipeline,
    OnlineLearner,
    QNBTrainingArguments,
    QuantumNeuralBrainTrainer,
)

# Register with Hugging Face AutoModel so users can do:
#   AutoConfig.register("quantum_neural_brain", QuantumNeuralBrainConfig)
#   AutoModelForCausalLM.register(QuantumNeuralBrainConfig, QuantumNeuralBrainModel)
try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("quantum_neural_brain", QuantumNeuralBrainConfig)
    AutoModelForCausalLM.register(QuantumNeuralBrainConfig, QuantumNeuralBrainModel)
except Exception:
    pass  # Auto-registration is optional

__all__ = [
    "QuantumNeuralBrainConfig",
    "QuantumNeuralBrainModel",
    "QuantumNeuralBrainPipeline",
    "ContextualMemory",
    "build_training_pipeline",
    "OnlineLearner",
    "QNBTrainingArguments",
    "QuantumNeuralBrainTrainer",
]

__version__ = "1.0.0"
