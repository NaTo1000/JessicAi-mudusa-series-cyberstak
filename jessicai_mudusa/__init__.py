"""
JessicaAi Mudusa – Quantum-Inspired Neural Mesh Transformer
============================================================

Public API for the ``jessicai_mudusa`` package.

Example usage::

    from jessicai_mudusa import MudusaConfig, MudusaForCausalLM, MudusaTokenizer

    config = MudusaConfig()
    model  = MudusaForCausalLM(config)

Or via the high-level pipeline::

    from jessicai_mudusa import MudusaPipeline
    pipe = MudusaPipeline.from_pretrained("NaTo1000/jessicai-mudusa")
    output = pipe("Tell me about quantum consciousness.")
    print(output)
"""

from .configuration_mudusa import MudusaConfig
from .modeling_mudusa import (
    MudusaModel,
    MudusaForCausalLM,
    MudusaPreTrainedModel,
)
from .tokenization_mudusa import MudusaTokenizer
from .pipeline_mudusa import MudusaPipeline

__version__ = "1.0.0"
__author__ = "NaTo1000"

__all__ = [
    "MudusaConfig",
    "MudusaModel",
    "MudusaForCausalLM",
    "MudusaPreTrainedModel",
    "MudusaTokenizer",
    "MudusaPipeline",
]
