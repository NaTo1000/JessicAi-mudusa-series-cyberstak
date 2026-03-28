"""
Setup configuration for the JessicaAi Mudusa AI model package.
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="jessicai-mudusa",
    version="1.0.0",
    description=(
        "JessicaAi Mudusa – a quantum-inspired, synapse-driven transformer "
        "AI model built for Hugging Face deployment."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NaTo1000",
    url="https://github.com/NaTo1000/JessicAi-mudusa-series-cyberstak",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "datasets>=2.18.0",
        "tokenizers>=0.19.0",
        "safetensors>=0.4.3",
        "huggingface_hub>=0.22.0",
        "peft>=0.10.0",
        "numpy>=1.26.0",
        "networkx>=3.2.1",
        "sentencepiece>=0.2.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "serve": ["fastapi>=0.110.0", "uvicorn[standard]>=0.29.0", "pydantic>=2.7.0"],
        "gradio": ["gradio>=4.28.0"],
        "train": ["trl>=0.8.6", "bitsandbytes>=0.43.0", "peft>=0.10.0"],
        "dev": ["pytest>=8.1.0", "pytest-asyncio>=0.23.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "hugging-face",
        "transformers",
        "large-language-model",
        "quantum-neural",
        "jessicai",
        "mudusa",
        "conversational-ai",
    ],
)
