"""setup.py – Package setup for the Quantum Neural Brain."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quantum-neural-brain",
    version="1.0.0",
    description=(
        "Quantum Neural Brain – a Hugging Face-compatible AI model combining "
        "quantum circuit simulation, topological neural mesh, and Qwen/Hauhau "
        "secondary inference for advanced language understanding."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="JessicAi Mudusa Series",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "huggingface-hub>=0.22.0",
        "tokenizers>=0.19.0",
        "datasets>=2.18.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "4bit": ["bitsandbytes>=0.43.0"],
        "dev": ["pytest>=7.4.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
