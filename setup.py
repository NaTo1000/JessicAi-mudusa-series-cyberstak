"""Package setup for JessicAi-mudusa-series-cyberstak."""

from setuptools import setup, find_packages

setup(
    name="jessicai-cyberstak",
    version="0.1.0",
    description="Cybersecurity and AI development suite with AicodeX, PortmanAI, Kali/BlackArch sandbox, and VPS scaling",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "click>=8.1",
        "rich>=13.0",
        "requests>=2.31",
        "docker>=7.0",
        "huggingface-hub>=0.22",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=4.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "jessicai=main:cli",
        ]
    },
)
