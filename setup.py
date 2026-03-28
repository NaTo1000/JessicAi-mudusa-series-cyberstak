"""Package setup for quantum_quad_brain."""

from setuptools import setup, find_packages

setup(
    name="quantum_quad_brain",
    version="1.0.0",
    description=(
        "Quantum Quad-Brain compute array integrating NVMe, DRAM, and "
        "Raspberry Pi CM4 compute modules."
    ),
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[],
    extras_require={"dev": ["pytest>=7.0"]},
    package_data={"quantum_quad_brain": ["config/*.yaml"]},
)
