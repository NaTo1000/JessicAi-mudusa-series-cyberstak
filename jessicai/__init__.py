"""
JessicAI Medusa Series Cyberstack
Hugging Face API integration package
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jessicai")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
