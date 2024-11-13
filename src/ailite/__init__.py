"""
AILite: A lightweight AI library for interacting with HuggingFace and Claude models.
"""

from .main._model._api import HUGPiLLM,HUGPIClient
from .main._ailite import ai, ClaudeEngine
from .main._ailite_api import serve
from .features import *

# Define what should be accessible when using "from ailite import *"
__all__ = [
    'ai',
    'HUGPIClient',
    'HUGPiLLM',
    'ClaudeEngine'
]

# Optional: Add version info
__version__ = "0.8.0"

# Optional: Add author info
__author__ = "Kammari Santhosh"

# Optional: Add any module level docstrings
__doc__ = """
AILite provides simple interfaces to use HuggingFace and Claude models.

Main components:
- HUGPiLLM: HuggingFace model interface
- HUGPIClient: HuggingFace API client
- ClaudeEngine: Interface for Claude models
- ai: Main AI interface
"""