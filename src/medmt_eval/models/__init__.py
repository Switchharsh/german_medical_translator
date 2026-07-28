"""Lazy-loading translation model adapters."""

from .base import Translator
from .factory import create_translator

__all__ = ["Translator", "create_translator"]
