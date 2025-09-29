"""
CuliFeed AI Processing Module
=============================

AI integration module providing multi-provider support for article relevance analysis
and summarization using Groq, Gemini, and OpenAI APIs with intelligent fallback.
"""

from .providers.base import AIProvider, AIResult, AIError
from .providers.groq_provider import GroqProvider
from .ai_manager import AIManager

__all__ = ["AIProvider", "AIResult", "AIError", "GroqProvider", "AIManager"]
