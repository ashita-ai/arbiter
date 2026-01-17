"""Core type definitions and enumerations for Arbiter.

This module defines the fundamental types used throughout the Arbiter
evaluation framework, including provider enumerations.
"""

from enum import Enum

__all__ = ["Provider"]


class Provider(str, Enum):
    """Enumeration of supported LLM providers.

    Each provider represents a different LLM API service. The enum
    values match PydanticAI's provider naming convention for the
    <provider>:<model> format (e.g., "openai:gpt-4o").

    Attributes:
        OPENAI: OpenAI's GPT models (GPT-3.5, GPT-4, etc.)
        ANTHROPIC: Anthropic's Claude models (Claude 3 family)
        GOOGLE: Google's Gemini models (Gemini Pro, etc.)
        GROQ: Groq's fast inference service for open models
        MISTRAL: Mistral AI models
        COHERE: Cohere models
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"  # For Gemini models
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
