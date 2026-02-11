"""Model categories for different agent tasks."""

from __future__ import annotations

from enum import Enum


class ModelCategory(str, Enum):
    """Categories of LLM usage within an agent.

    Each category maps to a list of allowed models in the TraT's
    ``tctx.allowed_models``. Use these when calling
    ``AgentContext.create_llm_client()``.

    Attributes:
        TRIAGE: Quick classification, routing, and filtering tasks.
            Typically uses a fast, inexpensive model (e.g. ``gpt-4o-mini``).
        ANALYSIS: Detailed code analysis and vulnerability detection.
            Typically uses a capable model (e.g. ``gpt-4o``).
        REASONING: Complex multi-step reasoning tasks.
            Typically uses a reasoning model (e.g. ``o1``).
        EMBEDDINGS: Embedding generation for similarity search.
            Typically uses an embedding model (e.g. ``text-embedding-3-small``).
    """

    TRIAGE = "triage"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    EMBEDDINGS = "embeddings"
