"""Model categories for different agent tasks."""

from __future__ import annotations

from enum import Enum


class ModelCategory(str, Enum):
    """Categories of LLM usage within an agent.

    Each category maps to a list of allowed models in the TraT's tctx.allowed_models.
    """

    TRIAGE = "triage"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    EMBEDDINGS = "embeddings"
