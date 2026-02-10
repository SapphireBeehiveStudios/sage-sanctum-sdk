"""ModelRef: Canonical model identifier in provider:model format."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ModelRefParseError

# Provider detection patterns
_PROVIDER_PREFIXES = {
    "claude": "anthropic",
    "anthropic/": "anthropic",
    "gemini": "google",
    "google/": "google",
}


def _detect_provider(model: str) -> str:
    """Auto-detect provider from model name."""
    model_lower = model.lower()
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model_lower.startswith(prefix):
            return provider
    return "openai"


@dataclass(frozen=True)
class ModelRef:
    """Canonical reference to an LLM model.

    Format: provider:model (e.g., 'openai:gpt-4o', 'anthropic:claude-3-5-sonnet-latest')
    """

    provider: str
    model: str

    @classmethod
    def parse(cls, ref: str) -> ModelRef:
        """Parse a model reference string.

        Accepts:
        - 'provider:model' (explicit)
        - 'model' (auto-detect provider)

        Raises:
            ModelRefParseError: If the reference is empty or invalid.
        """
        if not ref or not ref.strip():
            raise ModelRefParseError("Model reference cannot be empty")

        ref = ref.strip()

        if ":" in ref:
            parts = ref.split(":", 1)
            provider = parts[0].strip().lower()
            model = parts[1].strip()
            if not provider:
                raise ModelRefParseError(f"Empty provider in model reference: {ref!r}")
            if not model:
                raise ModelRefParseError(f"Empty model in model reference: {ref!r}")
            return cls(provider=provider, model=model)

        # Auto-detect provider from model name
        provider = _detect_provider(ref)
        return cls(provider=provider, model=ref)

    @property
    def for_litellm(self) -> str:
        """Format for LiteLLM routing.

        LiteLLM requires provider prefixes for non-OpenAI models:
        - anthropic/claude-3-5-sonnet -> anthropic routing
        - gemini/gemini-2.0-flash -> Google AI Studio routing
        - gpt-4o -> OpenAI (no prefix needed)
        """
        if self.provider == "anthropic":
            if self.model.startswith("anthropic/"):
                return self.model
            return f"anthropic/{self.model}"
        if self.provider == "google":
            if self.model.startswith("gemini/"):
                return self.model
            return f"gemini/{self.model}"
        return self.model

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:
        return f"ModelRef(provider={self.provider!r}, model={self.model!r})"
