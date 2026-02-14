"""Mock implementations for testing Sage Sanctum agents."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..auth.credentials import GatewayCredentials
from ..auth.trat import (
    AllowedModels,
    RequesterContext,
    TransactionContext,
    TransactionToken,
)
from ..gateway.client import GatewayClient


class MockGatewayClient(GatewayClient):
    """Mock gateway client for testing.

    Returns configurable credentials and endpoints without any real
    network or authentication dependencies.

    Args:
        is_gateway: Whether to simulate gateway mode. Defaults to ``False``.
        endpoints: Provider → URL mapping. Defaults to localhost endpoints.
        trat: Optional ``TransactionToken`` to return from ``get_trat()``.

    Example:
        ```python
        client = MockGatewayClient(is_gateway=True)
        creds = client.get_credentials()
        assert creds.spiffe_jwt == "mock-spiffe-jwt"
        ```
    """

    def __init__(
        self,
        is_gateway: bool = False,
        endpoints: dict[str, str] | None = None,
        trat: TransactionToken | None = None,
    ) -> None:
        self._is_gateway = is_gateway
        self._endpoints = endpoints or {
            "openai": "http://localhost:8080/v1",
            "anthropic": "http://localhost:8080/anthropic",
            "google": "http://localhost:8080/google",
        }
        self._trat = trat

    def get_credentials(self) -> GatewayCredentials:
        """Return mock credentials (``"mock-spiffe-jwt"`` / ``"mock-trat-jwt"``)."""
        return GatewayCredentials(
            spiffe_jwt="mock-spiffe-jwt",
            trat="mock-trat-jwt",
        )

    def get_endpoint(self, provider: str) -> str:
        """Return the mock endpoint URL for a provider."""
        return self._endpoints.get(provider, "http://localhost:8080/v1")

    @property
    def is_gateway_mode(self) -> bool:
        """Whether gateway mode is simulated."""
        return self._is_gateway

    def get_trat(self) -> TransactionToken | None:
        """Return the configured mock TraT, or ``None``."""
        return self._trat


class MockLLM(BaseChatModel):
    """Mock LangChain chat model for testing.

    Returns canned responses in order and records every call for assertion.

    Attributes:
        responses: Ordered list of response strings to return.
            When exhausted, returns ``"Mock response"``.
        calls: List of message lists received — one entry per ``invoke()`` call.

    Example:
        ```python
        llm = MockLLM(responses=["Finding: SQL injection", "No issues"])
        r1 = llm.invoke([...])  # "Finding: SQL injection"
        r2 = llm.invoke([...])  # "No issues"
        assert len(llm.calls) == 2
        ```
    """

    responses: list[str] = []
    calls: list[list[BaseMessage]] = []
    _call_index: int = 0

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, responses: list[str] | None = None, **kwargs: Any):
        """Initialize the mock LLM.

        Args:
            responses: Canned responses to return in order. If ``None``,
                every call returns ``"Mock response"``.
        """
        super().__init__(**kwargs)
        if responses:
            self.responses = responses
        self.calls = []
        self._call_index = 0

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Record the call and return the next canned response."""
        self.calls.append(messages)

        if self._call_index < len(self.responses):
            content = self.responses[self._call_index]
            self._call_index += 1
        else:
            content = "Mock response"

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=content),
                )
            ]
        )


class MockTratClient:
    """Mock Transaction Token client for testing.

    Returns a configurable ``TransactionToken`` without any file or
    sidecar dependencies. Ships with sensible defaults for common
    testing scenarios.

    Args:
        trat: An explicit ``TransactionToken`` to return.
        allowed_models: Custom model allowlists. Ignored if ``trat`` is
            provided. Defaults to gpt-4o-mini (triage), gpt-4o +
            claude-3-5-sonnet (analysis), o1 (reasoning),
            text-embedding-3-small (embeddings).
    """

    def __init__(
        self,
        trat: TransactionToken | None = None,
        allowed_models: AllowedModels | None = None,
    ) -> None:
        if trat:
            self._trat = trat
        else:
            self._trat = self._default_trat(allowed_models)

    def get_token(self) -> TransactionToken:
        """Return the mock Transaction Token."""
        return self._trat

    def invalidate(self) -> None:
        """No-op — mock tokens don't need cache invalidation."""
        pass

    @staticmethod
    def _default_trat(
        allowed_models: AllowedModels | None = None,
    ) -> TransactionToken:
        """Create a default test TraT with preset models and metadata."""
        import time

        if not allowed_models:
            allowed_models = AllowedModels(
                triage=["openai:gpt-4o-mini"],
                analysis=["openai:gpt-4o", "anthropic:claude-3-5-sonnet-latest"],
                reasoning=["openai:o1"],
                embeddings=["openai:text-embedding-3-small"],
            )

        return TransactionToken(
            raw="mock-trat-jwt",
            txn="run_test123",
            sub="github|org-12345",
            scope="scan.execute scan.upload",
            req_wl="spiffe://sage-sanctum.local/scanner/run_test123",
            iat=time.time(),
            exp=time.time() + 300,  # 5 minutes from now
            aud="sage-sanctum.local",
            iss="https://tts.sage-sanctum.local",
            tctx=TransactionContext(
                run_id="run_test123",
                org_id="12345",
                repo_url="https://github.com/acme/repo",
                agent_type="sage-scanner",
                agent_mode="standard",
                allowed_models=allowed_models,
                allowed_providers=["openai", "anthropic"],
            ),
            rctx=RequesterContext(
                trigger="pull_request",
                pr_number=42,
                actor="dependabot[bot]",
            ),
        )


class MockEmbeddings(Embeddings):
    """Mock LangChain embeddings for testing.

    Returns deterministic fixed-dimension vectors and records every call.

    Attributes:
        dimension: Dimension of returned vectors. Defaults to ``8``.
        calls: List of input lists received — one entry per call.

    Example:
        ```python
        embeddings = MockEmbeddings(dimension=4)
        vecs = embeddings.embed_documents(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 4
        assert len(embeddings.calls) == 1
        ```
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic vectors (hash-based) for each text."""
        self.calls.append(texts)
        return [self._deterministic_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a deterministic vector for a single query."""
        self.calls.append([text])
        return self._deterministic_vector(text)

    def _deterministic_vector(self, text: str) -> list[float]:
        """Generate a reproducible unit-length vector from text."""
        h = hash(text)
        raw = [float((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self.dimension)]
        # Normalize to unit length
        magnitude = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / magnitude for x in raw]
