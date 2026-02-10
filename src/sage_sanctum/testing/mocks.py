"""Mock implementations for testing Sage Sanctum agents."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

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
from ..llm.model_ref import ModelRef


class MockGatewayClient(GatewayClient):
    """Mock gateway client for testing.

    Returns configurable credentials and endpoints.
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
        return GatewayCredentials(
            spiffe_jwt="mock-spiffe-jwt",
            trat="mock-trat-jwt",
        )

    def get_endpoint(self, provider: str) -> str:
        return self._endpoints.get(provider, "http://localhost:8080/v1")

    @property
    def is_gateway_mode(self) -> bool:
        return self._is_gateway

    def get_trat(self) -> TransactionToken | None:
        return self._trat


class MockLLM(BaseChatModel):
    """Mock LLM for testing.

    Returns canned responses and tracks calls.
    """

    responses: list[str] = []
    calls: list[list[BaseMessage]] = []
    _call_index: int = 0

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, responses: list[str] | None = None, **kwargs: Any):
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
    """Mock Transaction Token client for testing."""

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
        return self._trat

    def invalidate(self) -> None:
        pass

    @staticmethod
    def _default_trat(
        allowed_models: AllowedModels | None = None,
    ) -> TransactionToken:
        """Create a default test TraT."""
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
