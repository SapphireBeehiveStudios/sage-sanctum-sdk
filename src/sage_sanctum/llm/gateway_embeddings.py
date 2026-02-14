"""GatewayEmbeddings: LangChain Embeddings wrapping the LLM gateway over UDS.

Mirrors GatewayChatModel — uses GatewayHttpClient to make /v1/embeddings
calls over Unix sockets with SPIFFE + TraT auth headers per request.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings

from ..errors import GatewayError
from ..gateway.http import GatewayHttpClient
from .model_ref import ModelRef

if TYPE_CHECKING:
    from ..gateway.client import GatewayClient

logger = logging.getLogger(__name__)


class GatewayEmbeddings(Embeddings):
    """LangChain embeddings that route through the Sage Sanctum LLM gateway.

    Injects SPIFFE JWT + TraT headers on every request. Communicates
    via Unix socket (production) or TCP (dev), using the same
    ``GatewayHttpClient`` as ``GatewayChatModel``.

    Args:
        model: Embedding model name (e.g. ``"text-embedding-3-small"``).
        gateway_client: Gateway client providing credentials.
        http_client: HTTP client for gateway communication.
    """

    def __init__(
        self,
        model: str,
        gateway_client: GatewayClient,
        http_client: GatewayHttpClient,
        provider: str = "openai",
    ) -> None:
        self._model = model
        self._gateway_client = gateway_client
        self._http_client = http_client
        self._provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts via the gateway.

        Sends all texts in a single request to ``/v1/embeddings``.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            GatewayError: On HTTP errors or malformed responses.
        """
        if not texts:
            return []

        response_data = self._call_embeddings(texts)

        # Parse embeddings from OpenAI-format response
        data = response_data.get("data", [])
        if len(data) != len(texts):
            raise GatewayError(
                f"Expected {len(texts)} embeddings, got {len(data)}"
            )

        # Sort by index to ensure correct ordering
        sorted_data = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text via the gateway.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.

        Raises:
            GatewayError: On HTTP errors or malformed responses.
        """
        response_data = self._call_embeddings([text])

        data = response_data.get("data", [])
        if not data:
            raise GatewayError("Gateway returned no embeddings")

        return data[0]["embedding"]

    def _call_embeddings(self, texts: list[str]) -> dict[str, Any]:
        """Make a /v1/embeddings request to the gateway.

        Fetches fresh credentials per request (SPIFFE JWT + TraT may rotate).

        Args:
            texts: Input texts to embed.

        Returns:
            Parsed JSON response body.
        """
        creds = self._gateway_client.get_credentials()

        request_body = {
            "model": self._model,
            "input": texts,
        }

        headers = creds.auth_headers()
        headers["X-Provider"] = self._provider

        try:
            response = self._http_client.request(
                method="POST",
                path="/v1/embeddings",
                headers=headers,
                body=request_body,
            )
        except GatewayError:
            raise
        except Exception as e:
            raise GatewayError(f"Embeddings request failed: {e}") from e

        if response.status != 200:
            raise GatewayError(
                f"Gateway returned status {response.status}: {response.data}"
            )

        try:
            return json.loads(response.data)
        except json.JSONDecodeError as e:
            raise GatewayError(
                f"Invalid JSON response from gateway: {e}"
            ) from e


def create_embeddings_for_gateway(
    model: str,
    gateway_client: GatewayClient,
    provider: str = "openai",
) -> Embeddings:
    """Create an embeddings client for the current gateway mode.

    In gateway mode (UDS/SPIFFE), returns a ``GatewayEmbeddings`` instance.
    In direct mode, returns ``OpenAIEmbeddings`` with env-var API keys.

    Args:
        model: Embedding model name (e.g. ``"text-embedding-3-small"``).
        gateway_client: Gateway client (determines gateway vs. direct mode).
        provider: Provider name for the ``X-Provider`` header. Defaults to ``"openai"``.

    Returns:
        A LangChain ``Embeddings`` instance.
    """
    if gateway_client.is_gateway_mode:
        endpoint = gateway_client.get_endpoint(provider)

        if endpoint.startswith("unix://"):
            socket_path = endpoint[len("unix://"):]
            http_client = GatewayHttpClient(socket_path=socket_path)
        else:
            from urllib.parse import urlparse

            parsed = urlparse(endpoint)
            http_client = GatewayHttpClient(
                host=parsed.hostname or "localhost",
                port=parsed.port or 8080,
            )

        return GatewayEmbeddings(
            model=model,
            gateway_client=gateway_client,
            http_client=http_client,
            provider=provider,
        )

    # Direct mode — fall back to OpenAIEmbeddings with env-var keys
    import os

    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, Any] = {"model": model}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key

    return OpenAIEmbeddings(**kwargs)
