"""GatewayChatModel: LangChain BaseChatModel wrapping the LLM gateway.

In gateway mode, injects SPIFFE JWT + TraT auth headers per request.
In direct mode, creates a standard ChatLiteLLM with direct API keys.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_litellm import ChatLiteLLM

from ..errors import GatewayError
from ..gateway.http import GatewayHttpClient
from .model_ref import ModelRef

if TYPE_CHECKING:
    from ..gateway.client import GatewayClient

logger = logging.getLogger(__name__)


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain messages to OpenAI-format dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "user"
        result.append({"role": role, "content": msg.content})
    return result


class GatewayChatModel(BaseChatModel):
    """LangChain chat model that routes through the Sage Sanctum LLM gateway.

    Injects SPIFFE JWT + TraT headers on every request. Communicates
    via Unix socket (production) or TCP (dev).

    Attributes:
        model_ref: Canonical model reference (provider + model name).
        gateway_client: Gateway client providing credentials.
        http_client: HTTP client for gateway communication.
        temperature: Sampling temperature. Defaults to ``0.0`` (deterministic).
    """

    model_ref: ModelRef
    gateway_client: Any  # GatewayClient - using Any to avoid pydantic issues
    http_client: Any  # GatewayHttpClient
    temperature: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "sage-sanctum-gateway"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": str(self.model_ref),
            "temperature": self.temperature,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Send messages through the gateway and return a chat response.

        Fetches fresh credentials, builds an OpenAI-format request, injects
        authentication headers, and sends the request via the HTTP client.

        Args:
            messages: LangChain message objects to send.
            stop: Optional stop sequences.
            run_manager: LangChain callback manager (unused).

        Returns:
            ``ChatResult`` with the assistant's response.

        Raises:
            GatewayError: On HTTP errors or malformed responses.
        """
        # Get fresh credentials
        creds = self.gateway_client.get_credentials()

        # Build request
        message_dicts = _messages_to_dicts(messages)
        request_body = {
            "model": self.model_ref.model,
            "messages": message_dicts,
            "temperature": self.temperature,
        }
        if stop:
            request_body["stop"] = stop

        # Build headers with auth
        headers = creds.auth_headers()
        headers["X-Provider"] = self.model_ref.provider

        # Send to gateway
        try:
            response = self.http_client.request(
                method="POST",
                path="/v1/chat/completions",
                headers=headers,
                body=request_body,
            )
        except Exception as e:
            raise GatewayError(f"Gateway request failed: {e}") from e

        if response.status != 200:
            raise GatewayError(
                f"Gateway returned status {response.status}: {response.data}"
            )

        # Parse OpenAI-format response
        try:
            data = json.loads(response.data)
        except json.JSONDecodeError as e:
            raise GatewayError(f"Invalid JSON response from gateway: {e}") from e

        choices = data.get("choices", [])
        if not choices:
            raise GatewayError("Gateway returned no choices")

        content = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=content),
                    generation_info={
                        "usage": usage,
                        "model": data.get("model", str(self.model_ref)),
                    },
                )
            ]
        )


def create_llm_for_gateway(
    model_ref: ModelRef,
    gateway_client: GatewayClient,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create an LLM client appropriate for the current gateway mode.

    Args:
        model_ref: The model to use.
        gateway_client: Gateway client (determines gateway vs. direct mode).
        temperature: Sampling temperature. Defaults to ``0.0``.

    Returns:
        ``GatewayChatModel`` when ``gateway_client.is_gateway_mode`` is ``True``,
        ``ChatLiteLLM`` otherwise.
    """
    if gateway_client.is_gateway_mode:
        # Determine connection method from endpoint
        endpoint = gateway_client.get_endpoint(model_ref.provider)
        if endpoint.startswith("unix://"):
            socket_path = endpoint[len("unix://"):]
            http_client = GatewayHttpClient(socket_path=socket_path)
        else:
            # Parse host:port from URL
            from urllib.parse import urlparse

            parsed = urlparse(endpoint)
            http_client = GatewayHttpClient(
                host=parsed.hostname or "localhost",
                port=parsed.port or 8080,
            )

        return GatewayChatModel(
            model_ref=model_ref,
            gateway_client=gateway_client,
            http_client=http_client,
            temperature=temperature,
        )
    else:
        # Direct mode — use ChatLiteLLM with API key
        from ..gateway.client import DirectProviderClient

        assert isinstance(gateway_client, DirectProviderClient)
        api_key = gateway_client.get_api_key(model_ref.provider)

        kwargs: dict[str, Any] = {
            "model": model_ref.for_litellm,
            "temperature": temperature,
            "api_key": api_key,
            "streaming": False,
        }

        endpoint = gateway_client.get_endpoint(model_ref.provider)
        if not endpoint.startswith("https://api."):
            kwargs["api_base"] = endpoint

        return ChatLiteLLM(**kwargs)
