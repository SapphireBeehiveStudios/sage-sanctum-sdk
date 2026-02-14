"""GatewayChatModel: LangChain BaseChatModel wrapping the LLM gateway.

In gateway mode, injects SPIFFE JWT + TraT auth headers per request.
In direct mode, creates a standard ChatLiteLLM with direct API keys.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

from ..errors import GatewayError, RateLimitError
from ..gateway.http import GatewayHttpClient
from ..logging import get_logger
from .model_ref import ModelRef

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput

    from ..gateway.client import GatewayClient

logger = get_logger(__name__)


def _strip_schema_extras(schema: dict[str, Any]) -> None:
    """Normalize a Pydantic JSON schema for OpenAI strict structured output.

    OpenAI's structured output rejects schemas containing ``title``,
    ``default``, and ``$defs``/``definitions`` at any nesting level.
    It also requires ``additionalProperties: false`` on every object.

    This uses schema-aware recursion rather than blindly iterating all dict
    values, so property names like ``title`` aren't accidentally stripped.
    """
    for key in ("title", "default"):
        schema.pop(key, None)

    # OpenAI strict mode requires additionalProperties: false on all objects,
    # and every property must appear in the "required" array.
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())

    # Inline $defs / definitions — OpenAI doesn't support $ref
    defs = schema.pop("$defs", schema.pop("definitions", None))

    # Recurse into known schema sub-structures (not the properties dict itself
    # since its keys are property names, not metadata)
    if "properties" in schema:
        for prop_schema in schema["properties"].values():
            if isinstance(prop_schema, dict):
                _strip_schema_extras(prop_schema)

    if "items" in schema and isinstance(schema["items"], dict):
        _strip_schema_extras(schema["items"])

    for keyword in ("anyOf", "allOf", "oneOf"):
        if keyword in schema and isinstance(schema[keyword], list):
            for item in schema[keyword]:
                if isinstance(item, dict):
                    _strip_schema_extras(item)

    # After stripping nested schemas, resolve $ref if we had defs
    if defs is not None:
        _resolve_refs(schema, defs)


def _resolve_refs(schema: dict[str, Any], defs: dict[str, Any]) -> None:
    """Resolve ``$ref`` pointers against the given definitions dict."""
    for key, value in list(schema.items()):
        if isinstance(value, dict):
            ref = value.get("$ref")
            if ref and isinstance(ref, str) and ref.startswith("#/$defs/"):
                ref_name = ref.split("/")[-1]
                if ref_name in defs:
                    resolved = dict(defs[ref_name])
                    _strip_schema_extras(resolved)
                    schema[key] = resolved
            else:
                _resolve_refs(value, defs)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    ref = item.get("$ref")
                    if ref and isinstance(ref, str) and ref.startswith("#/$defs/"):
                        ref_name = ref.split("/")[-1]
                        if ref_name in defs:
                            resolved = dict(defs[ref_name])
                            _strip_schema_extras(resolved)
                            value[i] = resolved
                    else:
                        _resolve_refs(item, defs)


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to OpenAI-format dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            d: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else tc["args"]
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(d)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })
        else:
            result.append({"role": "user", "content": msg.content})
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
        max_tokens: Maximum tokens to generate. ``None`` lets the provider decide.
        top_p: Nucleus sampling threshold. ``None`` lets the provider decide.
        seed: Random seed for deterministic generation. ``None`` for non-deterministic.
    """

    model_ref: ModelRef
    gateway_client: Any  # GatewayClient - using Any to avoid pydantic issues
    http_client: Any  # GatewayHttpClient
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None

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

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        """Return a Runnable that produces structured Pydantic output.

        For OpenAI models, uses native JSON schema mode
        (``response_format.type = "json_schema"``).  For other providers,
        uses ``json_object`` mode and relies on the parser for validation.

        Args:
            schema: Pydantic model class to parse responses into.
            include_raw: If ``True``, return a dict with ``raw``, ``parsed``,
                and ``parsing_error`` keys.

        Returns:
            A LangChain ``Runnable`` that yields ``schema`` instances (or
            raw+parsed dicts when *include_raw* is set).
        """
        parser = PydanticOutputParser(pydantic_object=schema)

        if self.model_ref.provider == "openai":
            json_schema = schema.model_json_schema()
            _strip_schema_extras(json_schema)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": True,
                },
            }
            llm = self.bind(response_format=response_format)
        else:
            llm = self.bind(response_format={"type": "json_object"})

        if include_raw:
            def _parse_with_raw(ai_message: AIMessage) -> dict:
                parsed = _safe_parse(parser, ai_message)
                error = _safe_parse_error(parser, ai_message)
                return {"raw": ai_message, "parsed": parsed, "parsing_error": error}

            return llm | _parse_with_raw

        return llm | parser

    def _build_request_body(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the OpenAI-format request body."""
        message_dicts = _messages_to_dicts(messages)
        request_body: dict[str, Any] = {
            "model": self.model_ref.model,
            "messages": message_dicts,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_body["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            request_body["top_p"] = self.top_p
        if self.seed is not None:
            request_body["seed"] = self.seed
        if stop:
            request_body["stop"] = stop
        # Forward response_format from bind() kwargs for structured output
        if "response_format" in kwargs:
            request_body["response_format"] = kwargs["response_format"]
        # Forward tools from bind() kwargs
        if "tools" in kwargs:
            request_body["tools"] = kwargs["tools"]
        return request_body

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
        request_body = self._build_request_body(messages, stop, **kwargs)

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
        except RateLimitError:
            raise
        except GatewayError:
            raise
        except Exception as e:
            raise GatewayError(f"Gateway request failed: {e}") from e

        if response.status == 429:
            raise RateLimitError(
                f"Rate limited (429): {response.data}"
            )

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

        message_data = choices[0].get("message", {})
        content = message_data.get("content", "")
        usage = data.get("usage", {})

        # Build AIMessage with tool calls if present
        ai_kwargs: dict[str, Any] = {"content": content}
        raw_tool_calls = message_data.get("tool_calls")
        if raw_tool_calls:
            ai_kwargs["tool_calls"] = [
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"])
                    if isinstance(tc["function"]["arguments"], str)
                    else tc["function"]["arguments"],
                }
                for tc in raw_tool_calls
            ]

        # Add usage_metadata for LangChain native usage tracking
        if usage:
            ai_kwargs["usage_metadata"] = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        ai_message = AIMessage(**ai_kwargs)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=ai_message,
                    generation_info={
                        "usage": usage,
                        "model": data.get("model", str(self.model_ref)),
                    },
                )
            ]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions from the gateway via SSE.

        Sends a streaming request with ``"stream": true`` and yields
        ``ChatGenerationChunk`` objects as SSE events arrive.

        NOTE: Streaming quality depends on the proxy passing through SSE
        events without buffering. If the proxy buffers, events will arrive
        in batches rather than individually.

        Args:
            messages: LangChain message objects to send.
            stop: Optional stop sequences.
            run_manager: LangChain callback manager.

        Yields:
            ``ChatGenerationChunk`` for each SSE event.

        Raises:
            GatewayError: On HTTP errors or malformed responses.
        """
        creds = self.gateway_client.get_credentials()
        request_body = self._build_request_body(messages, stop, **kwargs)
        request_body["stream"] = True

        headers = creds.auth_headers()
        headers["X-Provider"] = self.model_ref.provider

        try:
            line_iter = self.http_client.request_stream(
                method="POST",
                path="/v1/chat/completions",
                headers=headers,
                body=request_body,
            )
        except RateLimitError:
            raise
        except GatewayError:
            raise
        except Exception as e:
            raise GatewayError(f"Gateway stream failed: {e}") from e

        for raw_line in line_iter:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=content)
                )
                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk


def _safe_parse(parser: PydanticOutputParser, message: AIMessage) -> BaseModel | None:
    """Parse a message, returning None on failure."""
    try:
        return parser.parse(message.content)
    except Exception:
        return None


def _safe_parse_error(parser: PydanticOutputParser, message: AIMessage) -> str | None:
    """Parse a message, returning the error string on failure."""
    try:
        parser.parse(message.content)
        return None
    except Exception as e:
        return str(e)


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
