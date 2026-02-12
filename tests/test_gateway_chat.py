"""Tests for GatewayChatModel header injection and create_llm_for_gateway."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from sage_sanctum.auth.credentials import GatewayCredentials
from sage_sanctum.errors import GatewayError
from sage_sanctum.gateway.http import GatewayHttpClient, HttpResponse
from sage_sanctum.llm.gateway_chat import (
    GatewayChatModel,
    _messages_to_dicts,
    _strip_schema_extras,
    create_llm_for_gateway,
)
from sage_sanctum.llm.model_ref import ModelRef


class TestMessagesToDicts:
    def test_system_message(self):
        msgs = [SystemMessage(content="sys")]
        result = _messages_to_dicts(msgs)
        assert result == [{"role": "system", "content": "sys"}]

    def test_human_message(self):
        msgs = [HumanMessage(content="hello")]
        result = _messages_to_dicts(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_ai_message(self):
        msgs = [AIMessage(content="response")]
        result = _messages_to_dicts(msgs)
        assert result == [{"role": "assistant", "content": "response"}]

    def test_mixed_messages(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ]
        result = _messages_to_dicts(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


class TestGatewayChatModel:
    @pytest.fixture
    def mock_gateway_client(self):
        client = MagicMock()
        client.get_credentials.return_value = GatewayCredentials(
            spiffe_jwt="test-svid",
            trat="test-trat",
        )
        return client

    @pytest.fixture
    def mock_http_client(self):
        client = MagicMock()
        client.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=json.dumps({
                "choices": [
                    {"message": {"content": "Hello from gateway"}}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model": "gpt-4o",
            }),
        )
        return client

    def test_injects_auth_headers(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        result = model.invoke([HumanMessage(content="test")])
        assert result.content == "Hello from gateway"

        # Verify headers were injected
        call_args = mock_http_client.request.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers["Authorization"] == "Bearer test-svid"
        assert headers["Txn-Token"] == "test-trat"
        assert headers["X-Provider"] == "openai"

    def test_gateway_error_on_non_200(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=429,
            headers={},
            data="rate limited",
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="429"):
            model.invoke([HumanMessage(content="test")])

    def test_gateway_error_on_no_choices(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={},
            data=json.dumps({"choices": []}),
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="no choices"):
            model.invoke([HumanMessage(content="test")])

    def test_llm_type(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        assert model._llm_type == "sage-sanctum-gateway"

    def test_identifying_params(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
            temperature=0.5,
        )
        params = model._identifying_params
        assert params["model"] == "openai:gpt-4o"
        assert params["temperature"] == 0.5

    def test_stop_sequences_forwarded(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        model.invoke([HumanMessage(content="test")], stop=["STOP"])

        call_args = mock_http_client.request.call_args
        body = call_args.kwargs.get("body", {})
        assert body["stop"] == ["STOP"]

    def test_temperature_in_request_body(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
            temperature=0.7,
        )
        model.invoke([HumanMessage(content="test")])

        call_args = mock_http_client.request.call_args
        body = call_args.kwargs.get("body", {})
        assert body["temperature"] == 0.7

    def test_invalid_json_response_raises(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.return_value = HttpResponse(
            status=200,
            headers={},
            data="not json",
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="Invalid JSON"):
            model.invoke([HumanMessage(content="test")])

    def test_request_exception_wrapped(self, mock_gateway_client, mock_http_client):
        mock_http_client.request.side_effect = RuntimeError("connection reset")

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )

        with pytest.raises(GatewayError, match="Gateway request failed"):
            model.invoke([HumanMessage(content="test")])

    def test_usage_in_generation_info(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        result = model._generate([HumanMessage(content="test")])
        info = result.generations[0].generation_info
        assert info["usage"]["prompt_tokens"] == 10
        assert info["model"] == "gpt-4o"

    def test_model_name_in_request_body(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="anthropic", model="claude-3-5-sonnet"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        model.invoke([HumanMessage(content="test")])

        call_args = mock_http_client.request.call_args
        body = call_args.kwargs.get("body", {})
        assert body["model"] == "claude-3-5-sonnet"

    def test_posts_to_chat_completions(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        model.invoke([HumanMessage(content="test")])

        call_args = mock_http_client.request.call_args
        assert call_args.kwargs.get("method") == "POST"
        assert call_args.kwargs.get("path") == "/v1/chat/completions"


# ---------------------------------------------------------------------------
# create_llm_for_gateway
# ---------------------------------------------------------------------------


class TestCreateLlmForGateway:
    def test_gateway_mode_unix_socket(self):
        """Gateway mode with unix:// endpoint returns GatewayChatModel."""
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "unix:///run/sockets/gateway.sock"

        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert isinstance(result, GatewayChatModel)
        assert result.model_ref == model_ref
        assert isinstance(result.http_client, GatewayHttpClient)
        assert result.http_client.is_unix_socket
        assert result.temperature == 0.0

    def test_gateway_mode_tcp_endpoint(self):
        """Gateway mode with http:// endpoint returns GatewayChatModel with TCP client."""
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "http://gateway:8080/v1"

        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert isinstance(result, GatewayChatModel)
        assert isinstance(result.http_client, GatewayHttpClient)
        assert not result.http_client.is_unix_socket

    def test_gateway_mode_custom_temperature(self):
        """Temperature is propagated to GatewayChatModel."""
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "unix:///run/sockets/gw.sock"

        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client, temperature=0.9)

        assert isinstance(result, GatewayChatModel)
        assert result.temperature == 0.9

    def test_gateway_mode_passes_provider_to_get_endpoint(self):
        """Verifies that provider name is passed to get_endpoint."""
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "unix:///run/sockets/gw.sock"

        model_ref = ModelRef(provider="anthropic", model="claude-3-5-sonnet")
        create_llm_for_gateway(model_ref, client)

        client.get_endpoint.assert_called_once_with("anthropic")

    def test_direct_mode_returns_chat_litellm(self, monkeypatch):
        """Direct mode returns ChatLiteLLM."""
        from langchain_litellm import ChatLiteLLM

        from sage_sanctum.gateway.client import DirectProviderClient

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        client = DirectProviderClient()
        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert isinstance(result, ChatLiteLLM)

    def test_direct_mode_custom_temperature(self, monkeypatch):
        """Temperature is passed through to ChatLiteLLM."""
        from langchain_litellm import ChatLiteLLM

        from sage_sanctum.gateway.client import DirectProviderClient

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        client = DirectProviderClient()
        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client, temperature=0.5)

        assert isinstance(result, ChatLiteLLM)
        assert result.temperature == 0.5

    def test_direct_mode_non_standard_endpoint_sets_api_base(self, monkeypatch):
        """Non-standard endpoint sets api_base on ChatLiteLLM."""
        from langchain_litellm import ChatLiteLLM

        from sage_sanctum.gateway.client import DirectProviderClient

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

        client = DirectProviderClient()
        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert isinstance(result, ChatLiteLLM)
        assert result.api_base == "http://localhost:11434/v1"

    def test_direct_mode_standard_endpoint_no_api_base(self, monkeypatch):
        """Standard endpoint (https://api.*) does NOT set api_base."""
        from langchain_litellm import ChatLiteLLM

        from sage_sanctum.gateway.client import DirectProviderClient

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        # Default endpoint is https://api.openai.com/v1

        client = DirectProviderClient()
        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert isinstance(result, ChatLiteLLM)
        # api_base should not be set for standard endpoints
        assert result.api_base is None


# ---------------------------------------------------------------------------
# _strip_schema_extras
# ---------------------------------------------------------------------------


class TestStripSchemaExtras:
    def test_removes_title_and_default(self):
        schema = {"title": "Foo", "default": 42, "type": "object"}
        _strip_schema_extras(schema)
        assert "title" not in schema
        assert "default" not in schema
        assert schema["type"] == "object"

    def test_recursive_strip(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"title": "Name", "type": "string", "default": ""},
            },
        }
        _strip_schema_extras(schema)
        assert "title" not in schema["properties"]["name"]
        assert "default" not in schema["properties"]["name"]

    def test_resolves_defs(self):
        schema = {
            "type": "object",
            "$defs": {
                "Inner": {"title": "Inner", "type": "string"},
            },
            "properties": {
                "child": {"$ref": "#/$defs/Inner"},
            },
        }
        _strip_schema_extras(schema)
        assert "$defs" not in schema
        # $ref should be resolved to the inner schema
        assert schema["properties"]["child"]["type"] == "string"
        assert "title" not in schema["properties"]["child"]

    def test_preserves_non_title_keys(self):
        schema = {"type": "object", "description": "A model", "title": "X"}
        _strip_schema_extras(schema)
        assert schema["description"] == "A model"
        assert "title" not in schema


# ---------------------------------------------------------------------------
# with_structured_output
# ---------------------------------------------------------------------------


class _Finding(BaseModel):
    severity: str
    message: str


class TestStructuredOutput:
    @pytest.fixture
    def mock_gateway_client(self):
        client = MagicMock()
        client.get_credentials.return_value = GatewayCredentials(
            spiffe_jwt="test-svid",
            trat="test-trat",
        )
        return client

    def _make_http_client(self, content: str) -> MagicMock:
        client = MagicMock()
        client.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=json.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                "model": "gpt-4o",
            }),
        )
        return client

    def test_returns_runnable(self, mock_gateway_client):
        http = self._make_http_client('{"severity": "high", "message": "test"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding)
        assert isinstance(chain, Runnable)

    def test_openai_sends_json_schema_response_format(self, mock_gateway_client):
        http = self._make_http_client('{"severity": "high", "message": "test"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding)
        result = chain.invoke([HumanMessage(content="analyze")])

        assert isinstance(result, _Finding)
        assert result.severity == "high"
        assert result.message == "test"

        # Verify response_format was sent in request body
        call_args = http.request.call_args
        body = call_args.kwargs.get("body", {})
        rf = body["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "_Finding"
        assert rf["json_schema"]["strict"] is True

    def test_anthropic_sends_json_object_response_format(self, mock_gateway_client):
        http = self._make_http_client('{"severity": "low", "message": "ok"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="anthropic", model="claude-3-5-sonnet"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding)
        result = chain.invoke([HumanMessage(content="analyze")])

        assert isinstance(result, _Finding)
        assert result.severity == "low"

        call_args = http.request.call_args
        body = call_args.kwargs.get("body", {})
        rf = body["response_format"]
        assert rf["type"] == "json_object"

    def test_google_sends_json_object_response_format(self, mock_gateway_client):
        http = self._make_http_client('{"severity": "medium", "message": "warn"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="google", model="gemini-2.0-flash"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding)
        result = chain.invoke([HumanMessage(content="analyze")])

        assert isinstance(result, _Finding)
        call_args = http.request.call_args
        body = call_args.kwargs.get("body", {})
        assert body["response_format"]["type"] == "json_object"

    def test_response_format_not_in_regular_request(self, mock_gateway_client):
        http = self._make_http_client("Hello")
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        model.invoke([HumanMessage(content="test")])

        call_args = http.request.call_args
        body = call_args.kwargs.get("body", {})
        assert "response_format" not in body
