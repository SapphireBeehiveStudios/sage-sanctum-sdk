"""Tests for GatewayChatModel header injection and create_llm_for_gateway."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from sage_sanctum.auth.credentials import GatewayCredentials
from sage_sanctum.errors import GatewayError, RateLimitError
from sage_sanctum.gateway.http import GatewayHttpClient, HttpResponse
from sage_sanctum.llm.gateway_chat import (
    GatewayChatModel,
    _messages_to_dicts,
    _safe_parse,
    _safe_parse_error,
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

    def test_unknown_message_type_defaults_to_user(self):
        """Non-standard message types fall back to 'user' role."""
        from langchain_core.messages import BaseMessage

        class CustomMessage(BaseMessage):
            type: str = "custom"

        result = _messages_to_dicts([CustomMessage(content="hi")])
        assert result == [{"role": "user", "content": "hi"}]


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

        # Verify request body structure
        body = call_args.kwargs.get("body", {})
        assert "messages" in body
        assert body["messages"] == [{"role": "user", "content": "test"}]

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
        # Verify host and port were parsed from the endpoint URL
        assert result.http_client._host == "gateway"
        assert result.http_client._port == 8080

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

    def test_gateway_mode_tcp_passes_gateway_client(self):
        """Gateway client is correctly propagated to GatewayChatModel in TCP mode."""
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "http://gateway:8080/v1"

        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert result.gateway_client is client

    def test_gateway_mode_tcp_no_port_uses_default(self):
        """TCP endpoint without port falls back to 8080."""
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "http://gateway/v1"

        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert result.http_client._host == "gateway"
        assert result.http_client._port == 8080

    def test_gateway_mode_tcp_no_hostname_uses_localhost(self):
        """TCP endpoint without hostname falls back to localhost."""
        client = MagicMock()
        client.is_gateway_mode = True
        client.get_endpoint.return_value = "http://:9090/v1"

        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert result.http_client._host == "localhost"
        assert result.http_client._port == 9090

    def test_direct_mode_returns_chat_litellm(self, monkeypatch):
        """Direct mode returns ChatLiteLLM with correct configuration."""
        from langchain_litellm import ChatLiteLLM

        from sage_sanctum.gateway.client import DirectProviderClient

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        client = DirectProviderClient()
        model_ref = ModelRef(provider="openai", model="gpt-4o")
        result = create_llm_for_gateway(model_ref, client)

        assert isinstance(result, ChatLiteLLM)
        assert result.model == "gpt-4o"  # OpenAI: no prefix in for_litellm
        assert result.streaming is False
        assert result.api_key == "sk-test-key"

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

    def test_direct_mode_calls_get_api_key_with_provider(self, monkeypatch):
        """get_api_key is called with the correct provider name."""
        from sage_sanctum.gateway.client import DirectProviderClient

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        client = DirectProviderClient()
        # Spy on get_api_key to verify it's called with the right arg
        original_get_api_key = client.get_api_key
        calls = []

        def spy_get_api_key(provider):
            calls.append(provider)
            return original_get_api_key(provider)

        client.get_api_key = spy_get_api_key

        model_ref = ModelRef(provider="openai", model="gpt-4o")
        create_llm_for_gateway(model_ref, client)

        assert calls == ["openai"]

    def test_direct_mode_calls_get_endpoint_with_provider(self, monkeypatch):
        """get_endpoint is called with the correct provider name in direct mode."""
        from sage_sanctum.gateway.client import DirectProviderClient

        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        client = DirectProviderClient()
        original_get_endpoint = client.get_endpoint
        calls = []

        def spy_get_endpoint(provider):
            calls.append(provider)
            return original_get_endpoint(provider)

        client.get_endpoint = spy_get_endpoint

        model_ref = ModelRef(provider="openai", model="gpt-4o")
        create_llm_for_gateway(model_ref, client)

        assert calls == ["openai"]


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

    def test_type_object_without_properties(self):
        """type:object without properties still gets additionalProperties: false."""
        schema = {"type": "object"}
        _strip_schema_extras(schema)
        assert schema["additionalProperties"] is False
        assert "required" not in schema

    def test_additional_properties_key_exact(self):
        """Verify the exact key name 'additionalProperties' is set (case-sensitive)."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        _strip_schema_extras(schema)
        assert "additionalProperties" in schema
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["a"]

    def test_strips_extras_from_array_items(self):
        """Array schema items are recursively stripped."""
        schema = {
            "type": "array",
            "title": "MyList",
            "items": {"title": "Item", "type": "string", "default": "x"},
        }
        _strip_schema_extras(schema)
        assert "title" not in schema
        assert "title" not in schema["items"]
        assert "default" not in schema["items"]
        assert schema["items"]["type"] == "string"

    def test_strips_extras_from_anyof(self):
        """anyOf items are recursively stripped."""
        schema = {
            "anyOf": [
                {"title": "Str", "type": "string"},
                {"title": "Null", "type": "null"},
            ]
        }
        _strip_schema_extras(schema)
        assert "title" not in schema["anyOf"][0]
        assert "title" not in schema["anyOf"][1]

    def test_strips_extras_from_allof_and_oneof(self):
        """allOf and oneOf items are recursively stripped."""
        schema = {
            "allOf": [{"title": "A", "type": "string"}],
            "oneOf": [{"title": "B", "type": "integer"}],
        }
        _strip_schema_extras(schema)
        assert "title" not in schema["allOf"][0]
        assert "title" not in schema["oneOf"][0]

    def test_resolves_refs_in_anyof_list(self):
        """$ref inside anyOf lists should be resolved."""
        schema = {
            "type": "object",
            "$defs": {
                "Status": {"title": "Status", "type": "string", "enum": ["ok", "err"]},
            },
            "properties": {
                "value": {
                    "anyOf": [
                        {"$ref": "#/$defs/Status"},
                        {"type": "null"},
                    ],
                },
            },
        }
        _strip_schema_extras(schema)
        assert "$defs" not in schema
        any_of = schema["properties"]["value"]["anyOf"]
        assert any_of[0]["type"] == "string"
        assert any_of[0]["enum"] == ["ok", "err"]
        assert "title" not in any_of[0]
        assert any_of[1]["type"] == "null"

    def test_resolves_multiple_refs_in_list(self):
        """Multiple $ref entries in a list should all be resolved."""
        schema = {
            "type": "object",
            "$defs": {
                "Foo": {"title": "Foo", "type": "string"},
                "Bar": {"title": "Bar", "type": "integer"},
            },
            "properties": {
                "value": {
                    "oneOf": [
                        {"$ref": "#/$defs/Foo"},
                        {"$ref": "#/$defs/Bar"},
                    ],
                },
            },
        }
        _strip_schema_extras(schema)
        assert "$defs" not in schema
        one_of = schema["properties"]["value"]["oneOf"]
        assert one_of[0]["type"] == "string"
        assert one_of[1]["type"] == "integer"
        assert "title" not in one_of[0]
        assert "title" not in one_of[1]

    def test_properties_without_type_object(self):
        """Schema with 'properties' but no 'type' still gets additionalProperties."""
        schema = {
            "properties": {
                "name": {"title": "Name", "type": "string"},
            },
        }
        _strip_schema_extras(schema)
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["name"]
        assert "title" not in schema["properties"]["name"]

    def test_resolves_definitions_key(self):
        """Schemas using 'definitions' (not '$defs') are also resolved."""
        schema = {
            "type": "object",
            "definitions": {
                "Color": {"title": "Color", "type": "string", "enum": ["red", "blue"]},
            },
            "properties": {
                "color": {"$ref": "#/$defs/Color"},
            },
        }
        _strip_schema_extras(schema)
        assert "definitions" not in schema
        assert schema["properties"]["color"]["type"] == "string"
        assert schema["properties"]["color"]["enum"] == ["red", "blue"]
        assert "title" not in schema["properties"]["color"]

    def test_nested_dict_in_anyof_without_ref(self):
        """Non-$ref dict items in anyOf lists get recursive resolution of nested $ref."""
        schema = {
            "type": "object",
            "$defs": {
                "Inner": {"title": "Inner", "type": "string"},
            },
            "properties": {
                "field": {
                    "anyOf": [
                        {
                            "type": "object",
                            "title": "Nested",
                            "properties": {
                                "child": {"$ref": "#/$defs/Inner"},
                            },
                        },
                        {"type": "null"},
                    ],
                },
            },
        }
        _strip_schema_extras(schema)
        assert "$defs" not in schema
        nested = schema["properties"]["field"]["anyOf"][0]
        assert nested["properties"]["child"]["type"] == "string"
        assert "title" not in nested["properties"]["child"]


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

    def test_include_raw_returns_runnable_openai(self, mock_gateway_client):
        """include_raw=True constructs a chain with a lambda (not plain parser)."""
        http = self._make_http_client('{"severity": "high", "message": "test"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding, include_raw=True)
        assert isinstance(chain, Runnable)

        # invoke should return a dict with raw, parsed, parsing_error keys
        result = chain.invoke([HumanMessage(content="analyze")])
        assert isinstance(result, dict)
        assert "raw" in result
        assert "parsed" in result
        assert "parsing_error" in result
        assert isinstance(result["raw"], AIMessage)
        assert isinstance(result["parsed"], _Finding)
        assert result["parsing_error"] is None

    def test_include_raw_returns_runnable_non_openai(self, mock_gateway_client):
        """include_raw=True works for non-OpenAI providers too."""
        http = self._make_http_client('{"severity": "low", "message": "ok"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="anthropic", model="claude-3-5-sonnet"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding, include_raw=True)
        assert isinstance(chain, Runnable)

    def test_openai_response_format_schema_key(self, mock_gateway_client):
        """The OpenAI response_format includes the 'schema' key with the JSON schema."""
        http = self._make_http_client('{"severity": "high", "message": "test"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding)
        chain.invoke([HumanMessage(content="analyze")])

        call_args = http.request.call_args
        body = call_args.kwargs.get("body", {})
        rf = body["response_format"]
        assert "schema" in rf["json_schema"]
        json_schema = rf["json_schema"]["schema"]
        assert "properties" in json_schema
        assert "severity" in json_schema["properties"]
        assert "message" in json_schema["properties"]

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


# ---------------------------------------------------------------------------
# _safe_parse / _safe_parse_error
# ---------------------------------------------------------------------------


class TestSafeParseFunctions:
    """Test _safe_parse and _safe_parse_error used by include_raw path."""

    def _make_parser(self):
        from langchain_core.output_parsers import PydanticOutputParser
        return PydanticOutputParser(pydantic_object=_Finding)

    def test_safe_parse_valid(self):
        parser = self._make_parser()
        msg = AIMessage(content='{"severity": "high", "message": "test"}')
        result = _safe_parse(parser, msg)
        assert isinstance(result, _Finding)
        assert result.severity == "high"
        assert result.message == "test"

    def test_safe_parse_invalid_returns_none(self):
        parser = self._make_parser()
        msg = AIMessage(content="not json")
        result = _safe_parse(parser, msg)
        assert result is None

    def test_safe_parse_error_valid_returns_none(self):
        parser = self._make_parser()
        msg = AIMessage(content='{"severity": "high", "message": "test"}')
        result = _safe_parse_error(parser, msg)
        assert result is None

    def test_safe_parse_error_invalid_returns_string(self):
        parser = self._make_parser()
        msg = AIMessage(content="not json")
        result = _safe_parse_error(parser, msg)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# include_raw end-to-end
# ---------------------------------------------------------------------------


class TestIncludeRawEndToEnd:
    """Test that include_raw=True returns raw, parsed, and parsing_error."""

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

    def test_include_raw_success_returns_parsed_model(self, mock_gateway_client):
        http = self._make_http_client('{"severity": "high", "message": "found issue"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding, include_raw=True)
        result = chain.invoke([HumanMessage(content="analyze")])

        assert isinstance(result, dict)
        assert isinstance(result["raw"], AIMessage)
        assert result["raw"].content == '{"severity": "high", "message": "found issue"}'
        assert isinstance(result["parsed"], _Finding)
        assert result["parsed"].severity == "high"
        assert result["parsed"].message == "found issue"
        assert result["parsing_error"] is None

    def test_include_raw_parse_failure_returns_error_string(self, mock_gateway_client):
        http = self._make_http_client("this is not valid json")
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding, include_raw=True)
        result = chain.invoke([HumanMessage(content="analyze")])

        assert isinstance(result, dict)
        assert isinstance(result["raw"], AIMessage)
        assert result["parsed"] is None
        assert isinstance(result["parsing_error"], str)
        assert len(result["parsing_error"]) > 0

    def test_include_raw_non_openai_provider(self, mock_gateway_client):
        http = self._make_http_client('{"severity": "low", "message": "ok"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="anthropic", model="claude-3-5-sonnet"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding, include_raw=True)
        result = chain.invoke([HumanMessage(content="analyze")])

        assert isinstance(result, dict)
        assert isinstance(result["raw"], AIMessage)
        assert isinstance(result["parsed"], _Finding)
        assert result["parsing_error"] is None


# ---------------------------------------------------------------------------
# Generation parameters (max_tokens, top_p, seed)
# ---------------------------------------------------------------------------


class TestGenerationParameters:
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
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                "model": "gpt-4o",
            }),
        )
        return client

    def test_max_tokens_forwarded_when_set(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
            max_tokens=512,
        )
        model.invoke([HumanMessage(content="test")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert body["max_tokens"] == 512

    def test_top_p_forwarded_when_set(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
            top_p=0.9,
        )
        model.invoke([HumanMessage(content="test")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert body["top_p"] == 0.9

    def test_seed_forwarded_when_set(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
            seed=42,
        )
        model.invoke([HumanMessage(content="test")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert body["seed"] == 42

    def test_all_generation_params_forwarded_together(
        self, mock_gateway_client, mock_http_client
    ):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
            max_tokens=1024,
            top_p=0.95,
            seed=7,
        )
        model.invoke([HumanMessage(content="test")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert body["max_tokens"] == 1024
        assert body["top_p"] == 0.95
        assert body["seed"] == 7

    def test_max_tokens_absent_when_none(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        model.invoke([HumanMessage(content="test")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert "max_tokens" not in body

    def test_top_p_absent_when_none(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        model.invoke([HumanMessage(content="test")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert "top_p" not in body

    def test_seed_absent_when_none(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        model.invoke([HumanMessage(content="test")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert "seed" not in body


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


class TestToolCalling:
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
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                "model": "gpt-4o",
            }),
        )
        return client

    def test_messages_to_dicts_ai_message_with_tool_calls(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "name": "get_weather",
                    "args": {"location": "NYC"},
                },
            ],
        )
        result = _messages_to_dicts([ai_msg])

        assert len(result) == 1
        d = result[0]
        assert d["role"] == "assistant"
        assert d["content"] == ""
        assert len(d["tool_calls"]) == 1

        tc = d["tool_calls"][0]
        assert tc["id"] == "call_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "NYC"}

    def test_messages_to_dicts_ai_message_without_tool_calls(self):
        ai_msg = AIMessage(content="plain response")
        result = _messages_to_dicts([ai_msg])

        assert len(result) == 1
        d = result[0]
        assert d["role"] == "assistant"
        assert d["content"] == "plain response"
        assert "tool_calls" not in d

    def test_messages_to_dicts_tool_message(self):
        tool_msg = ToolMessage(
            content='{"temp": 72}',
            tool_call_id="call_abc123",
        )
        result = _messages_to_dicts([tool_msg])

        assert len(result) == 1
        d = result[0]
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_abc123"
        assert d["content"] == '{"temp": 72}'

    def test_messages_to_dicts_full_tool_call_conversation(self):
        """Verify a complete tool-calling conversation converts correctly."""
        msgs = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "SF"},
                    },
                ],
            ),
            ToolMessage(content='{"temp": 65}', tool_call_id="call_1"),
            AIMessage(content="It's 65 degrees in SF."),
        ]
        result = _messages_to_dicts(msgs)

        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_1"
        assert result[3]["role"] == "assistant"
        assert "tool_calls" not in result[3]

    def test_response_with_tool_calls_parsed(self, mock_gateway_client):
        """Response containing tool_calls sets tool_calls on AIMessage."""
        http = MagicMock()
        http.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=json.dumps({
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "langchain"}',
                                },
                            },
                        ],
                    },
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
                "model": "gpt-4o",
            }),
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        result = model.invoke([HumanMessage(content="search for langchain")])

        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_xyz"
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["args"] == {"query": "langchain"}

    def test_tools_kwarg_forwarded_in_request_body(
        self, mock_gateway_client, mock_http_client
    ):
        """tools kwarg from bind() is forwarded in the request body."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            },
        ]

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        bound = model.bind(tools=tools)
        bound.invoke([HumanMessage(content="weather in NYC")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert body["tools"] == tools

    def test_tools_absent_when_not_bound(self, mock_gateway_client, mock_http_client):
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http_client,
        )
        model.invoke([HumanMessage(content="test")])

        body = mock_http_client.request.call_args.kwargs["body"]
        assert "tools" not in body


# ---------------------------------------------------------------------------
# Token usage tracking (usage_metadata)
# ---------------------------------------------------------------------------


class TestTokenUsageTracking:
    @pytest.fixture
    def mock_gateway_client(self):
        client = MagicMock()
        client.get_credentials.return_value = GatewayCredentials(
            spiffe_jwt="test-svid",
            trat="test-trat",
        )
        return client

    def test_usage_metadata_set_on_ai_message(self, mock_gateway_client):
        http = MagicMock()
        http.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=json.dumps({
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 10,
                    "total_tokens": 35,
                },
                "model": "gpt-4o",
            }),
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        result = model.invoke([HumanMessage(content="test")])

        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] == 25
        assert result.usage_metadata["output_tokens"] == 10
        assert result.usage_metadata["total_tokens"] == 35

    def test_usage_metadata_absent_when_no_usage_in_response(self, mock_gateway_client):
        http = MagicMock()
        http.request.return_value = HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            data=json.dumps({
                "choices": [{"message": {"content": "Hello"}}],
                "model": "gpt-4o",
            }),
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        result = model.invoke([HumanMessage(content="test")])

        # usage_metadata should be empty/absent when no usage in response
        assert not result.usage_metadata


# ---------------------------------------------------------------------------
# Streaming (_stream)
# ---------------------------------------------------------------------------


class TestStreaming:
    @pytest.fixture
    def mock_gateway_client(self):
        client = MagicMock()
        client.get_credentials.return_value = GatewayCredentials(
            spiffe_jwt="test-svid",
            trat="test-trat",
        )
        return client

    def test_stream_yields_chunks(self, mock_gateway_client):
        mock_http = MagicMock()
        sse_lines = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}',
            b"data: [DONE]",
        ]
        mock_http.request_stream.return_value = iter(sse_lines)

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http,
        )

        chunks = list(model._stream([HumanMessage(content="test")]))

        assert len(chunks) == 2
        assert isinstance(chunks[0], ChatGenerationChunk)
        assert chunks[0].message.content == "Hello"
        assert isinstance(chunks[1], ChatGenerationChunk)
        assert chunks[1].message.content == " world"

    def test_stream_done_terminates(self, mock_gateway_client):
        """data: [DONE] stops yielding even if more lines follow."""
        mock_http = MagicMock()
        sse_lines = [
            b'data: {"choices": [{"delta": {"content": "first"}}]}',
            b"data: [DONE]",
            b'data: {"choices": [{"delta": {"content": "ignored"}}]}',
        ]
        mock_http.request_stream.return_value = iter(sse_lines)

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http,
        )

        chunks = list(model._stream([HumanMessage(content="test")]))

        assert len(chunks) == 1
        assert chunks[0].message.content == "first"

    def test_stream_skips_empty_lines(self, mock_gateway_client):
        mock_http = MagicMock()
        sse_lines = [
            b"",
            b'data: {"choices": [{"delta": {"content": "hi"}}]}',
            b"",
            b"data: [DONE]",
        ]
        mock_http.request_stream.return_value = iter(sse_lines)

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http,
        )

        chunks = list(model._stream([HumanMessage(content="test")]))

        assert len(chunks) == 1
        assert chunks[0].message.content == "hi"

    def test_stream_skips_non_data_lines(self, mock_gateway_client):
        mock_http = MagicMock()
        sse_lines = [
            b": keep-alive comment",
            b'data: {"choices": [{"delta": {"content": "ok"}}]}',
            b"event: done",
            b"data: [DONE]",
        ]
        mock_http.request_stream.return_value = iter(sse_lines)

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http,
        )

        chunks = list(model._stream([HumanMessage(content="test")]))

        assert len(chunks) == 1
        assert chunks[0].message.content == "ok"

    def test_stream_skips_empty_content_deltas(self, mock_gateway_client):
        mock_http = MagicMock()
        sse_lines = [
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}',
            b'data: {"choices": [{"delta": {"content": "text"}}]}',
            b'data: {"choices": [{"delta": {"content": ""}}]}',
            b"data: [DONE]",
        ]
        mock_http.request_stream.return_value = iter(sse_lines)

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http,
        )

        chunks = list(model._stream([HumanMessage(content="test")]))

        assert len(chunks) == 1
        assert chunks[0].message.content == "text"

    def test_stream_skips_invalid_json(self, mock_gateway_client):
        mock_http = MagicMock()
        sse_lines = [
            b"data: {invalid json}",
            b'data: {"choices": [{"delta": {"content": "valid"}}]}',
            b"data: [DONE]",
        ]
        mock_http.request_stream.return_value = iter(sse_lines)

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http,
        )

        chunks = list(model._stream([HumanMessage(content="test")]))

        assert len(chunks) == 1
        assert chunks[0].message.content == "valid"

    def test_stream_sends_stream_true_in_body(self, mock_gateway_client):
        mock_http = MagicMock()
        mock_http.request_stream.return_value = iter([b"data: [DONE]"])

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http,
        )

        list(model._stream([HumanMessage(content="test")]))

        call_args = mock_http.request_stream.call_args
        body = call_args.kwargs["body"]
        assert body["stream"] is True

    def test_stream_request_exception_wrapped(self, mock_gateway_client):
        mock_http = MagicMock()
        mock_http.request_stream.side_effect = RuntimeError("socket closed")

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=mock_http,
        )

        with pytest.raises(GatewayError, match="Gateway stream failed"):
            list(model._stream([HumanMessage(content="test")]))


# ---------------------------------------------------------------------------
# Rate limiting (429 -> RateLimitError)
# ---------------------------------------------------------------------------


class TestRateLimiting:
    @pytest.fixture
    def mock_gateway_client(self):
        client = MagicMock()
        client.get_credentials.return_value = GatewayCredentials(
            spiffe_jwt="test-svid",
            trat="test-trat",
        )
        return client

    def test_429_raises_rate_limit_error(self, mock_gateway_client):
        http = MagicMock()
        http.request.return_value = HttpResponse(
            status=429,
            headers={},
            data="Too Many Requests",
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )

        with pytest.raises(RateLimitError, match="429"):
            model.invoke([HumanMessage(content="test")])

    def test_rate_limit_error_is_gateway_error_subclass(self, mock_gateway_client):
        """RateLimitError inherits from GatewayError."""
        http = MagicMock()
        http.request.return_value = HttpResponse(
            status=429,
            headers={},
            data="rate limited",
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )

        with pytest.raises(GatewayError):
            model.invoke([HumanMessage(content="test")])

    def test_rate_limit_error_has_correct_exit_code(self, mock_gateway_client):
        http = MagicMock()
        http.request.return_value = HttpResponse(
            status=429,
            headers={},
            data="rate limited",
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )

        with pytest.raises(RateLimitError) as exc_info:
            model.invoke([HumanMessage(content="test")])
        assert exc_info.value.exit_code == 31

    def test_rate_limit_error_includes_response_body(self, mock_gateway_client):
        http = MagicMock()
        http.request.return_value = HttpResponse(
            status=429,
            headers={},
            data="retry after 30 seconds",
        )

        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )

        with pytest.raises(RateLimitError, match="retry after 30 seconds"):
            model.invoke([HumanMessage(content="test")])
