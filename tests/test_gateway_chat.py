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
        """include_raw=True constructs a chain with RunnableAssign (not plain parser)."""
        from langchain_core.runnables.passthrough import RunnableAssign

        http = self._make_http_client('{"severity": "high", "message": "test"}')
        model = GatewayChatModel(
            model_ref=ModelRef(provider="openai", model="gpt-4o"),
            gateway_client=mock_gateway_client,
            http_client=http,
        )
        chain = model.with_structured_output(_Finding, include_raw=True)
        assert isinstance(chain, Runnable)
        # The last step in the include_raw chain is RunnableAssign (not PydanticOutputParser)
        assert isinstance(chain.last, RunnableAssign)

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
