"""Tests for GatewayChatModel header injection."""

import json
from unittest.mock import MagicMock, patch

import pytest

from sage_sanctum.auth.credentials import GatewayCredentials
from sage_sanctum.errors import GatewayError
from sage_sanctum.gateway.http import HttpResponse
from sage_sanctum.llm.gateway_chat import GatewayChatModel, _messages_to_dicts
from sage_sanctum.llm.model_ref import ModelRef

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


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
