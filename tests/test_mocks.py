"""Tests for testing utilities and mocks."""

from langchain_core.messages import HumanMessage

from sage_sanctum.testing.mocks import MockGatewayClient, MockLLM, MockTratClient


class TestMockGatewayClient:
    def test_default_not_gateway_mode(self):
        client = MockGatewayClient()
        assert not client.is_gateway_mode

    def test_gateway_mode(self):
        client = MockGatewayClient(is_gateway=True)
        assert client.is_gateway_mode

    def test_credentials(self):
        client = MockGatewayClient()
        creds = client.get_credentials()
        assert creds.spiffe_jwt == "mock-spiffe-jwt"
        assert creds.trat == "mock-trat-jwt"

    def test_endpoints(self):
        client = MockGatewayClient()
        assert "localhost" in client.get_endpoint("openai")

    def test_custom_endpoints(self):
        client = MockGatewayClient(
            endpoints={"openai": "http://custom:8080/v1"}
        )
        assert client.get_endpoint("openai") == "http://custom:8080/v1"

    def test_trat_none_by_default(self):
        client = MockGatewayClient()
        assert client.get_trat() is None

    def test_trat_with_value(self):
        mock_trat_client = MockTratClient()
        trat = mock_trat_client.get_token()
        client = MockGatewayClient(trat=trat)
        assert client.get_trat() is not None
        assert client.get_trat().txn == "run_test123"


class TestMockLLM:
    def test_default_response(self):
        llm = MockLLM()
        result = llm.invoke([HumanMessage(content="hello")])
        assert result.content == "Mock response"

    def test_canned_responses(self):
        llm = MockLLM(responses=["first", "second"])
        r1 = llm.invoke([HumanMessage(content="a")])
        r2 = llm.invoke([HumanMessage(content="b")])
        assert r1.content == "first"
        assert r2.content == "second"

    def test_tracks_calls(self):
        llm = MockLLM(responses=["reply"])
        msgs = [HumanMessage(content="test")]
        llm.invoke(msgs)
        assert len(llm.calls) == 1

    def test_fallback_to_default_after_responses(self):
        llm = MockLLM(responses=["only one"])
        llm.invoke([HumanMessage(content="a")])
        r2 = llm.invoke([HumanMessage(content="b")])
        assert r2.content == "Mock response"

    def test_llm_type(self):
        llm = MockLLM()
        assert llm._llm_type == "mock"


class TestMockTratClient:
    def test_default_trat(self):
        client = MockTratClient()
        token = client.get_token()
        assert token.txn == "run_test123"
        assert token.tctx.org_id == "12345"
        assert token.tctx.allowed_models.triage == ["openai:gpt-4o-mini"]

    def test_custom_allowed_models(self):
        from sage_sanctum.auth.trat import AllowedModels

        am = AllowedModels(triage=["google:gemini-2.0-flash"])
        client = MockTratClient(allowed_models=am)
        token = client.get_token()
        assert token.tctx.allowed_models.triage == ["google:gemini-2.0-flash"]

    def test_invalidate(self):
        client = MockTratClient()
        client.invalidate()  # Should not raise
