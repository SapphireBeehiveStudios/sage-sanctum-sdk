"""Tests for gateway client implementations."""

import os

import pytest

from sage_sanctum.errors import ConfigurationError
from sage_sanctum.gateway.client import DirectProviderClient, SpiffeGatewayClient


class TestDirectProviderClient:
    def test_requires_allow_direct(self, monkeypatch):
        monkeypatch.delenv("SAGE_SANCTUM_ALLOW_DIRECT", raising=False)
        with pytest.raises(ConfigurationError, match="SAGE_SANCTUM_ALLOW_DIRECT"):
            DirectProviderClient()

    def test_creates_with_flag(self, monkeypatch):
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        client = DirectProviderClient()
        assert not client.is_gateway_mode

    def test_get_credentials_empty(self, monkeypatch):
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        client = DirectProviderClient()
        creds = client.get_credentials()
        assert creds.spiffe_jwt == ""
        assert creds.trat == ""

    def test_get_endpoint_defaults(self, monkeypatch):
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.delenv("GOOGLE_BASE_URL", raising=False)

        client = DirectProviderClient()
        assert "openai.com" in client.get_endpoint("openai")
        assert "anthropic.com" in client.get_endpoint("anthropic")
        assert "googleapis.com" in client.get_endpoint("google")

    def test_get_api_key(self, monkeypatch):
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        client = DirectProviderClient()
        assert client.get_api_key("openai") == "sk-test"

    def test_get_api_key_missing_raises(self, monkeypatch):
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        client = DirectProviderClient()
        with pytest.raises(ConfigurationError, match="API key not set"):
            client.get_api_key("openai")

    def test_get_trat_returns_none(self, monkeypatch):
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        client = DirectProviderClient()
        assert client.get_trat() is None


class TestSpiffeGatewayClient:
    def test_is_gateway_mode(self, tmp_path):
        from unittest.mock import MagicMock

        jwt_source = MagicMock()
        trat_client = MagicMock()

        client = SpiffeGatewayClient(
            jwt_source=jwt_source,
            trat_client=trat_client,
        )
        assert client.is_gateway_mode

    def test_get_credentials(self):
        from unittest.mock import MagicMock

        from sage_sanctum.auth.trat import TransactionToken

        jwt_source = MagicMock()
        jwt_source.get_token.return_value = "spiffe-jwt-token"

        trat_client = MagicMock()
        mock_trat = MagicMock(spec=TransactionToken)
        mock_trat.raw = "trat-jwt-token"
        trat_client.get_token.return_value = mock_trat

        client = SpiffeGatewayClient(
            jwt_source=jwt_source,
            trat_client=trat_client,
        )
        creds = client.get_credentials()
        assert creds.spiffe_jwt == "spiffe-jwt-token"
        assert creds.trat == "trat-jwt-token"

    def test_get_endpoint_with_socket(self, tmp_path):
        from unittest.mock import MagicMock

        client = SpiffeGatewayClient(
            jwt_source=MagicMock(),
            trat_client=MagicMock(),
            gateway_socket=tmp_path / "gateway.sock",
        )
        endpoint = client.get_endpoint("openai")
        assert endpoint.startswith("unix://")

    def test_get_endpoint_from_env(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy:8080/v1")

        client = SpiffeGatewayClient(
            jwt_source=MagicMock(),
            trat_client=MagicMock(),
        )
        endpoint = client.get_endpoint("openai")
        assert endpoint == "http://proxy:8080/v1"
