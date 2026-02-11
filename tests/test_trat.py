"""Tests for Transaction Token client and parsing."""

import base64
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from sage_sanctum.auth.trat import (
    AllowedModels,
    RequesterContext,
    TransactionContext,
    TransactionToken,
    TransactionTokenClient,
)
from sage_sanctum.errors import TraTAcquisitionError, TraTExpiredError


def _make_trat(payload: dict) -> str:
    """Create a fake TraT JWT for testing."""
    header = {"alg": "EdDSA", "typ": "txntoken+jwt"}
    h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    s = base64.urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


def _sample_trat_payload() -> dict:
    return {
        "txn": "run_abc123",
        "sub": "github|org-12345",
        "scope": "scan.execute scan.upload",
        "req_wl": "spiffe://sage-sanctum.local/scanner/run_abc123",
        "iat": time.time(),
        "exp": time.time() + 300,
        "aud": "sage-sanctum.local",
        "iss": "https://tts.sage-sanctum.local",
        "tctx": {
            "run_id": "run_abc123",
            "org_id": "12345",
            "repo_url": "https://github.com/acme/repo",
            "agent_type": "sage-scanner",
            "agent_mode": "standard",
            "allowed_models": {
                "triage": ["openai:gpt-4o-mini"],
                "analysis": ["openai:gpt-4o"],
                "reasoning": ["openai:o1"],
                "embeddings": ["openai:text-embedding-3-small"],
            },
            "allowed_providers": ["openai", "anthropic"],
        },
        "rctx": {
            "trigger": "pull_request",
            "pr_number": 42,
            "actor": "dependabot[bot]",
        },
    }


class TestTransactionToken:
    def test_parse_valid_trat(self):
        payload = _sample_trat_payload()
        token = TransactionToken.from_jwt(_make_trat(payload))

        assert token.txn == "run_abc123"
        assert token.sub == "github|org-12345"
        assert token.scope == "scan.execute scan.upload"
        assert token.req_wl == "spiffe://sage-sanctum.local/scanner/run_abc123"

    def test_tctx_parsed(self):
        payload = _sample_trat_payload()
        token = TransactionToken.from_jwt(_make_trat(payload))

        assert token.tctx.run_id == "run_abc123"
        assert token.tctx.org_id == "12345"
        assert token.tctx.agent_type == "sage-scanner"
        assert token.tctx.allowed_models.triage == ["openai:gpt-4o-mini"]
        assert token.tctx.allowed_providers == ["openai", "anthropic"]

    def test_rctx_parsed(self):
        payload = _sample_trat_payload()
        token = TransactionToken.from_jwt(_make_trat(payload))

        assert token.rctx.trigger == "pull_request"
        assert token.rctx.pr_number == 42
        assert token.rctx.actor == "dependabot[bot]"

    def test_missing_txn_raises(self):
        payload = {"sub": "test", "exp": time.time() + 300}
        with pytest.raises(TraTAcquisitionError, match="txn"):
            TransactionToken.from_jwt(_make_trat(payload))

    def test_is_expired(self):
        payload = _sample_trat_payload()
        payload["exp"] = time.time() - 10  # expired
        token = TransactionToken.from_jwt(_make_trat(payload))
        assert token.is_expired

    def test_is_expired_boundary_at_current_time(self):
        """Token expiring at exactly the current time is considered expired (>= semantics)."""
        payload = _sample_trat_payload()
        now = time.time()
        payload["exp"] = now
        token = TransactionToken.from_jwt(_make_trat(payload))
        # time.time() >= exp, so this should be expired
        assert token.is_expired

    def test_raw_jwt_preserved_for_gateway_forwarding(self):
        """The raw JWT string is preserved exactly for forwarding to the gateway.

        The agent does NOT verify signatures — the gateway does. So the raw
        token must be forwarded bit-for-bit, including the original signature.
        """
        payload = _sample_trat_payload()
        original_jwt = _make_trat(payload)
        token = TransactionToken.from_jwt(original_jwt)

        # The raw field must be the exact original JWT string
        assert token.raw == original_jwt

        # Verify all three JWT segments are preserved (header.payload.signature)
        parts = token.raw.split(".")
        assert len(parts) == 3
        # The signature segment must match the original
        assert parts[2] == original_jwt.split(".")[2]

    def test_tampered_payload_still_parsed_but_raw_preserved(self):
        """If an attacker modifies the payload, the raw JWT still reflects the
        modification. The agent trusts the gateway to reject tampered tokens
        via signature verification. This test documents that the agent does NOT
        perform client-side signature validation.
        """
        payload = _sample_trat_payload()
        original_jwt = _make_trat(payload)

        # Tamper with the payload: change allowed_models
        tampered_payload = _sample_trat_payload()
        tampered_payload["tctx"]["allowed_models"]["reasoning"] = [
            "openai:o1-pro"  # escalated model
        ]
        tampered_jwt = _make_trat(tampered_payload)

        # Agent parses it without error (no signature check)
        token = TransactionToken.from_jwt(tampered_jwt)
        assert token.tctx.allowed_models.reasoning == ["openai:o1-pro"]

        # But the raw JWT is the tampered one — gateway will reject it
        assert token.raw == tampered_jwt
        assert token.raw != original_jwt

    def test_check_not_expired_raises(self):
        payload = _sample_trat_payload()
        payload["exp"] = time.time() - 10
        token = TransactionToken.from_jwt(_make_trat(payload))
        with pytest.raises(TraTExpiredError):
            token.check_not_expired()

    def test_check_not_expired_passes(self):
        payload = _sample_trat_payload()
        token = TransactionToken.from_jwt(_make_trat(payload))
        token.check_not_expired()  # Should not raise


class TestAllowedModels:
    def test_from_dict(self):
        data = {
            "triage": ["openai:gpt-4o-mini"],
            "analysis": ["openai:gpt-4o"],
        }
        am = AllowedModels.from_dict(data)
        assert am.triage == ["openai:gpt-4o-mini"]
        assert am.analysis == ["openai:gpt-4o"]
        assert am.reasoning == []
        assert am.embeddings == []

    def test_to_dict(self):
        am = AllowedModels(
            triage=["openai:gpt-4o-mini"],
            analysis=["openai:gpt-4o"],
        )
        d = am.to_dict()
        assert d["triage"] == ["openai:gpt-4o-mini"]
        assert d["embeddings"] == []


class TestTransactionTokenClient:
    def test_read_from_file(self, tmp_path):
        trat_file = tmp_path / "trat.jwt"
        payload = _sample_trat_payload()
        trat_file.write_text(_make_trat(payload))

        client = TransactionTokenClient(trat_file=trat_file)
        token = client.get_token()
        assert token.txn == "run_abc123"

    def test_file_not_found(self, tmp_path):
        client = TransactionTokenClient(trat_file=tmp_path / "missing.jwt")
        with pytest.raises(TraTAcquisitionError, match="not found"):
            client.get_token()

    def test_empty_file(self, tmp_path):
        trat_file = tmp_path / "empty.jwt"
        trat_file.write_text("")
        client = TransactionTokenClient(trat_file=trat_file)
        with pytest.raises(TraTAcquisitionError, match="empty"):
            client.get_token()

    def test_no_source_configured(self):
        client = TransactionTokenClient()
        with pytest.raises(TraTAcquisitionError, match="No TraT source"):
            client.get_token()

    def test_caches_valid_token(self, tmp_path):
        trat_file = tmp_path / "trat.jwt"
        payload = _sample_trat_payload()
        trat_file.write_text(_make_trat(payload))

        client = TransactionTokenClient(trat_file=trat_file)
        token1 = client.get_token()
        # Modify file
        trat_file.write_text("changed")
        token2 = client.get_token()
        assert token1.raw == token2.raw  # Same cached token

    def test_invalidate_clears_cache(self, tmp_path):
        trat_file = tmp_path / "trat.jwt"
        payload = _sample_trat_payload()
        trat_file.write_text(_make_trat(payload))

        client = TransactionTokenClient(trat_file=trat_file)
        client.get_token()
        client.invalidate()

        # Write new token
        payload2 = _sample_trat_payload()
        payload2["txn"] = "run_new"
        trat_file.write_text(_make_trat(payload2))

        token = client.get_token()
        assert token.txn == "run_new"

    def test_expired_token_re_acquired(self, tmp_path):
        trat_file = tmp_path / "trat.jwt"
        # First token: already expired
        payload1 = _sample_trat_payload()
        payload1["exp"] = time.time() - 10
        trat_file.write_text(_make_trat(payload1))

        client = TransactionTokenClient(trat_file=trat_file)
        # First call will acquire, then check_not_expired raises
        with pytest.raises(TraTExpiredError):
            client.get_token()

        # Write a valid token and try again
        payload2 = _sample_trat_payload()
        payload2["txn"] = "run_fresh"
        trat_file.write_text(_make_trat(payload2))

        token = client.get_token()
        assert token.txn == "run_fresh"


# ---------------------------------------------------------------------------
# _read_from_sidecar
# ---------------------------------------------------------------------------


class TestReadFromSidecar:
    def test_sidecar_socket_not_found(self, tmp_path):
        """Missing sidecar socket raises TraTAcquisitionError."""
        client = TransactionTokenClient(
            sidecar_socket=tmp_path / "missing.sock",
        )
        with pytest.raises(TraTAcquisitionError, match="socket not found"):
            client.get_token()

    def test_sidecar_successful_read(self, tmp_path):
        """Successful sidecar read returns a valid TransactionToken."""
        from sage_sanctum.gateway.http import HttpResponse

        # Create a fake socket file so the existence check passes
        sock_file = tmp_path / "auth.sock"
        sock_file.touch()

        payload = _sample_trat_payload()
        trat_jwt = _make_trat(payload)

        client = TransactionTokenClient(sidecar_socket=sock_file)

        # Track the init kwargs to verify socket_path is passed correctly
        init_kwargs_captured = {}
        mock_http = MagicMock()
        mock_http.request.return_value = HttpResponse(
            status=200, headers={}, data=trat_jwt,
        )

        def mock_init(**kwargs):
            init_kwargs_captured.update(kwargs)
            return mock_http

        with patch(
            "sage_sanctum.gateway.http.GatewayHttpClient", side_effect=mock_init,
        ):
            token = client.get_token()

        assert token.txn == "run_abc123"
        mock_http.request.assert_called_once_with("GET", "/trat")
        # Verify the sidecar socket path was passed to GatewayHttpClient
        assert init_kwargs_captured["socket_path"] == sock_file

    def test_sidecar_non_200_status(self, tmp_path):
        """Non-200 sidecar response raises TraTAcquisitionError."""
        from sage_sanctum.gateway.http import HttpResponse

        sock_file = tmp_path / "auth.sock"
        sock_file.touch()

        client = TransactionTokenClient(sidecar_socket=sock_file)

        mock_http = MagicMock()
        mock_http.request.return_value = HttpResponse(
            status=503, headers={}, data="service unavailable",
        )

        with patch(
            "sage_sanctum.gateway.http.GatewayHttpClient", return_value=mock_http,
        ):
            with pytest.raises(TraTAcquisitionError, match="status 503"):
                client.get_token()

    def test_sidecar_connection_error(self, tmp_path):
        """Connection error from sidecar is wrapped in TraTAcquisitionError."""
        from sage_sanctum.errors import GatewayUnavailableError

        sock_file = tmp_path / "auth.sock"
        sock_file.touch()

        client = TransactionTokenClient(sidecar_socket=sock_file)

        mock_http = MagicMock()
        mock_http.request.side_effect = GatewayUnavailableError("connection refused")

        with patch(
            "sage_sanctum.gateway.http.GatewayHttpClient", return_value=mock_http,
        ):
            with pytest.raises(TraTAcquisitionError, match="Failed to read TraT from sidecar"):
                client.get_token()

    def test_sidecar_strips_whitespace(self, tmp_path):
        """JWT from sidecar is stripped of whitespace."""
        from sage_sanctum.gateway.http import HttpResponse

        sock_file = tmp_path / "auth.sock"
        sock_file.touch()

        payload = _sample_trat_payload()
        trat_jwt = _make_trat(payload)

        client = TransactionTokenClient(sidecar_socket=sock_file)

        mock_http = MagicMock()
        mock_http.request.return_value = HttpResponse(
            status=200, headers={}, data=f"  {trat_jwt}\n  ",
        )

        with patch(
            "sage_sanctum.gateway.http.GatewayHttpClient", return_value=mock_http,
        ):
            token = client.get_token()

        assert token.txn == "run_abc123"

    def test_file_preferred_over_sidecar(self, tmp_path):
        """When both file and sidecar are configured, file is tried first."""
        trat_file = tmp_path / "trat.jwt"
        payload = _sample_trat_payload()
        payload["txn"] = "from_file"
        trat_file.write_text(_make_trat(payload))

        sock_file = tmp_path / "auth.sock"
        sock_file.touch()

        client = TransactionTokenClient(
            trat_file=trat_file,
            sidecar_socket=sock_file,
        )
        token = client.get_token()
        assert token.txn == "from_file"


# ---------------------------------------------------------------------------
# RequesterContext and TransactionContext
# ---------------------------------------------------------------------------


class TestRequesterContext:
    def test_from_dict_full(self):
        data = {
            "trigger": "push",
            "pr_number": 7,
            "actor": "octocat",
            "source_ip": "10.0.0.1",
        }
        rctx = RequesterContext.from_dict(data)
        assert rctx.trigger == "push"
        assert rctx.pr_number == 7
        assert rctx.actor == "octocat"
        assert rctx.source_ip == "10.0.0.1"

    def test_from_dict_defaults(self):
        rctx = RequesterContext.from_dict({})
        assert rctx.trigger == ""
        assert rctx.pr_number is None
        assert rctx.actor == ""
        assert rctx.source_ip == ""


class TestTransactionContext:
    def test_from_dict_full(self):
        data = {
            "run_id": "run-1",
            "org_id": "org-1",
            "repo_url": "https://github.com/acme/repo",
            "agent_type": "scanner",
            "agent_mode": "deep",
            "allowed_models": {
                "triage": ["openai:gpt-4o-mini"],
                "analysis": ["openai:gpt-4o"],
            },
            "allowed_providers": ["openai"],
            "allowed_tools": {"mcp-server": ["tool1", "tool2"]},
        }
        tctx = TransactionContext.from_dict(data)
        assert tctx.run_id == "run-1"
        assert tctx.org_id == "org-1"
        assert tctx.agent_type == "scanner"
        assert tctx.agent_mode == "deep"
        assert tctx.allowed_models.triage == ["openai:gpt-4o-mini"]
        assert tctx.allowed_providers == ["openai"]
        assert tctx.allowed_tools == {"mcp-server": ["tool1", "tool2"]}

    def test_from_dict_defaults(self):
        tctx = TransactionContext.from_dict({})
        assert tctx.run_id == ""
        assert tctx.allowed_models.triage == []
        assert tctx.allowed_providers == []
        assert tctx.allowed_tools == {}
