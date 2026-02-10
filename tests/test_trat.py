"""Tests for Transaction Token client and parsing."""

import base64
import json
import time

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
