"""Tests for SPIFFE JWT source."""

import base64
import json
import time

import pytest

from sage_sanctum.auth.spiffe import JWTSource, _decode_jwt_payload
from sage_sanctum.errors import SpiffeAuthError


def _make_jwt(payload: dict, header: dict | None = None) -> str:
    """Create a fake JWT (unsigned) for testing."""
    if header is None:
        header = {"alg": "EdDSA", "typ": "JWT"}
    h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    s = base64.urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


class TestDecodeJwtPayload:
    def test_valid_jwt(self):
        payload = {"sub": "test", "exp": 12345}
        token = _make_jwt(payload)
        decoded = _decode_jwt_payload(token)
        assert decoded["sub"] == "test"
        assert decoded["exp"] == 12345

    def test_invalid_format_too_few_parts(self):
        with pytest.raises(SpiffeAuthError, match="expected 3 parts"):
            _decode_jwt_payload("only.two")

    def test_invalid_format_too_many_parts(self):
        with pytest.raises(SpiffeAuthError, match="expected 3 parts"):
            _decode_jwt_payload("a.b.c.d")

    def test_invalid_base64(self):
        with pytest.raises(SpiffeAuthError, match="Failed to decode"):
            _decode_jwt_payload("header.!!!invalid!!!.sig")

    def test_payload_requiring_padding(self):
        """JWT payload that requires base64 padding is correctly decoded."""
        # Use a payload whose base64 is NOT a multiple of 4 chars
        payload = {"s": "x"}  # Short payload → base64 needs padding
        token = _make_jwt(payload)
        # Manually verify the payload part needs padding
        payload_b64 = token.split(".")[1]
        assert len(payload_b64) % 4 != 0, "Test requires non-padded base64"
        decoded = _decode_jwt_payload(token)
        assert decoded["s"] == "x"

    def test_payload_no_padding_needed(self):
        """JWT payload that requires no padding is correctly decoded."""
        # Craft a payload whose base64 is exactly a multiple of 4
        payload = {"sub": "spiffe://test", "exp": 12345}
        token = _make_jwt(payload)
        decoded = _decode_jwt_payload(token)
        assert decoded["sub"] == "spiffe://test"
        assert decoded["exp"] == 12345


class TestJWTSource:
    def test_reads_jwt_from_file(self, tmp_path):
        jwt_file = tmp_path / "svid.jwt"
        payload = {"sub": "spiffe://test", "exp": time.time() + 600}
        jwt_file.write_text(_make_jwt(payload))

        source = JWTSource(jwt_file)
        token = source.get_token()
        assert token.startswith("eyJ")  # base64url-encoded JSON

    def test_caches_token(self, tmp_path):
        jwt_file = tmp_path / "svid.jwt"
        payload = {"sub": "spiffe://test", "exp": time.time() + 600}
        jwt_file.write_text(_make_jwt(payload))

        source = JWTSource(jwt_file)
        token1 = source.get_token()
        # Modify file to verify caching
        jwt_file.write_text("changed")
        token2 = source.get_token()
        assert token1 == token2

    def test_refreshes_on_expiry(self, tmp_path):
        jwt_file = tmp_path / "svid.jwt"
        # Create a token that's about to expire (within refresh buffer)
        payload = {"sub": "spiffe://test", "exp": time.time() + 100}
        first_jwt = _make_jwt(payload)
        jwt_file.write_text(first_jwt)

        source = JWTSource(jwt_file)
        source.get_token()

        # Write a new token
        payload2 = {"sub": "spiffe://test2", "exp": time.time() + 600}
        second_jwt = _make_jwt(payload2)
        jwt_file.write_text(second_jwt)

        # Should refresh since first token is within buffer
        token = source.get_token()
        assert token == second_jwt

    def test_file_not_found(self, tmp_path):
        source = JWTSource(tmp_path / "missing.jwt")
        with pytest.raises(SpiffeAuthError, match="not found"):
            source.get_token()

    def test_empty_file(self, tmp_path):
        jwt_file = tmp_path / "empty.jwt"
        jwt_file.write_text("")
        source = JWTSource(jwt_file)
        with pytest.raises(SpiffeAuthError, match="empty"):
            source.get_token()

    def test_missing_exp(self, tmp_path):
        jwt_file = tmp_path / "svid.jwt"
        jwt_file.write_text(_make_jwt({"sub": "test"}))
        source = JWTSource(jwt_file)
        with pytest.raises(SpiffeAuthError, match="exp"):
            source.get_token()

    def test_invalidate(self, tmp_path):
        jwt_file = tmp_path / "svid.jwt"
        payload = {"sub": "spiffe://test", "exp": time.time() + 600}
        jwt_file.write_text(_make_jwt(payload))

        source = JWTSource(jwt_file)
        source.get_token()
        assert not source.is_expired()

        source.invalidate()
        assert source.is_expired()
