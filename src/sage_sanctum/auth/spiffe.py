"""SPIFFE JWT Source: read, cache, and refresh SPIFFE JWTs from file."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

from ..errors import SpiffeAuthError

logger = logging.getLogger(__name__)

# Refresh JWT 5 minutes before expiry
_REFRESH_BUFFER_SECONDS = 300


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without signature verification.

    Only used locally to check expiry — the gateway performs full verification.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise SpiffeAuthError(f"Invalid JWT format: expected 3 parts, got {len(parts)}")

    # Base64url decode the payload (part 1)
    payload_b64 = parts[1]
    # Add padding if needed
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding

    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes)
    except Exception as e:
        raise SpiffeAuthError(f"Failed to decode JWT payload: {e}") from e


class JWTSource:
    """Reads and caches SPIFFE JWTs from a file path.

    The SPIRE agent writes the JWT SVID to a well-known file path. This class
    reads it, caches it in memory, and automatically refreshes 5 minutes before
    expiry (``_REFRESH_BUFFER_SECONDS``).

    Args:
        jwt_path: Filesystem path to the SPIFFE JWT SVID file.

    Attributes:
        path: Resolved ``Path`` to the JWT file.
    """

    def __init__(self, jwt_path: str | Path) -> None:
        self._path = Path(jwt_path)
        self._cached_token: str | None = None
        self._cached_expiry: float = 0.0

    @property
    def path(self) -> Path:
        """Filesystem path to the JWT file."""
        return self._path

    def get_token(self) -> str:
        """Get a valid SPIFFE JWT, refreshing from file if needed.

        Returns the cached token if it is still valid (with a 5-minute buffer
        before expiry). Otherwise, reads a fresh token from the file.

        Returns:
            Raw JWT string suitable for the ``Authorization: Bearer`` header.

        Raises:
            SpiffeAuthError: If the JWT file cannot be read or is invalid.
        """
        now = time.time()

        # Return cached token if still valid (with buffer)
        if self._cached_token and now < (self._cached_expiry - _REFRESH_BUFFER_SECONDS):
            return self._cached_token

        # Read fresh token from file
        return self._refresh()

    def _refresh(self) -> str:
        """Read JWT from file and update cache."""
        if not self._path.exists():
            raise SpiffeAuthError(f"SPIFFE JWT file not found: {self._path}")

        try:
            token = self._path.read_text().strip()
        except OSError as e:
            raise SpiffeAuthError(f"Failed to read SPIFFE JWT from {self._path}: {e}") from e

        if not token:
            raise SpiffeAuthError(f"SPIFFE JWT file is empty: {self._path}")

        # Decode payload to get expiry (no signature verification)
        payload = _decode_jwt_payload(token)
        exp = payload.get("exp")
        if exp is None:
            raise SpiffeAuthError("SPIFFE JWT missing 'exp' claim")

        self._cached_token = token
        self._cached_expiry = float(exp)

        logger.debug(
            "Refreshed SPIFFE JWT from %s (expires at %s)",
            self._path,
            self._cached_expiry,
        )
        return token

    def is_expired(self) -> bool:
        """Check if the cached token is expired or about to expire.

        Returns:
            ``True`` if no token is cached, or it will expire within
            ``_REFRESH_BUFFER_SECONDS`` (300 s).
        """
        if not self._cached_token:
            return True
        return time.time() >= (self._cached_expiry - _REFRESH_BUFFER_SECONDS)

    def invalidate(self) -> None:
        """Clear the cached token, forcing a refresh on next access."""
        self._cached_token = None
        self._cached_expiry = 0.0
