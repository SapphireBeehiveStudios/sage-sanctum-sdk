"""Transaction Token (TraT) client and data models.

Reads TraTs from file (TRAT_FILE) or auth sidecar socket (AUTH_SIDECAR_SOCKET).
TraTs are IETF-standard JWTs (draft-ietf-oauth-transaction-tokens) that authorize
a specific transaction within the Sage Sanctum platform.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import TraTAcquisitionError, TraTExpiredError
from .spiffe import _decode_jwt_payload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequesterContext:
    """Audit metadata from the TraT rctx claim."""

    trigger: str = ""
    pr_number: int | None = None
    actor: str = ""
    source_ip: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> RequesterContext:
        return cls(
            trigger=data.get("trigger", ""),
            pr_number=data.get("pr_number"),
            actor=data.get("actor", ""),
            source_ip=data.get("source_ip", ""),
        )


@dataclass(frozen=True)
class AllowedModels:
    """Model allowlists per category from tctx.allowed_models."""

    triage: list[str] = field(default_factory=list)
    analysis: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    embeddings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> AllowedModels:
        return cls(
            triage=data.get("triage", []),
            analysis=data.get("analysis", []),
            reasoning=data.get("reasoning", []),
            embeddings=data.get("embeddings", []),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "triage": self.triage,
            "analysis": self.analysis,
            "reasoning": self.reasoning,
            "embeddings": self.embeddings,
        }


@dataclass(frozen=True)
class TransactionContext:
    """Immutable run parameters from the TraT tctx claim."""

    run_id: str = ""
    org_id: str = ""
    repo_url: str = ""
    agent_type: str = ""
    agent_mode: str = ""
    allowed_models: AllowedModels = field(default_factory=AllowedModels)
    allowed_providers: list[str] = field(default_factory=list)
    allowed_tools: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> TransactionContext:
        allowed_models_data = data.get("allowed_models", {})
        return cls(
            run_id=data.get("run_id", ""),
            org_id=data.get("org_id", ""),
            repo_url=data.get("repo_url", ""),
            agent_type=data.get("agent_type", ""),
            agent_mode=data.get("agent_mode", ""),
            allowed_models=AllowedModels.from_dict(allowed_models_data),
            allowed_providers=data.get("allowed_providers", []),
            allowed_tools=data.get("allowed_tools", {}),
        )


@dataclass(frozen=True)
class TransactionToken:
    """Parsed Transaction Token with all claims."""

    raw: str
    txn: str
    sub: str
    scope: str
    req_wl: str
    iat: float
    exp: float
    aud: str = ""
    iss: str = ""
    tctx: TransactionContext = field(default_factory=TransactionContext)
    rctx: RequesterContext = field(default_factory=RequesterContext)

    @classmethod
    def from_jwt(cls, token: str) -> TransactionToken:
        """Parse a TraT JWT (without signature verification).

        Signature verification is done by the gateway, not the agent.

        Raises:
            TraTAcquisitionError: If the token is malformed or missing required claims.
        """
        try:
            payload = _decode_jwt_payload(token)
        except Exception as e:
            raise TraTAcquisitionError(f"Failed to parse TraT: {e}") from e

        # Required claims
        txn = payload.get("txn", "")
        if not txn:
            raise TraTAcquisitionError("TraT missing required 'txn' claim")

        return cls(
            raw=token,
            txn=txn,
            sub=payload.get("sub", ""),
            scope=payload.get("scope", ""),
            req_wl=payload.get("req_wl", ""),
            iat=float(payload.get("iat", 0)),
            exp=float(payload.get("exp", 0)),
            aud=payload.get("aud", ""),
            iss=payload.get("iss", ""),
            tctx=TransactionContext.from_dict(payload.get("tctx", {})),
            rctx=RequesterContext.from_dict(payload.get("rctx", {})),
        )

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.exp

    def check_not_expired(self) -> None:
        """Raise if token is expired.

        Raises:
            TraTExpiredError: If the token has expired.
        """
        if self.is_expired:
            raise TraTExpiredError(
                f"Transaction Token expired at {self.exp} "
                f"(current time: {time.time():.0f})"
            )


class TransactionTokenClient:
    """Reads TraTs from file or auth sidecar socket.

    In production, the auth sidecar acquires and refreshes TraTs. The agent
    reads them from a well-known file path or queries the sidecar socket.
    """

    def __init__(
        self,
        trat_file: str | Path | None = None,
        sidecar_socket: str | Path | None = None,
    ) -> None:
        self._trat_file = Path(trat_file) if trat_file else None
        self._sidecar_socket = Path(sidecar_socket) if sidecar_socket else None
        self._cached: TransactionToken | None = None

    def get_token(self) -> TransactionToken:
        """Get the current Transaction Token.

        Tries file first, then sidecar socket.

        Raises:
            TraTAcquisitionError: If the TraT cannot be acquired.
            TraTExpiredError: If the acquired TraT is expired.
        """
        # Return cached if still valid
        if self._cached and not self._cached.is_expired:
            return self._cached

        token = self._acquire()
        token.check_not_expired()
        self._cached = token
        return token

    def _acquire(self) -> TransactionToken:
        """Acquire TraT from file or sidecar."""
        # Try file first
        if self._trat_file:
            return self._read_from_file()

        # Try sidecar socket
        if self._sidecar_socket:
            return self._read_from_sidecar()

        raise TraTAcquisitionError(
            "No TraT source configured. Set TRAT_FILE or AUTH_SIDECAR_SOCKET."
        )

    def _read_from_file(self) -> TransactionToken:
        """Read TraT JWT from file."""
        if not self._trat_file or not self._trat_file.exists():
            raise TraTAcquisitionError(
                f"TraT file not found: {self._trat_file}"
            )

        try:
            raw = self._trat_file.read_text().strip()
        except OSError as e:
            raise TraTAcquisitionError(
                f"Failed to read TraT from {self._trat_file}: {e}"
            ) from e

        if not raw:
            raise TraTAcquisitionError(f"TraT file is empty: {self._trat_file}")

        return TransactionToken.from_jwt(raw)

    def _read_from_sidecar(self) -> TransactionToken:
        """Read TraT from auth sidecar socket.

        The auth sidecar exposes a simple HTTP API over a Unix socket:
        GET /trat -> JWT string
        """
        if not self._sidecar_socket or not self._sidecar_socket.exists():
            raise TraTAcquisitionError(
                f"Auth sidecar socket not found: {self._sidecar_socket}"
            )

        import urllib3

        # Connect via Unix socket
        http = urllib3.HTTPConnectionPool(
            host="localhost",
            port=0,  # Unused for UDS
        )
        # urllib3 doesn't natively support Unix sockets for plain pools.
        # Use our gateway HTTP client for this.
        from ..gateway.http import GatewayHttpClient

        client = GatewayHttpClient(socket_path=self._sidecar_socket)
        try:
            response = client.request("GET", "/trat")
            if response.status != 200:
                raise TraTAcquisitionError(
                    f"Auth sidecar returned status {response.status}: {response.data}"
                )
            raw = response.data.strip()
            return TransactionToken.from_jwt(raw)
        except TraTAcquisitionError:
            raise
        except Exception as e:
            raise TraTAcquisitionError(
                f"Failed to read TraT from sidecar: {e}"
            ) from e

    def invalidate(self) -> None:
        """Clear cached token."""
        self._cached = None
