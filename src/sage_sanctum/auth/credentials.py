"""Gateway credentials: combined SPIFFE JWT + TraT for gateway authentication."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayCredentials:
    """Combined credentials for authenticating to LLM/MCP gateways.

    The gateway validates both tokens:
    - SPIFFE JWT proves agent identity
    - TraT authorizes the specific transaction
    """

    spiffe_jwt: str
    trat: str

    def auth_headers(self) -> dict[str, str]:
        """Return HTTP headers for gateway authentication."""
        return {
            "Authorization": f"Bearer {self.spiffe_jwt}",
            "Txn-Token": self.trat,
        }
