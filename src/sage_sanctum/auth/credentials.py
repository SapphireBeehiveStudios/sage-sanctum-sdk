"""Gateway credentials: combined SPIFFE JWT + TraT for gateway authentication."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayCredentials:
    """Combined credentials for authenticating to LLM/MCP gateways.

    The gateway validates both tokens on every request:

    - **SPIFFE JWT** proves the agent's identity.
    - **TraT** authorizes the specific transaction.

    Attributes:
        spiffe_jwt: SPIFFE JWT SVID for agent identity (``Authorization`` header).
        trat: Transaction Token JWT for authorization (``Txn-Token`` header).
    """

    spiffe_jwt: str
    trat: str

    def __repr__(self) -> str:
        """Return masked representation to prevent credential leakage in logs."""
        return (
            f"GatewayCredentials("
            f"spiffe_jwt='{self.spiffe_jwt[:8]}...' "
            f"trat='{self.trat[:8]}...')"
        )

    def __str__(self) -> str:
        """Return masked string representation."""
        return self.__repr__()

    def auth_headers(self) -> dict[str, str]:
        """Return HTTP headers for gateway authentication.

        Returns:
            Dictionary with ``Authorization`` (Bearer) and ``Txn-Token`` headers.
        """
        return {
            "Authorization": f"Bearer {self.spiffe_jwt}",
            "Txn-Token": self.trat,
        }
