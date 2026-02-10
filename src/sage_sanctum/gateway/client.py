"""Gateway client implementations for LLM access.

Two modes:
- SpiffeGatewayClient: Production client using SPIFFE + TraT via Unix sockets
- DirectProviderClient: Local dev client using direct API keys
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from ..auth.credentials import GatewayCredentials
from ..auth.spiffe import JWTSource
from ..auth.trat import TransactionToken, TransactionTokenClient
from ..errors import ConfigurationError, GatewayError

logger = logging.getLogger(__name__)


class GatewayClient(ABC):
    """Abstract gateway client for accessing LLM providers."""

    @abstractmethod
    def get_credentials(self) -> GatewayCredentials:
        """Get current gateway credentials.

        Raises:
            AuthError: If credentials cannot be acquired.
        """

    @abstractmethod
    def get_endpoint(self, provider: str) -> str:
        """Get the gateway endpoint URL for a specific provider.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'google')

        Returns:
            Base URL for the provider endpoint.
        """

    @property
    @abstractmethod
    def is_gateway_mode(self) -> bool:
        """Whether this client routes through the gateway."""

    def get_trat(self) -> TransactionToken | None:
        """Get the current Transaction Token, if available."""
        return None


class SpiffeGatewayClient(GatewayClient):
    """Production gateway client using SPIFFE identity and Transaction Tokens.

    Connects to the LLM gateway via Unix socket with SPIFFE JWT + TraT auth.
    """

    def __init__(
        self,
        jwt_source: JWTSource,
        trat_client: TransactionTokenClient,
        gateway_socket: str | Path | None = None,
    ) -> None:
        self._jwt_source = jwt_source
        self._trat_client = trat_client
        self._gateway_socket = Path(gateway_socket) if gateway_socket else None

    def get_credentials(self) -> GatewayCredentials:
        jwt = self._jwt_source.get_token()
        trat = self._trat_client.get_token()
        return GatewayCredentials(spiffe_jwt=jwt, trat=trat.raw)

    def get_endpoint(self, provider: str) -> str:
        """Return the Unix socket-based endpoint.

        In gateway mode, all providers are accessed through the same socket.
        The provider is specified via X-Provider header.
        """
        if self._gateway_socket:
            # For Unix socket, we return a special URL that the HTTP client understands
            return f"unix://{self._gateway_socket}"
        # Fallback to env-var based endpoints
        endpoints = {
            "openai": os.environ.get("OPENAI_BASE_URL", "http://gateway:8080/v1"),
            "anthropic": os.environ.get("ANTHROPIC_BASE_URL", "http://gateway:8080/anthropic"),
            "google": os.environ.get("GOOGLE_BASE_URL", "http://gateway:8080/google"),
        }
        return endpoints.get(provider, endpoints["openai"])

    @property
    def is_gateway_mode(self) -> bool:
        return True

    def get_trat(self) -> TransactionToken | None:
        try:
            return self._trat_client.get_token()
        except Exception:
            return None


class DirectProviderClient(GatewayClient):
    """Local development client using direct API keys.

    Requires SAGE_SANCTUM_ALLOW_DIRECT=1 for safety.
    Reads API keys from environment variables.
    """

    def __init__(self) -> None:
        if not os.environ.get("SAGE_SANCTUM_ALLOW_DIRECT"):
            raise ConfigurationError(
                "DirectProviderClient requires SAGE_SANCTUM_ALLOW_DIRECT=1. "
                "This bypasses gateway security and should only be used for local development."
            )

    def get_credentials(self) -> GatewayCredentials:
        # In direct mode, we use API keys directly — no SPIFFE/TraT
        return GatewayCredentials(spiffe_jwt="", trat="")

    def get_endpoint(self, provider: str) -> str:
        """Return the provider's native API endpoint."""
        endpoints = {
            "openai": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "anthropic": os.environ.get(
                "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
            ),
            "google": os.environ.get(
                "GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com"
            ),
        }
        return endpoints.get(provider, endpoints["openai"])

    @property
    def is_gateway_mode(self) -> bool:
        return False

    def get_api_key(self, provider: str) -> str:
        """Get the API key for a provider from environment.

        Raises:
            ConfigurationError: If the API key is not set.
        """
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        var_name = env_vars.get(provider, "OPENAI_API_KEY")
        key = os.environ.get(var_name)
        if not key:
            raise ConfigurationError(
                f"API key not set for {provider}. Set {var_name} environment variable."
            )
        return key
