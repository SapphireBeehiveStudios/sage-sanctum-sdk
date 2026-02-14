"""MCP Gateway Client stub — NOT YET AVAILABLE.

This module is a placeholder for future MCP tool invocation via the
MCP gateway socket. It is intentionally excluded from the public API
(not exported from ``sage_sanctum`` or ``sage_sanctum.mcp``).

Do not depend on this interface; it will change when the MCP gateway
is implemented.
"""

from __future__ import annotations

from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)


class MCPGatewayClient:
    """Client for invoking MCP tools via the MCP gateway.

    This is a placeholder implementation. The MCP gateway routes tool
    invocations to appropriate MCP servers with Topaz authorization.
    """

    def __init__(self, socket_path: str | None = None) -> None:
        self._socket_path = socket_path

    async def invoke_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke an MCP tool via the gateway.

        Args:
            server: MCP server name (e.g., 'file_tools', 'git_tools')
            tool: Tool name (e.g., 'file_read', 'git_log')
            arguments: Tool arguments

        Returns:
            Tool result as a dictionary.

        Raises:
            NotImplementedError: MCP gateway integration pending.
        """
        raise NotImplementedError(
            "MCP gateway integration is not yet implemented. "
            f"Attempted to invoke {server}/{tool}"
        )
