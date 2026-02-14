"""Tests for MCP gateway client stub."""

import pytest

from sage_sanctum.mcp.client import MCPGatewayClient


class TestMCPGatewayClient:
    def test_stores_socket_path(self):
        client = MCPGatewayClient(socket_path="/var/run/mcp.sock")
        assert client._socket_path == "/var/run/mcp.sock"

    def test_default_socket_path_is_none(self):
        client = MCPGatewayClient()
        assert client._socket_path is None

    @pytest.mark.asyncio
    async def test_invoke_tool_raises_not_implemented(self):
        client = MCPGatewayClient()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await client.invoke_tool("file_tools", "file_read")

    @pytest.mark.asyncio
    async def test_invoke_tool_includes_server_and_tool_in_message(self):
        client = MCPGatewayClient()
        with pytest.raises(NotImplementedError, match="file_tools/file_read"):
            await client.invoke_tool("file_tools", "file_read", {"path": "/tmp"})
