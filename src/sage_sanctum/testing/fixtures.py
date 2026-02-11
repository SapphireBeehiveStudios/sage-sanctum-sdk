"""Pytest fixtures for testing Sage Sanctum agents."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ..auth.trat import TransactionToken
from ..context import AgentContext
from ..llm.model_selector import StaticModelSelector
from .mocks import MockGatewayClient, MockTratClient


@pytest.fixture
def mock_gateway() -> MockGatewayClient:
    """Pytest fixture providing a ``MockGatewayClient`` in direct mode.

    Returns:
        A ``MockGatewayClient`` with ``is_gateway=False`` and
        localhost endpoints.
    """
    return MockGatewayClient(is_gateway=False)


@pytest.fixture
def sample_trat() -> TransactionToken:
    """Pytest fixture providing a ``TransactionToken`` with standard models.

    Default allowlists: gpt-4o-mini (triage), gpt-4o + claude-3-5-sonnet
    (analysis), o1 (reasoning), text-embedding-3-small (embeddings).

    Returns:
        A non-expired ``TransactionToken`` with mock claims.
    """
    client = MockTratClient()
    return client.get_token()


@pytest.fixture
def mock_context(tmp_path: Path, mock_gateway: MockGatewayClient) -> AgentContext:
    """Pytest fixture providing a fully configured ``AgentContext``.

    Uses ``tmp_path`` for work/output directories, a ``MockGatewayClient``,
    and a ``StaticModelSelector`` fixed on ``gpt-4o``.

    Returns:
        An ``AgentContext`` ready for agent instantiation in tests.
    """
    return AgentContext(
        run_id="test-run-123",
        org_id="test-org",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        gateway_client=mock_gateway,
        model_selector=StaticModelSelector("gpt-4o"),
        logger=logging.getLogger("test"),
    )
