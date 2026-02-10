"""Pytest fixtures for testing Sage Sanctum agents."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ..auth.trat import AllowedModels, TransactionToken
from ..context import AgentContext
from ..llm.model_selector import ModelSelector, StaticModelSelector
from .mocks import MockGatewayClient, MockTratClient


@pytest.fixture
def mock_gateway() -> MockGatewayClient:
    """Mock gateway client for testing."""
    return MockGatewayClient(is_gateway=False)


@pytest.fixture
def sample_trat() -> TransactionToken:
    """Sample Transaction Token with standard preset models."""
    client = MockTratClient()
    return client.get_token()


@pytest.fixture
def mock_context(tmp_path: Path, mock_gateway: MockGatewayClient) -> AgentContext:
    """AgentContext with mocked gateway and static model selector."""
    return AgentContext(
        run_id="test-run-123",
        org_id="test-org",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        gateway_client=mock_gateway,
        model_selector=StaticModelSelector("gpt-4o"),
        logger=logging.getLogger("test"),
    )
