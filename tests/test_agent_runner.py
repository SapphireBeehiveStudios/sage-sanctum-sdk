"""Tests for AgentRunner lifecycle and exit codes."""

from unittest.mock import MagicMock

import pytest

from sage_sanctum.agent import AgentResult, AgentRunner, SageSanctumAgent
from sage_sanctum.errors import SpiffeAuthError
from sage_sanctum.io.inputs import AgentInput
from sage_sanctum.io.outputs import SarifOutput


class SimpleAgent(SageSanctumAgent):
    """Simple test agent."""

    @property
    def name(self) -> str:
        return "test-agent"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: AgentInput) -> AgentResult:
        return AgentResult(
            output=SarifOutput(
                tool_name="test-agent",
                tool_version="0.1.0",
            ),
            exit_code=0,
        )


class FailingAgent(SageSanctumAgent):
    """Agent that raises an error."""

    @property
    def name(self) -> str:
        return "failing-agent"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: AgentInput) -> AgentResult:
        raise SpiffeAuthError("JWT expired")


class TestAgentResult:
    def test_default_values(self):
        result = AgentResult()
        assert result.output is None
        assert result.exit_code == 0
        assert result.error == ""
        assert result.duration_seconds == 0.0
        assert result.metadata == {}


class TestSageSanctumAgent:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            SageSanctumAgent(MagicMock())


class TestAgentRunner:
    def test_maps_sdk_error_to_exit_code(self, monkeypatch, tmp_path):
        """Test that SageSanctumErrors map to their exit codes."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("ORG_ID", "test-org")
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("REPO_PATH", str(repo_dir))

        runner = AgentRunner(FailingAgent)
        exit_code = runner.run()
        assert exit_code == 11  # SpiffeAuthError exit code

    def test_successful_run(self, monkeypatch, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        output_dir = tmp_path / "output"

        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("ORG_ID", "test-org")
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("REPO_PATH", str(repo_dir))
        monkeypatch.setenv("OUTPUT_PATH", str(output_dir))

        runner = AgentRunner(SimpleAgent)
        exit_code = runner.run()
        assert exit_code == 0

    def test_configuration_error(self, monkeypatch):
        monkeypatch.delenv("RUN_ID", raising=False)
        monkeypatch.delenv("ORG_ID", raising=False)

        runner = AgentRunner(SimpleAgent)
        exit_code = runner.run()
        assert exit_code == 43  # ConfigurationError
