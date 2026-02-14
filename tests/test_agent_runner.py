"""Tests for AgentRunner lifecycle and exit codes."""

import asyncio
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


class SlowAgent(SageSanctumAgent):
    """Agent that blocks until cancelled."""

    @property
    def name(self) -> str:
        return "slow-agent"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: AgentInput) -> AgentResult:
        await asyncio.sleep(3600)  # Block for a long time
        return AgentResult(exit_code=0)


class TestShutdownEvent:
    def test_shutdown_cancels_agent(self, monkeypatch, tmp_path):
        """Verify that setting the shutdown event cancels the running agent and returns 130."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("ORG_ID", "test-org")
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("REPO_PATH", str(repo_dir))

        runner = AgentRunner(SlowAgent)

        async def run_with_signal():
            """Run the agent and fire the shutdown event after a short delay."""
            task = asyncio.create_task(runner._run_async())
            # Wait a moment for the agent to start, then signal shutdown
            await asyncio.sleep(0.1)
            import signal
            runner._handle_signal(signal.SIGTERM)
            return await task

        exit_code = asyncio.run(run_with_signal())
        assert exit_code == 130
