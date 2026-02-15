"""Tests for AgentRunner lifecycle and exit codes."""

import asyncio
from unittest.mock import MagicMock

import pytest

from sage_sanctum.agent import AgentResult, AgentRunner, SageSanctumAgent
from sage_sanctum.errors import SpiffeAuthError
from sage_sanctum.io.inputs import AgentInput, RepositoryInput
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


class ExternalLlmAgent(SageSanctumAgent):
    """Agent that doesn't need the gateway."""

    requires_gateway = False

    @property
    def name(self) -> str:
        return "external-agent"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: AgentInput) -> AgentResult:
        # Verify we don't have gateway access
        assert self.context.gateway_client is None
        return AgentResult(exit_code=0)


class ShutdownAwareAgent(SageSanctumAgent):
    """Agent that uses the shutdown event for cooperative cancellation."""

    requires_gateway = False

    @property
    def name(self) -> str:
        return "shutdown-aware"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: AgentInput) -> AgentResult:
        # Race subprocess work against shutdown
        shutdown_task = asyncio.create_task(self.wait_for_shutdown())
        work_task = asyncio.create_task(asyncio.sleep(3600))

        done, pending = await asyncio.wait(
            {shutdown_task, work_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if shutdown_task in done:
            return AgentResult(exit_code=130)

        return AgentResult(exit_code=0)


class TypedAgent(SageSanctumAgent[RepositoryInput]):
    """Agent with typed input."""

    @property
    def name(self) -> str:
        return "typed-agent"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: RepositoryInput) -> AgentResult:
        # Type narrowing works — can access .path directly
        assert hasattr(agent_input, "path")
        return AgentResult(exit_code=0)


class TestRequiresGateway:
    def test_external_llm_agent_skips_gateway(self, monkeypatch, tmp_path):
        """Agents with requires_gateway=False skip SPIFFE/gateway setup."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("ORG_ID", "test-org")
        monkeypatch.setenv("REPO_PATH", str(repo_dir))
        # Deliberately NOT setting SPIFFE/gateway vars

        runner = AgentRunner(ExternalLlmAgent)
        exit_code = runner.run()
        assert exit_code == 0


class TestShutdownEventPropagation:
    def test_shutdown_event_shared_with_agent(self, monkeypatch, tmp_path):
        """Verify the runner shares its shutdown event with the agent instance."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("ORG_ID", "test-org")
        monkeypatch.setenv("REPO_PATH", str(repo_dir))

        runner = AgentRunner(ShutdownAwareAgent)

        async def run_with_signal():
            task = asyncio.create_task(runner._run_async())
            await asyncio.sleep(0.1)
            import signal
            runner._handle_signal(signal.SIGTERM)
            return await task

        exit_code = asyncio.run(run_with_signal())
        # The runner returns 130 because the shutdown event fires
        assert exit_code == 130


class TestGenericInputType:
    def test_typed_agent_runs(self, monkeypatch, tmp_path):
        """Typed agent (SageSanctumAgent[RepositoryInput]) runs successfully."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("ORG_ID", "test-org")
        monkeypatch.setenv("SAGE_SANCTUM_ALLOW_DIRECT", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("REPO_PATH", str(repo_dir))

        runner = AgentRunner(TypedAgent)
        exit_code = runner.run()
        assert exit_code == 0


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
