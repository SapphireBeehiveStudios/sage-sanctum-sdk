"""SageSanctumAgent base class and AgentRunner."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .context import AgentContext
from .errors import SageSanctumError
from .io.inputs import AgentInput
from .io.outputs import AgentOutput

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of an agent run."""

    output: AgentOutput | None = None
    exit_code: int = 0
    error: str = ""
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SageSanctumAgent(ABC):
    """Base class for Sage Sanctum agents.

    Subclasses implement the async `run()` method with their business logic.
    The SDK handles authentication, context initialization, and output writing.
    """

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name (e.g., 'sage-security-scanner')."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Agent version."""

    @abstractmethod
    async def run(self, agent_input: AgentInput) -> AgentResult:
        """Execute the agent's main logic.

        Args:
            agent_input: The validated input for this run.

        Returns:
            AgentResult with output and metadata.
        """


class AgentRunner:
    """Runs a SageSanctumAgent with proper lifecycle management.

    Handles:
    - Event loop creation
    - Signal handling (SIGTERM, SIGINT)
    - Context initialization from environment
    - Error → exit code mapping
    - Output writing
    """

    def __init__(self, agent_class: type[SageSanctumAgent]) -> None:
        self._agent_class = agent_class
        self._shutdown_event: asyncio.Event | None = None

    def run(self) -> int:
        """Run the agent synchronously. Returns exit code."""
        try:
            return asyncio.run(self._run_async())
        except KeyboardInterrupt:
            logger.info("Agent interrupted by user")
            return 130  # Standard SIGINT exit code
        except SageSanctumError as e:
            logger.error("Agent error: %s (exit code %d)", e, e.exit_code)
            return e.exit_code
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)
            return 1

    async def _run_async(self) -> int:
        """Async agent lifecycle."""
        self._shutdown_event = asyncio.Event()

        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal, sig)

        start_time = time.monotonic()

        # Initialize context
        logger.info("Initializing agent context...")
        context = await AgentContext.from_environment_async()

        # Create agent
        agent = self._agent_class(context)
        logger.info(
            "Starting agent %s v%s (run_id=%s)",
            agent.name,
            agent.version,
            context.run_id,
        )

        # Load input
        agent_input = context.load_input()

        # Run agent
        result = await agent.run(agent_input)

        duration = time.monotonic() - start_time
        result.duration_seconds = duration

        # Write output
        if result.output:
            files = context.write_output(result.output)
            logger.info("Output written: %s", files)

        logger.info(
            "Agent %s completed in %.1fs (exit_code=%d)",
            agent.name,
            duration,
            result.exit_code,
        )

        return result.exit_code

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %s, shutting down...", sig.name)
        if self._shutdown_event:
            self._shutdown_event.set()
