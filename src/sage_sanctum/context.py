"""AgentContext: central context object for agent execution."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from .auth.spiffe import JWTSource
from .auth.trat import TransactionTokenClient
from .errors import ConfigurationError
from .gateway.client import DirectProviderClient, GatewayClient, SpiffeGatewayClient
from .io.inputs import RepositoryInput
from .io.outputs import AgentOutput
from .llm.gateway_chat import create_llm_for_gateway
from .llm.model_category import ModelCategory
from .llm.model_selector import ModelSelector, StaticModelSelector

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Central context for agent execution.

    Provides access to run metadata, LLM clients, model selection, and I/O.
    Created automatically by ``AgentRunner`` or manually via factory methods.

    Attributes:
        run_id: Unique identifier for this agent run.
        org_id: Organization identifier.
        work_dir: Working directory for temporary files.
        output_dir: Directory where agent output is written.
        gateway_client: Client for LLM gateway access (SPIFFE or direct).
        model_selector: Resolves ``ModelCategory`` to concrete ``ModelRef``.
        logger: Logger instance for the agent.

    Example:
        ```python
        # Production (via AgentRunner — automatic):
        context = AgentContext.from_environment()

        # Local development:
        context = AgentContext.for_local_development(
            work_dir="/tmp/work",
            output_dir="/tmp/output",
            model="openai:gpt-4o",
        )
        ```
    """

    run_id: str
    org_id: str
    work_dir: Path
    output_dir: Path
    gateway_client: GatewayClient
    model_selector: ModelSelector
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("sage_sanctum"))

    def create_llm_client(self, category: ModelCategory) -> BaseChatModel:
        """Create an LLM client for the specified model category.

        Resolves the model from the selector, then creates an appropriate
        LLM client (``GatewayChatModel`` in gateway mode, ``ChatLiteLLM``
        in direct mode).

        Args:
            category: The model category (e.g. ``ModelCategory.ANALYSIS``).

        Returns:
            A LangChain ``BaseChatModel`` ready for ``invoke()`` calls.

        Raises:
            ModelNotAvailableError: If no model is configured for the category.
        """
        model_ref = self.model_selector.select(category)
        logger.debug("Creating LLM client for %s: %s", category.value, model_ref)
        return create_llm_for_gateway(
            model_ref=model_ref,
            gateway_client=self.gateway_client,
        )

    def load_input(self) -> RepositoryInput:
        """Load agent input from environment.

        Returns:
            RepositoryInput from REPO_PATH env var.
        """
        repo_input = RepositoryInput.from_environment()
        repo_input.validate()
        return repo_input

    def write_output(self, output: AgentOutput) -> list[str]:
        """Write agent output to the output directory.

        Returns:
            List of filenames written.
        """
        return output.write(self.output_dir)

    @classmethod
    def from_environment(cls) -> AgentContext:
        """Create context from environment variables (synchronous).

        Expected env vars:
        - RUN_ID: Run identifier (required)
        - ORG_ID: Organization identifier (required)
        - WORK_DIR: Working directory (default: /work)
        - OUTPUT_PATH: Output directory (default: /output)
        - SPIFFE_JWT_PATH: Path to SPIFFE JWT file
        - TRAT_FILE: Path to TraT file
        - AUTH_SIDECAR_SOCKET: Path to auth sidecar Unix socket
        - LLM_GATEWAY_SOCKET: Path to LLM gateway Unix socket
        - SAGE_SANCTUM_ALLOW_DIRECT: Set to '1' for direct provider access
        """
        run_id = os.environ.get("RUN_ID", "")
        org_id = os.environ.get("ORG_ID", "")

        if not run_id:
            raise ConfigurationError("RUN_ID environment variable not set")
        if not org_id:
            raise ConfigurationError("ORG_ID environment variable not set")

        work_dir = Path(os.environ.get("WORK_DIR", "/work"))
        output_dir = Path(os.environ.get("OUTPUT_PATH", "/output"))

        # Build gateway client
        gateway_client = cls._create_gateway_client()

        # Build model selector from TraT or environment
        model_selector = cls._create_model_selector(gateway_client)

        return cls(
            run_id=run_id,
            org_id=org_id,
            work_dir=work_dir,
            output_dir=output_dir,
            gateway_client=gateway_client,
            model_selector=model_selector,
        )

    @classmethod
    async def from_environment_async(cls) -> AgentContext:
        """Async version of from_environment. Currently identical."""
        return cls.from_environment()

    @classmethod
    def for_local_development(
        cls,
        work_dir: str | Path = ".",
        output_dir: str | Path = "./output",
        model: str = "gpt-4o",
    ) -> AgentContext:
        """Create context for local development with direct API keys.

        Bypasses the gateway and calls LLM providers directly using API keys
        from environment variables. Sets ``SAGE_SANCTUM_ALLOW_DIRECT=1``
        automatically.

        Args:
            work_dir: Working directory for temporary files.
            output_dir: Directory for agent output.
            model: Model reference string (e.g. ``"openai:gpt-4o"``). Used
                for all categories via ``StaticModelSelector``.

        Returns:
            A fully configured ``AgentContext`` in direct mode.
        """
        os.environ.setdefault("SAGE_SANCTUM_ALLOW_DIRECT", "1")

        return cls(
            run_id="local",
            org_id="local",
            work_dir=Path(work_dir),
            output_dir=Path(output_dir),
            gateway_client=DirectProviderClient(),
            model_selector=StaticModelSelector(model),
        )

    @classmethod
    def _create_gateway_client(cls) -> GatewayClient:
        """Create the appropriate gateway client based on environment."""
        # Check for direct mode first
        if os.environ.get("SAGE_SANCTUM_ALLOW_DIRECT"):
            return DirectProviderClient()

        # Production mode: SPIFFE + TraT
        jwt_path = os.environ.get("SPIFFE_JWT_PATH")
        if not jwt_path:
            raise ConfigurationError(
                "SPIFFE_JWT_PATH not set. Set SAGE_SANCTUM_ALLOW_DIRECT=1 for local dev."
            )

        jwt_source = JWTSource(jwt_path)

        trat_file = os.environ.get("TRAT_FILE")
        sidecar_socket = os.environ.get("AUTH_SIDECAR_SOCKET")
        trat_client = TransactionTokenClient(
            trat_file=trat_file,
            sidecar_socket=sidecar_socket,
        )

        gateway_socket = os.environ.get("LLM_GATEWAY_SOCKET")

        return SpiffeGatewayClient(
            jwt_source=jwt_source,
            trat_client=trat_client,
            gateway_socket=gateway_socket,
        )

    @classmethod
    def _create_model_selector(cls, gateway_client: GatewayClient) -> ModelSelector:
        """Create model selector from TraT or fallback to env var."""
        trat = gateway_client.get_trat()
        if trat and trat.tctx.allowed_models.triage:
            # Use TraT's allowed_models
            return ModelSelector(trat.tctx.allowed_models.to_dict())

        # Fallback to static selector from env
        model = (
            os.environ.get("SAGE_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o"
        )
        return StaticModelSelector(model)
