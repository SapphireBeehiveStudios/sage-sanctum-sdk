"""Sage Sanctum Agent SDK.

SDK for building agents that run within the Sage Sanctum infrastructure.
Handles SPIFFE authentication, Transaction Tokens, LLM gateway access,
and standardized input/output formats.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sage-sanctum-sdk")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .agent import AgentResult, AgentRunner, SageSanctumAgent
from .context import AgentContext
from .errors import SageSanctumError
from .io.inputs import AgentInput, RepositoryInput
from .io.outputs import AgentOutput, Finding, Location, SarifOutput, TokenUsage
from .llm.model_category import ModelCategory
from .llm.model_ref import ModelRef
from .logging import configure_logging, get_logger

__all__ = [
    "AgentContext",
    "AgentInput",
    "AgentOutput",
    "AgentResult",
    "AgentRunner",
    "Finding",
    "Location",
    "ModelCategory",
    "ModelRef",
    "RepositoryInput",
    "SageSanctumAgent",
    "SageSanctumError",
    "SarifOutput",
    "TokenUsage",
    "configure_logging",
    "get_logger",
]
