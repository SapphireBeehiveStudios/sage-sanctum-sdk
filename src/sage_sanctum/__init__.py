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
from .errors import (
    ExternalToolError,
    OutputParseError,
    SageSanctumError,
    SubprocessError,
    SubprocessTimeoutError,
)
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
    "ExternalToolError",
    "Finding",
    "Location",
    "ModelCategory",
    "ModelRef",
    "OutputParseError",
    "RepositoryInput",
    "SageSanctumAgent",
    "SageSanctumError",
    "SarifOutput",
    "SubprocessError",
    "SubprocessTimeoutError",
    "TokenUsage",
    "configure_logging",
    "get_logger",
]
