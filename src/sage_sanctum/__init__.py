"""Sage Sanctum Agent SDK.

SDK for building agents that run within the Sage Sanctum infrastructure.
Handles SPIFFE authentication, Transaction Tokens, LLM gateway access,
and standardized input/output formats.
"""

__version__ = "0.1.0"

from .agent import AgentResult, AgentRunner, SageSanctumAgent
from .context import AgentContext
from .errors import SageSanctumError
from .io.inputs import AgentInput, RepositoryInput
from .io.outputs import AgentOutput, Finding, Location, SarifOutput, TokenUsage
from .llm.model_category import ModelCategory
from .llm.model_ref import ModelRef

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
]
