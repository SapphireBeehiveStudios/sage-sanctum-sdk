"""Error hierarchy for Sage Sanctum Agent SDK.

Exit code ranges:
- 10-19: Authentication errors
- 20-29: Authorization errors
- 30-39: Gateway errors
- 40-49: Validation errors
- 50-59: Output errors
- 60-69: Model errors
- 70-79: External tool errors
"""

from __future__ import annotations


class SageSanctumError(Exception):
    """Base error for all Sage Sanctum SDK errors.

    Every subclass defines a class-level ``exit_code`` so the ``AgentRunner``
    can map exceptions to process exit codes automatically.

    Attributes:
        exit_code: Process exit code returned when this error propagates
            to ``AgentRunner``. Defaults to ``1``.

    Args:
        message: Human-readable error description.
        exit_code: Override the class-level exit code for this instance.
    """

    exit_code: int = 1

    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


# ============================================================================
# Authentication errors (10-19)
# ============================================================================


class AuthError(SageSanctumError):
    """Base authentication error (exit codes 10–19).

    Raised when agent identity or token acquisition fails.
    """

    exit_code = 10


class SpiffeAuthError(AuthError):
    """SPIFFE JWT acquisition or validation failed.

    Raised when the JWT SVID file cannot be read, is empty, or contains
    a malformed token.
    """

    exit_code = 11


class TraTAcquisitionError(AuthError):
    """Failed to acquire a Transaction Token.

    Raised when the TraT file or auth sidecar is unavailable, or the
    token is malformed.
    """

    exit_code = 12


class TraTExpiredError(AuthError):
    """Transaction Token has expired.

    Raised by ``TransactionToken.check_not_expired()`` when the ``exp``
    claim is in the past.
    """

    exit_code = 13


# ============================================================================
# Authorization errors (20-29)
# ============================================================================


class ForbiddenError(SageSanctumError):
    """Action not authorized (exit codes 20–29).

    Raised when the agent attempts an operation it is not permitted to perform.
    """

    exit_code = 20


class ModelNotAuthorizedError(ForbiddenError):
    """Requested model is not in the TraT ``allowed_models`` list.

    Raised by ``ModelSelector.validate_model()`` when the model is not
    permitted for the given category.
    """

    exit_code = 21


class ScopeNotAuthorizedError(ForbiddenError):
    """Requested scope is not authorized in the Transaction Token."""

    exit_code = 22


# ============================================================================
# Gateway errors (30-39)
# ============================================================================


class GatewayError(SageSanctumError):
    """Base gateway communication error (exit codes 30–39).

    Raised when communication with the LLM or MCP gateway fails.
    """

    exit_code = 30


class RateLimitError(GatewayError):
    """Rate limit exceeded at the gateway or upstream provider."""

    exit_code = 31


class GatewayUnavailableError(GatewayError):
    """Gateway is unreachable (connection refused, socket not found, etc.)."""

    exit_code = 32


# ============================================================================
# Validation errors (40-49)
# ============================================================================


class ValidationError(SageSanctumError):
    """Base validation error (exit codes 40–49).

    Raised when input, path, or configuration validation fails.
    """

    exit_code = 40


class InputValidationError(ValidationError):
    """Agent input validation failed.

    Raised when ``RepositoryInput.validate()`` detects a missing or
    invalid repository path.
    """

    exit_code = 41


class PathTraversalError(ValidationError):
    """Path traversal attempt detected in a file path (``..`` components)."""

    exit_code = 42


class ConfigurationError(ValidationError):
    """Missing or invalid SDK configuration.

    Raised when required environment variables are missing or mutually
    exclusive options conflict.
    """

    exit_code = 43


# ============================================================================
# Output errors (50-59)
# ============================================================================


class OutputError(SageSanctumError):
    """Base output error (exit codes 50–59).

    Raised when agent output cannot be written or is invalid.
    """

    exit_code = 50


class OutputWriteError(OutputError):
    """Failed to write output files to the output directory."""

    exit_code = 51


class SarifValidationError(OutputError):
    """SARIF output failed schema validation."""

    exit_code = 52


# ============================================================================
# Model errors (60-69)
# ============================================================================


class ModelError(SageSanctumError):
    """Base model error (exit codes 60–69).

    Raised when model selection or reference parsing fails.
    """

    exit_code = 60


class ModelNotAvailableError(ModelError):
    """No model is configured for the requested category.

    Raised by ``ModelSelector.select()`` when the TraT ``allowed_models``
    list is empty for a given ``ModelCategory``.
    """

    exit_code = 61


class ModelRefParseError(ModelError):
    """Failed to parse a model reference string.

    Raised by ``ModelRef.parse()`` when the input is empty or malformed.
    """

    exit_code = 62


# ============================================================================
# External tool errors (70-79)
# ============================================================================


class ExternalToolError(SageSanctumError):
    """Base external tool error (exit codes 70–79).

    Raised when an external subprocess or tool used by the agent fails.
    """

    exit_code = 70


class SubprocessError(ExternalToolError):
    """External subprocess exited with a non-zero return code.

    Attributes:
        returncode: The process exit code.
        stderr: Captured stderr output (may be truncated).
    """

    exit_code = 71

    def __init__(
        self,
        message: str,
        *,
        returncode: int = 1,
        stderr: str = "",
        exit_code: int | None = None,
    ):
        super().__init__(message, exit_code=exit_code)
        self.returncode = returncode
        self.stderr = stderr


class SubprocessTimeoutError(ExternalToolError):
    """External subprocess exceeded its time limit."""

    exit_code = 72


class OutputParseError(ExternalToolError):
    """Failed to parse output from an external tool."""

    exit_code = 73
