"""Error hierarchy for Sage Sanctum Agent SDK.

Exit code ranges:
- 10-19: Authentication errors
- 20-29: Authorization errors
- 30-39: Gateway errors
- 40-49: Validation errors
- 50-59: Output errors
- 60-69: Model errors
"""

from __future__ import annotations


class SageSanctumError(Exception):
    """Base error for all Sage Sanctum SDK errors."""

    exit_code: int = 1

    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


# ============================================================================
# Authentication errors (10-19)
# ============================================================================


class AuthError(SageSanctumError):
    """Base authentication error."""

    exit_code = 10


class SpiffeAuthError(AuthError):
    """SPIFFE JWT acquisition or validation failed."""

    exit_code = 11


class TraTAcquisitionError(AuthError):
    """Failed to acquire Transaction Token."""

    exit_code = 12


class TraTExpiredError(AuthError):
    """Transaction Token has expired."""

    exit_code = 13


# ============================================================================
# Authorization errors (20-29)
# ============================================================================


class ForbiddenError(SageSanctumError):
    """Action not authorized."""

    exit_code = 20


class ModelNotAuthorizedError(ForbiddenError):
    """Requested model not in allowed_models."""

    exit_code = 21


class ScopeNotAuthorizedError(ForbiddenError):
    """Requested scope not authorized in TraT."""

    exit_code = 22


# ============================================================================
# Gateway errors (30-39)
# ============================================================================


class GatewayError(SageSanctumError):
    """Base gateway communication error."""

    exit_code = 30


class RateLimitError(GatewayError):
    """Rate limit exceeded at gateway."""

    exit_code = 31


class GatewayUnavailableError(GatewayError):
    """Gateway is unavailable."""

    exit_code = 32


# ============================================================================
# Validation errors (40-49)
# ============================================================================


class ValidationError(SageSanctumError):
    """Base validation error."""

    exit_code = 40


class InputValidationError(ValidationError):
    """Agent input validation failed."""

    exit_code = 41


class PathTraversalError(ValidationError):
    """Path traversal attempt detected."""

    exit_code = 42


class ConfigurationError(ValidationError):
    """Invalid configuration."""

    exit_code = 43


# ============================================================================
# Output errors (50-59)
# ============================================================================


class OutputError(SageSanctumError):
    """Base output error."""

    exit_code = 50


class OutputWriteError(OutputError):
    """Failed to write output files."""

    exit_code = 51


class SarifValidationError(OutputError):
    """SARIF output validation failed."""

    exit_code = 52


# ============================================================================
# Model errors (60-69)
# ============================================================================


class ModelError(SageSanctumError):
    """Base model error."""

    exit_code = 60


class ModelNotAvailableError(ModelError):
    """Requested model is not available."""

    exit_code = 61


class ModelRefParseError(ModelError):
    """Failed to parse model reference."""

    exit_code = 62
