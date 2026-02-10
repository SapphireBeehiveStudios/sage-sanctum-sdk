"""Tests for error hierarchy and exit codes."""

from sage_sanctum.errors import (
    AuthError,
    ConfigurationError,
    ForbiddenError,
    GatewayError,
    GatewayUnavailableError,
    InputValidationError,
    ModelError,
    ModelNotAuthorizedError,
    ModelNotAvailableError,
    ModelRefParseError,
    OutputError,
    OutputWriteError,
    PathTraversalError,
    RateLimitError,
    SageSanctumError,
    SarifValidationError,
    ScopeNotAuthorizedError,
    SpiffeAuthError,
    TraTAcquisitionError,
    TraTExpiredError,
    ValidationError,
)


class TestErrorHierarchy:
    def test_base_error(self):
        e = SageSanctumError("test")
        assert str(e) == "test"
        assert e.exit_code == 1

    def test_base_error_custom_exit_code(self):
        e = SageSanctumError("test", exit_code=99)
        assert e.exit_code == 99

    def test_auth_errors(self):
        assert AuthError("x").exit_code == 10
        assert SpiffeAuthError("x").exit_code == 11
        assert TraTAcquisitionError("x").exit_code == 12
        assert TraTExpiredError("x").exit_code == 13

    def test_authz_errors(self):
        assert ForbiddenError("x").exit_code == 20
        assert ModelNotAuthorizedError("x").exit_code == 21
        assert ScopeNotAuthorizedError("x").exit_code == 22

    def test_gateway_errors(self):
        assert GatewayError("x").exit_code == 30
        assert RateLimitError("x").exit_code == 31
        assert GatewayUnavailableError("x").exit_code == 32

    def test_validation_errors(self):
        assert ValidationError("x").exit_code == 40
        assert InputValidationError("x").exit_code == 41
        assert PathTraversalError("x").exit_code == 42
        assert ConfigurationError("x").exit_code == 43

    def test_output_errors(self):
        assert OutputError("x").exit_code == 50
        assert OutputWriteError("x").exit_code == 51
        assert SarifValidationError("x").exit_code == 52

    def test_model_errors(self):
        assert ModelError("x").exit_code == 60
        assert ModelNotAvailableError("x").exit_code == 61
        assert ModelRefParseError("x").exit_code == 62

    def test_inheritance(self):
        assert isinstance(SpiffeAuthError("x"), AuthError)
        assert isinstance(AuthError("x"), SageSanctumError)
        assert isinstance(RateLimitError("x"), GatewayError)
        assert isinstance(GatewayError("x"), SageSanctumError)
        assert isinstance(ModelNotAuthorizedError("x"), ForbiddenError)
        assert isinstance(ForbiddenError("x"), SageSanctumError)
