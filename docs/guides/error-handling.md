# Error Handling

## Error Hierarchy

All SDK errors extend `SageSanctumError`, which carries an `exit_code` for process-level signaling:

```
SageSanctumError (1)
├── AuthError (10)
│   ├── SpiffeAuthError (11)
│   ├── TraTAcquisitionError (12)
│   └── TraTExpiredError (13)
├── ForbiddenError (20)
│   ├── ModelNotAuthorizedError (21)
│   └── ScopeNotAuthorizedError (22)
├── GatewayError (30)
│   ├── RateLimitError (31)
│   └── GatewayUnavailableError (32)
├── ValidationError (40)
│   ├── InputValidationError (41)
│   ├── PathTraversalError (42)
│   └── ConfigurationError (43)
├── OutputError (50)
│   ├── OutputWriteError (51)
│   └── SarifValidationError (52)
├── ModelError (60)
│   ├── ModelNotAvailableError (61)
│   └── ModelRefParseError (62)
└── ExternalToolError (70)
    ├── SubprocessError (71)
    ├── SubprocessTimeoutError (72)
    └── OutputParseError (73)
```

## Exit Codes

The `AgentRunner` maps exceptions to exit codes automatically:

| Range | Category | Meaning |
|-------|----------|---------|
| 0 | Success | Agent completed successfully |
| 1 | General | Unhandled exception |
| 10-19 | Authentication | Identity or token errors |
| 20-29 | Authorization | Permission denied |
| 30-39 | Gateway | LLM gateway errors |
| 40-49 | Validation | Input or config errors |
| 50-59 | Output | Output writing errors |
| 60-69 | Model | Model selection errors |
| 70-79 | External Tool | Subprocess or external tool errors |

## Catching Errors

### Catch Specific Errors

```python
from sage_sanctum.errors import RateLimitError, ModelNotAvailableError

try:
    response = llm.invoke(messages)
except RateLimitError:
    # Back off and retry
    ...
except ModelNotAvailableError as e:
    # Fall back to a different model
    ...
```

### Catch by Category

```python
from sage_sanctum.errors import GatewayError, AuthError

try:
    response = llm.invoke(messages)
except AuthError:
    # Any authentication issue
    ...
except GatewayError:
    # Any gateway issue (rate limit, unavailable, etc.)
    ...
```

### Let the Runner Handle It

In most cases, you can let exceptions propagate. The `AgentRunner` catches all `SageSanctumError` subclasses and maps them to the correct exit code:

```python
async def run(self, agent_input: AgentInput) -> AgentResult:
    # If this raises ModelNotAvailableError, the runner
    # returns exit code 61 automatically
    llm = self.context.create_llm_client(ModelCategory.ANALYSIS)
    ...
```

## Custom Exit Codes

You can set a custom exit code when raising an error:

```python
raise SageSanctumError("Custom error", exit_code=99)
```

Or return it in the result:

```python
return AgentResult(exit_code=2, error="Partial failure")
```

## External Tool Errors

When agents wrap external tools (e.g., Claude Code CLI), use the `ExternalToolError` family to report failures:

```python
from sage_sanctum.errors import SubprocessError, SubprocessTimeoutError, OutputParseError

# Subprocess failed
raise SubprocessError(
    "Claude Code exited with error",
    returncode=1,
    stderr="Error: model not found",
)

# Subprocess timed out
raise SubprocessTimeoutError("Claude Code exceeded 600s time limit")

# Could not parse tool output
raise OutputParseError("Failed to extract findings JSON from Claude Code output")
```

`SubprocessError` carries additional attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `returncode` | `int` | The subprocess exit code (default: `1`) |
| `stderr` | `str` | Captured stderr output |
