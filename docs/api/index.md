# API Reference

## Public API Surface

The top-level `sage_sanctum` package re-exports the most commonly used classes:

```python
from sage_sanctum import (
    AgentContext,
    AgentInput,
    AgentOutput,
    AgentResult,
    AgentRunner,
    Finding,
    Location,
    ModelCategory,
    ModelRef,
    RepositoryInput,
    SageSanctumAgent,
    SageSanctumError,
    SarifOutput,
    TokenUsage,
)
```

## Package Structure

| Module | Description |
|--------|-------------|
| [`sage_sanctum.agent`](agent.md) | Agent base class and runner |
| [`sage_sanctum.context`](context.md) | Central agent context |
| [`sage_sanctum.errors`](errors.md) | Error hierarchy with exit codes |
| [`sage_sanctum.auth`](auth/index.md) | SPIFFE and Transaction Token authentication |
| [`sage_sanctum.gateway`](gateway/index.md) | LLM gateway client and HTTP transport |
| [`sage_sanctum.io`](io/index.md) | Agent input and output types |
| [`sage_sanctum.llm`](llm/index.md) | Model categories, references, and selection |
| [`sage_sanctum.mcp`](mcp/index.md) | MCP tool client (placeholder) |
| [`sage_sanctum.testing`](testing/index.md) | Mocks and fixtures for unit testing |
