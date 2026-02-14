# Sage Sanctum SDK

## Purpose

SDK for building agents that run within the Sage Sanctum infrastructure. Handles SPIFFE authentication, Transaction Tokens (TraTs), LLM gateway access over Unix domain sockets, and standardized SARIF output.

## Architecture

```
Agent -> (UDS) -> Proxy -> (TCP) -> LLM Gateway -> Provider (OpenAI/Anthropic/Google)
```

- **Production**: AF_UNIX sockets only (AF_INET blocked by seccomp)
- **Local dev**: Direct provider API calls via `SAGE_SANCTUM_ALLOW_DIRECT=1`
- Auth: SPIFFE JWT (identity) + TraT (transaction authorization) per request

## Key Files

| Path | Purpose |
|------|---------|
| `src/sage_sanctum/agent.py` | `SageSanctumAgent` base class, `AgentRunner` lifecycle |
| `src/sage_sanctum/context.py` | `AgentContext` — central context, LLM/embeddings client creation |
| `src/sage_sanctum/gateway/http.py` | Raw HTTP client over UDS/TCP with retry logic |
| `src/sage_sanctum/gateway/client.py` | `SpiffeGatewayClient` (prod) / `DirectProviderClient` (dev) |
| `src/sage_sanctum/llm/gateway_chat.py` | `GatewayChatModel` — LangChain `BaseChatModel` over gateway |
| `src/sage_sanctum/llm/gateway_embeddings.py` | `GatewayEmbeddings` — LangChain `Embeddings` over gateway |
| `src/sage_sanctum/auth/trat.py` | Transaction Token parsing and client |
| `src/sage_sanctum/auth/spiffe.py` | SPIFFE JWT source |
| `src/sage_sanctum/errors.py` | Error hierarchy with exit codes (10-69) |
| `src/sage_sanctum/io/` | Input (RepositoryInput) and output (SARIF) types |
| `src/sage_sanctum/testing/mocks.py` | Mock implementations for testing |

## Running Tests

```bash
uv run pytest tests/ -v
```

## Conventions

- **Error hierarchy**: Every error has an `exit_code` (10-69). `AgentRunner` maps these to process exit codes.
- **Factory methods**: Use `AgentContext.from_environment()` (prod) or `.for_local_development()` (dev).
- **Dual-mode design**: Gateway mode (UDS + SPIFFE) vs direct mode (API keys). Controlled by `SAGE_SANCTUM_ALLOW_DIRECT`.
- **Model selection**: TraT's `tctx.allowed_models` per category (triage/analysis/reasoning/embeddings).
- **Version**: Single source of truth in `pyproject.toml`, read via `importlib.metadata`.

## What NOT To Do

- Do not log credentials (SPIFFE JWTs, TraTs, API keys). Use `GatewayCredentials.__repr__()` which masks tokens.
- Do not add new providers without updating `ModelRef._PROVIDER_PREFIXES` and `_detect_provider()`.
- Do not use AF_INET in production — only AF_UNIX is allowed.
- Do not import `MCPGatewayClient` in public APIs — it's a stub.
