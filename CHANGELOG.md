# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-14

### Added
- Streaming support (`_stream()`) for `GatewayChatModel` via SSE
- Tool/function calling support in `_messages_to_dicts()` and response parsing
- Token usage tracking via `usage_metadata` on `AIMessage`
- Generation parameters: `max_tokens`, `top_p`, `seed` on `GatewayChatModel`
- Retry logic with exponential backoff and jitter on `GatewayHttpClient`
- `RateLimitError` detection (429) with automatic retry and `Retry-After` support
- Connection health check (`health_check()`) on `GatewayHttpClient` and `check_gateway_health()` on `AgentContext`
- `AllowedModels.has_any()` method for checking if any category has models
- `GatewayCredentials.__repr__`/`__str__` with token masking
- `py.typed` marker for PEP 561 typed package support
- `provider` parameter on `GatewayEmbeddings` and `create_embeddings_for_gateway()`
- Cooperative shutdown via `asyncio.wait()` in `AgentRunner`
- `CLAUDE.md` project guide
- `CHANGELOG.md` retroactive changelog

### Fixed
- `__version__` now reads from `pyproject.toml` via `importlib.metadata` (was hardcoded to `0.1.1`)
- `with_structured_output(include_raw=True)` — was piping `AIMessage` into `RunnablePassthrough.assign()` which expects dict
- Path traversal check uses `path.parts` instead of substring match (fixes false positives on names like `..foo`)
- `TimeoutError` in `_send_raw` now raises `GatewayError` instead of silently truncating response
- Model selector checks any category via `has_any()`, not just triage
- Embedding provider no longer hardcoded to `"openai"` — accepts `provider` parameter

### Changed
- MCP stub client docstring updated to clarify it's not yet available and excluded from public API

## [0.2.9] - 2025-01-XX

### Fixed
- Fix escaped CRLF in chunked transfer encoding decoder

## [0.2.8] - 2025-01-XX

### Fixed
- Detect expired JWTs and fallback to sidecar for TraTs
- Decode chunked transfer encoding in gateway responses

## [0.2.7] - 2025-01-XX

### Added
- `GatewayEmbeddings` for UDS-native embedding requests via `/v1/embeddings`

## [0.2.6] - 2025-01-XX

### Fixed
- Ensure all properties in required array for OpenAI strict mode schemas

## [0.2.5] - 2025-01-XX

### Fixed
- OpenAI strict mode schema: resolve `$ref`, strip `title`/`default`, add `additionalProperties: false`

## [0.2.4] - 2025-01-XX

### Added
- `with_structured_output()` on `GatewayChatModel` with OpenAI JSON schema mode
