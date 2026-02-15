# Claude Code Integration

This guide covers building a Sage Sanctum agent that uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (via the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)) as its LLM engine, routing all traffic through the Sage Sanctum gateway for authentication and policy enforcement.

## Architecture

```
Scanner Pod (scan-run namespace)
+-- Init Containers
|   +-- fetch-jwt   -> /var/run/spiffe/jwt   (SPIRE agent)
|   +-- write-trat  -> /run/sage/trat.jwt    (Orchestrator-issued TraT)
|
+-- scanner container
|   +-- Python Agent (SageSanctumAgent, requires_gateway=False)
|   |   +-- Claude Agent SDK (query() async generator)
|   |   |   +-- claude CLI subprocess
|   |   |       +-- ANTHROPIC_BASE_URL=http://127.0.0.1:8082
|   |   +-- MCP tool: report_finding (in-process, structured output)
|   |
|   +-- bridge.py (auth-injecting HTTP proxy)
|       +-- Listens: 127.0.0.1:8082 (TCP, loopback only)
|       +-- Reads JWT + TraT from files per request
|       +-- Injects Authorization: Bearer <jwt>
|       +-- Injects Txn-Token: <trat>
|       +-- Forwards to UDS: /run/sage/llm.sock
|
+-- llm-gateway sidecar (forwarder mode)
    +-- Listens: /run/sage/llm.sock (UDS)
    +-- Zero-logic reverse proxy, forwards all headers unchanged
    +-- No secrets, no config
            |
            | TCP (ClusterIP)
            v
Central LLM Gateway (scan-infra namespace, full mode)
    +-- Validates Authorization (SPIFFE JWT) + Txn-Token (TraT)
    +-- DLP scanning
    +-- BYOK key injection (adds real provider API key)
    +-- Request translation
    +-- Forwards to upstream provider (Anthropic API)
```

Claude Code doesn't natively support SPIFFE/TraT authentication, so an **auth-injecting HTTP proxy** (`bridge.py`) runs inside the scanner container to bridge the gap. It reads JWT and TraT from files (mounted by init containers), injects them as headers on every request, and forwards to the forwarder sidecar over UDS. The forwarder is zero-logic — all intelligence (auth validation, DLP, BYOK key injection) lives at the central gateway.

## Prerequisites

```bash
pip install sage-sanctum-sdk claude-agent-sdk
```

## Agent Implementation

```python
import asyncio
import os

from claude_agent_sdk import (
    ClaudeAgentOptions,
    create_sdk_mcp_server,
    query,
    tool,
)

from sage_sanctum import AgentResult, SageSanctumAgent
from sage_sanctum.errors import ExternalToolError
from sage_sanctum.io.inputs import RepositoryInput
from sage_sanctum.io.outputs import Finding, Location, SarifOutput

SCAN_PROMPT = """\
You are a security scanner. Analyze this repository for vulnerabilities.

For each vulnerability you find, call the report_finding tool with structured
data. Be thorough: check for injection flaws, auth bypass, secrets in code,
path traversal, SSRF, insecure deserialization, and other OWASP Top 10 issues.

When finished scanning, provide a brief summary of what you found.
"""

BRIDGE_PORT = 8082


class ClaudeCodeAgent(SageSanctumAgent[RepositoryInput]):
    requires_gateway = False

    @property
    def name(self) -> str:
        return "claude-code-scanner"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: RepositoryInput) -> AgentResult:
        findings: list[Finding] = []

        # --- Define MCP tool for structured finding collection ---

        @tool(
            name="report_finding",
            description="Report a security vulnerability found during analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Rule ID e.g. SQL-001",
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": [
                            "critical", "high", "medium", "low", "note",
                        ],
                    },
                    "file": {
                        "type": "string",
                        "description": "Path relative to repo root",
                    },
                    "start_line": {"type": "integer"},
                    "cwe": {
                        "type": "string",
                        "description": "e.g. CWE-89",
                    },
                    "remediation": {"type": "string"},
                },
                "required": [
                    "id", "title", "description",
                    "severity", "file", "start_line",
                ],
            },
        )
        async def handle_finding(args):
            findings.append(
                Finding(
                    id=args["id"],
                    title=args["title"],
                    description=args["description"],
                    severity=args["severity"],
                    location=Location(
                        file=args["file"],
                        start_line=args["start_line"],
                    ),
                    cwe=args.get("cwe", ""),
                    remediation=args.get("remediation", ""),
                )
            )
            return {
                "content": [
                    {"type": "text", "text": f"Recorded {args['id']}"}
                ]
            }

        mcp_server = create_sdk_mcp_server([handle_finding])

        # --- Run Claude via Agent SDK ---
        # bridge.py is started separately (by entrypoint or supervisor)
        # and listens on 127.0.0.1:BRIDGE_PORT

        cost = 0.0
        async for message in query(
            prompt=SCAN_PROMPT,
            options=ClaudeAgentOptions(
                cwd=str(agent_input.path),
                max_turns=30,
                model="claude-sonnet-4-5-20250929",
                permission_mode="bypassPermissions",
                mcp_servers={"scanner": mcp_server},
                env={
                    "ANTHROPIC_BASE_URL": (
                        f"http://127.0.0.1:{BRIDGE_PORT}"
                    ),
                    "ANTHROPIC_API_KEY": "gateway-injected",
                },
            ),
        ):
            if message.type == "result":
                cost = getattr(message, "cost_usd", 0.0)

        return AgentResult(
            output=SarifOutput(
                tool_name=self.name,
                tool_version=self.version,
                findings=findings,
            ),
            exit_code=0,
            metadata={"cost_usd": cost},
        )
```

## The Auth Bridge

Claude Code sends requests to `http://127.0.0.1:8082` with a placeholder `x-api-key`. The bridge (`bridge.py`) is a Python asyncio TCP server that:

1. Listens on `127.0.0.1:8082` (loopback only — seccomp blocks non-loopback AF_INET)
2. Parses each incoming HTTP request
3. Strips existing `Authorization:` and `Txn-Token:` headers (defense in depth)
4. Reads fresh SPIFFE JWT from `/var/run/spiffe/jwt` and TraT from `/run/sage/trat.jwt` (handles token rotation during long scans)
5. Injects `Authorization: Bearer <jwt>` and `Txn-Token: <trat>` headers
6. Forwards the modified request to the forwarder sidecar via Unix domain socket at `/run/sage/llm.sock`
7. Pipes the response back to Claude Code (supports both JSON and streaming SSE)
8. Returns HTTP 502 if the UDS connection fails

```
Claude Code
  -> http://127.0.0.1:8082 (x-api-key: gateway-injected)
  -> bridge.py (strips x-api-key, injects JWT + TraT)
  -> /run/sage/llm.sock (UDS)
  -> forwarder sidecar (passes all headers unchanged)
  -> central gateway (validates JWT + TraT, injects BYOK API key)
  -> api.anthropic.com
```

### Comparison with Standard SAGE Auth

| Aspect | SAGE Scanner | Claude Scanner |
|--------|-------------|----------------|
| Auth injection | SDK (`GatewayCredentials.auth_headers()`) | `bridge.py` (HTTP proxy) |
| Token reading | SDK reads files at startup + refresh | Bridge re-reads files per request |
| LLM endpoint | `OPENAI_BASE_URL` via UDS | `ANTHROPIC_BASE_URL` via TCP to bridge |
| API format | OpenAI-compatible (translated by gateway) | Native Anthropic (passthrough) |
| Extra hop | None (SDK -> UDS -> forwarder) | bridge.py (SDK -> TCP -> bridge -> UDS -> forwarder) |

## Token Lifecycle

### SPIFFE JWT

The `fetch-jwt` init container calls the SPIRE agent (mounted via CSI driver) and obtains a JWT for the scanner's per-run identity:

```
spiffe://sage-sanctum.local/scanner/{run_id}
```

Written to `/var/run/spiffe/jwt`. The bridge re-reads this file on every request to handle rotation.

### Transaction Token (TraT)

The orchestrator's TTS (Transaction Token Service) issues a TraT when creating the scanner job. The `write-trat` init container writes it to `/run/sage/trat.jwt` with `umask 077`. The bridge re-reads this file on every request.

## Entrypoint

```python
# __main__.py
from sage_sanctum import AgentRunner
from .agent import ClaudeCodeAgent

if __name__ == "__main__":
    exit(AgentRunner(ClaudeCodeAgent).run())
```

## Dockerfile

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# PyPI wheel bundles the claude CLI binary
RUN pip install --no-cache-dir claude-agent-sdk

COPY . /app
RUN pip install --no-cache-dir /app

RUN mkdir -p /work /output /repo && useradd -r -m scanner
USER scanner

ENV CLAUDE_CODE_ACCEPT_TOS=1 \
    DISABLE_AUTOUPDATER=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

ENTRYPOINT ["python", "-m", "my_scanner"]
```

## Running

### Production

The orchestrator creates the scanner pod with init containers, forwarder sidecar, and the repo mounted:

```bash
docker run --rm \
  --security-opt seccomp=seccomp-claude-scanner.json \
  -e RUN_ID=scan-42 \
  -e ORG_ID=acme \
  -e REPO_PATH=/repo \
  -e OUTPUT_PATH=/output \
  -v /path/to/repo:/repo:ro \
  -v /tmp/output:/output \
  my-scanner:latest

# SARIF output at /tmp/output/results.sarif
```

!!! note
    In production, the pod spec includes init containers (`fetch-jwt`, `write-trat`), the forwarder sidecar, and volume mounts for JWT/TraT files and the gateway UDS.

### Local Development

For local testing without the gateway, use direct mode with your own API key:

```bash
export SAGE_SANCTUM_ALLOW_DIRECT=1
export RUN_ID=local-test
export ORG_ID=dev
export REPO_PATH=/path/to/repo
export ANTHROPIC_API_KEY=sk-ant-...

python -m my_scanner
```

!!! note
    In direct/local mode the bridge is not needed. Adjust the agent to set `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` directly from environment when `LLM_GATEWAY_SOCKET` is not set.

## MCP Tools Instead of JSON Parsing

Rather than asking Claude to produce a JSON array and parsing free text, the agent defines a `report_finding` MCP tool. Claude calls it with typed arguments for each vulnerability found. This is dramatically more reliable than text parsing.

## Security Controls

| Control | Detail |
|---------|--------|
| Seccomp | `seccomp-claude-scanner.json` — allows AF_INET for bridge loopback, AF_UNIX for UDS, blocks AF_INET6 and raw sockets |
| NetworkPolicy | Scanner pods can only reach the central gateway in `scan-infra` namespace |
| Read-only rootfs | All containers use `readOnlyRootFilesystem: true` |
| No privilege escalation | `allowPrivilegeEscalation: false`, all capabilities dropped |

## Central Gateway Requirements

For this pattern the central gateway must:

| Requirement | Reason |
|-------------|--------|
| Validate SPIFFE JWT | Verify scanner identity against SPIRE trust domain |
| Validate TraT | Verify transaction authorization against orchestrator signing key |
| Route `/v1/messages` to Anthropic | Claude Code does not send `X-Provider` header |
| BYOK key injection | Inject real provider API key after auth validation |
| Support SSE streaming | Claude Code streams responses by default |
| Timeout >= max turn duration | Long-running analysis may take minutes per turn |

## Forwarder vs Full Mode

The sidecar runs in **forwarder mode** — a zero-logic reverse proxy (~100 LOC) that passes all headers unchanged:

- Zero secrets (no API keys)
- Zero config volumes
- 64Mi memory request / 128Mi limit
- No middleware chain

All intelligence (auth validation, DLP, BYOK key injection, request translation) lives at the central LLM gateway in the `scan-infra` namespace running in **full mode**.
