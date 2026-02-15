# Claude Code Integration

This guide covers building a Sage Sanctum agent that uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (via the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)) as its LLM engine, routing all traffic through the Sage Sanctum gateway for credential injection and policy enforcement.

## Architecture

```
Container
+-- Python Agent (SageSanctumAgent, requires_gateway=False)
|   +-- Claude Agent SDK (query() async generator)
|   |   +-- claude CLI subprocess
|   |       +-- ANTHROPIC_BASE_URL=http://127.0.0.1:8082
|   +-- MCP tool: report_finding (in-process, structured output)
|
+-- socat bridge (TCP:8082 -> UDS:/run/gateway.sock)
|
+-- LLM Gateway sidecar
    +-- Credential injection (adds API keys)
    +-- SPIFFE/TraT validation
    +-- Routes /v1/messages -> Anthropic API
```

The key insight: Claude Code speaks HTTP over TCP, but the gateway listens on a Unix Domain Socket. A **socat bridge** translates between them. The gateway handles credential injection, so no API keys are needed in the container.

## Prerequisites

```bash
pip install sage-sanctum-sdk claude-agent-sdk
apt-get install socat  # for the TCP-to-UDS bridge
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


class ClaudeCodeAgent(SageSanctumAgent[RepositoryInput]):
    requires_gateway = False

    @property
    def name(self) -> str:
        return "claude-code-scanner"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: RepositoryInput) -> AgentResult:
        gateway_socket = os.environ.get(
            "LLM_GATEWAY_SOCKET", "/run/llm-gateway.sock"
        )
        bridge_port = 8082
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

        # --- Start socat bridge: TCP loopback -> gateway UDS ---

        bridge = await asyncio.create_subprocess_exec(
            "socat",
            f"TCP-LISTEN:{bridge_port},bind=127.0.0.1,fork,reuseaddr",
            f"UNIX-CONNECT:{gateway_socket}",
        )

        try:
            await asyncio.sleep(0.3)
            if bridge.returncode is not None:
                raise ExternalToolError("socat bridge failed to start")

            # --- Run Claude via Agent SDK ---

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
                            f"http://127.0.0.1:{bridge_port}"
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

        finally:
            bridge.terminate()
            await bridge.wait()
```

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

RUN apt-get update && apt-get install -y --no-install-recommends git socat && \
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

The orchestrator sets up the gateway sidecar and mounts the repo:

```bash
docker run --rm \
  --security-opt seccomp=sage-sanctum.json \
  -e RUN_ID=scan-42 \
  -e ORG_ID=acme \
  -e REPO_PATH=/repo \
  -e OUTPUT_PATH=/output \
  -e LLM_GATEWAY_SOCKET=/run/llm-gateway.sock \
  -v /path/to/repo:/repo:ro \
  -v /var/run/gateway:/run:ro \
  -v /tmp/output:/output \
  my-scanner:latest

# SARIF output at /tmp/output/results.sarif
```

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
    In direct/local mode you don't need socat. Adjust the agent to skip the bridge when `LLM_GATEWAY_SOCKET` is not set.

## How It Works

### The Bridge

Claude Code (Node.js) speaks HTTP over TCP. The Sage Sanctum gateway listens on a Unix Domain Socket. socat bridges them:

```
Claude Code -> TCP 127.0.0.1:8082 -> socat -> AF_UNIX /run/gateway.sock -> Gateway
```

### Credential Injection

The gateway sees standard Anthropic API requests:

```
POST /v1/messages HTTP/1.1
x-api-key: gateway-injected
anthropic-version: 2023-06-01
Content-Type: application/json
```

It strips the placeholder `x-api-key`, injects the real Anthropic API key, validates the request against SPIFFE/TraT policy, and forwards to `api.anthropic.com`.

### MCP Tools Instead of JSON Parsing

Rather than asking Claude to produce a JSON array and parsing free text, the agent defines a `report_finding` MCP tool. Claude calls it with typed arguments for each vulnerability found. This is dramatically more reliable than text parsing.

### Seccomp Considerations

The socat bridge binds to `127.0.0.1` (loopback). If the seccomp profile blocks all `AF_INET` including loopback, you need to either:

1. **Allow loopback in seccomp** (recommended) and block external egress via Kubernetes NetworkPolicy
2. **Use a pure UDS path** with a Node.js `--require` loader that patches HTTP to use socketPath (fragile, not recommended)

## Gateway Requirements

For this pattern the gateway must:

| Requirement | Reason |
|-------------|--------|
| Route `/v1/messages` to Anthropic | Claude Code does not send `X-Provider` header |
| Replace `x-api-key` header | Inject real API key, strip placeholder |
| Support SSE streaming | Claude Code streams responses by default |
| Timeout >= max turn duration | Long-running analysis may take minutes per turn |
