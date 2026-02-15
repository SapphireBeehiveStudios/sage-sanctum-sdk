# Sage Sanctum Agent SDK

SDK for building agents that run within the [Sage Sanctum](https://sagesecurity.ai) secure multi-agent execution platform.

## Features

- **SPIFFE Authentication** - Automatic JWT SVID management for agent identity
- **Transaction Tokens (TraT)** - IETF-standard authorization for scoped transactions
- **LLM Gateway Integration** - Route LLM calls through authenticated, policy-enforced gateways
- **Multi-Provider Support** - OpenAI, Anthropic, and Google via unified interface
- **SARIF Output** - Standard static analysis output format for GitHub Code Scanning
- **External LLM Support** - Wrap tools like Claude Code (via Agent SDK) that manage their own LLM calls
- **Testing Utilities** - Mock gateway, LLM, and TraT clients for unit testing

## Quick Start

```python
from sage_sanctum import AgentContext, AgentRunner, SageSanctumAgent, AgentResult
from sage_sanctum.io.inputs import AgentInput, RepositoryInput
from sage_sanctum.io.outputs import SarifOutput, Finding, Location
from sage_sanctum.llm.model_category import ModelCategory


class MySecurityAgent(SageSanctumAgent[RepositoryInput]):
    @property
    def name(self) -> str:
        return "my-security-agent"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: RepositoryInput) -> AgentResult:
        # Get an LLM client for analysis
        llm = self.context.create_llm_client(ModelCategory.ANALYSIS)

        # Your agent logic here...
        response = llm.invoke([...])

        return AgentResult(
            output=SarifOutput(
                tool_name=self.name,
                tool_version=self.version,
                findings=[...],
            ),
            exit_code=0,
        )


# Entry point
if __name__ == "__main__":
    import sys
    sys.exit(AgentRunner(MySecurityAgent).run())
```

## Installation

```bash
pip install sage-sanctum-sdk
```

## Architecture

Agents run in isolated containers with no direct network access. All external communication (LLM calls, MCP tools, packages) flows through authenticated gateways:

```
Agent Pod (seccomp: no AF_INET)
  └── Agent SDK
       ├── SPIFFE JWT (identity)
       ├── Transaction Token (authorization)
       └── Gateway Client (Unix socket)
            └── LLM Gateway → OpenAI / Anthropic / Google
```

### External LLM Mode

For agents that wrap tools like Claude Code (via the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)), set `requires_gateway = False`. The SDK handles only I/O and lifecycle; the external tool communicates with the gateway via an auth-injecting bridge:

```
Scanner Pod
├── Python Agent (requires_gateway=False)
│   └── Claude Agent SDK (async subprocess)
│       └── ANTHROPIC_BASE_URL=http://127.0.0.1:8082
├── bridge.py (injects SPIFFE JWT + TraT headers)
├── Forwarder sidecar (zero-logic, UDS -> central gateway)
└── Central LLM Gateway (validates auth, BYOK key injection)
```

## Documentation

Full documentation is available at [sapphirebeehivestudios.github.io/sage-sanctum-sdk](https://sapphirebeehivestudios.github.io/sage-sanctum-sdk/).

## License

MIT
