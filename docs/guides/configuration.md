# Configuration

All SDK configuration is done through environment variables. No config files are needed.

## Core Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUN_ID` | Yes | — | Unique identifier for this agent run |
| `ORG_ID` | Yes | — | Organization identifier |
| `WORK_DIR` | No | `/work` | Working directory for temporary files |
| `OUTPUT_PATH` | No | `/output` | Directory where agent output is written |

## Repository Input

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REPO_PATH` | Yes | — | Path to the cloned repository |
| `REPO_REF` | No | `""` | Git ref (branch, tag, commit) |
| `REPO_URL` | No | `""` | Remote repository URL |

## Authentication (Production)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SPIFFE_JWT_PATH` | No | — | Path to SPIFFE JWT SVID file |
| `TRAT_FILE` | No | — | Path to Transaction Token file |
| `AUTH_SIDECAR_SOCKET` | No | — | Unix socket path for auth sidecar |
| `LLM_GATEWAY_SOCKET` | No | — | Unix socket path for LLM gateway |

## Local Development

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SAGE_SANCTUM_ALLOW_DIRECT` | No | — | Set to `1` to enable direct provider access |
| `SAGE_MODEL` | No | — | Override model for all categories |
| `OPENAI_MODEL` | No | — | Override model (fallback if `SAGE_MODEL` not set) |

## Provider API Keys (Direct Mode Only)

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google AI |

## Example: Production

In production, the Sage Sanctum platform sets all environment variables automatically. Your agent doesn't need to configure anything.

```bash
# Set by the platform — you don't need to set these
RUN_ID=run-abc123
ORG_ID=org-xyz
REPO_PATH=/work/repo
SPIFFE_JWT_PATH=/run/secrets/spiffe/jwt
TRAT_FILE=/run/secrets/trat
LLM_GATEWAY_SOCKET=/run/sockets/gateway.sock
OUTPUT_PATH=/output
```

## Example: Local Development

```bash
export SAGE_SANCTUM_ALLOW_DIRECT=1
export RUN_ID=local-test
export ORG_ID=dev
export REPO_PATH=/path/to/repo
export OPENAI_API_KEY=sk-...

python my_agent.py
```

## Example: Local with Anthropic

```bash
export SAGE_SANCTUM_ALLOW_DIRECT=1
export RUN_ID=local-test
export ORG_ID=dev
export REPO_PATH=/path/to/repo
export SAGE_MODEL=anthropic:claude-3-5-sonnet-latest
export ANTHROPIC_API_KEY=sk-ant-...

python my_agent.py
```
