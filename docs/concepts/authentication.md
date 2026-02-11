# Authentication

Sage Sanctum uses a two-layer authentication model: **SPIFFE** for identity and **Transaction Tokens** for authorization.

## SPIFFE (Identity)

[SPIFFE](https://spiffe.io/) (Secure Production Identity Framework for Everyone) provides each agent with a cryptographic identity via JWT SVIDs.

### How It Works

1. The platform mounts a SPIFFE JWT file into the agent container
2. The SDK reads the JWT via `JWTSource`
3. On each gateway call, the JWT is sent as a `Bearer` token in the `Authorization` header
4. The gateway verifies the JWT signature against the SPIFFE trust bundle

```python
from sage_sanctum.auth.spiffe import JWTSource

jwt_source = JWTSource("/run/secrets/spiffe/jwt")
token = jwt_source.get_token()  # Cached, auto-refreshes before expiry
```

### Token Caching

`JWTSource` caches the JWT in memory and refreshes it 5 minutes before expiry (`_REFRESH_BUFFER_SECONDS = 300`). This avoids unnecessary file reads while ensuring tokens don't expire mid-request.

## Transaction Tokens (Authorization)

[Transaction Tokens](https://datatracker.ietf.org/doc/draft-ietf-oauth-transaction-tokens/) (TraT) are an IETF draft standard for carrying authorization context across service boundaries.

### Structure

A TraT contains:

| Claim | Description |
|-------|-------------|
| `txn` | Transaction ID |
| `sub` | Subject (agent SPIFFE ID) |
| `scope` | Authorized scopes |
| `iat` / `exp` | Issued at / expiration |
| `tctx` | Transaction context (models, providers, tools) |
| `rctx` | Requester context (trigger, actor, PR number) |

### Transaction Context (tctx)

The `tctx` claim carries the run's authorization parameters:

```python
token.tctx.allowed_models   # AllowedModels(triage=[...], analysis=[...], ...)
token.tctx.allowed_providers  # ["openai", "anthropic"]
token.tctx.allowed_tools      # {"mcp-server": ["tool1", "tool2"]}
token.tctx.run_id             # "run-abc123"
token.tctx.org_id             # "org-xyz"
token.tctx.repo_url           # "https://github.com/org/repo"
```

### Requester Context (rctx)

The `rctx` claim carries audit metadata about what triggered the run:

```python
token.rctx.trigger     # "pull_request"
token.rctx.pr_number   # 42
token.rctx.actor       # "dependabot[bot]"
token.rctx.source_ip   # "192.168.1.1"
```

### Acquiring Tokens

The `TransactionTokenClient` reads TraTs from either a file or the auth sidecar:

```python
from sage_sanctum.auth.trat import TransactionTokenClient

# From file
client = TransactionTokenClient(trat_file="/run/secrets/trat")

# From auth sidecar
client = TransactionTokenClient(sidecar_socket="/run/sockets/auth.sock")

token = client.get_token()  # Cached, checks expiry
```

## GatewayCredentials

`GatewayCredentials` bundles the SPIFFE JWT and TraT together for gateway requests:

```python
from sage_sanctum.auth.credentials import GatewayCredentials

creds = gateway_client.get_credentials()
headers = creds.auth_headers()
# {"Authorization": "Bearer <jwt>", "Txn-Token": "<trat>"}
```

!!! note
    You rarely interact with authentication directly. `AgentContext.create_llm_client()` handles credential injection automatically.
