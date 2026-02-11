# Context

`AgentContext` is the central runtime object that every agent receives. It provides access to LLM clients, model selection, input loading, and output writing.

## Initialization Modes

### From Environment (Production)

```python
context = AgentContext.from_environment()
```

Reads all configuration from environment variables. Used by `AgentRunner` automatically.

### From Environment Async

```python
context = await AgentContext.from_environment_async()
```

Async variant for use in async initialization flows.

### For Local Development

```python
context = AgentContext.for_local_development(
    work_dir="/tmp/work",
    output_dir="/tmp/output",
    model="openai:gpt-4o",
)
```

Creates a context that calls LLM providers directly (no gateway). Requires `SAGE_SANCTUM_ALLOW_DIRECT=1`.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Unique run identifier |
| `org_id` | `str` | Organization identifier |
| `work_dir` | `Path` | Working directory for temporary files |
| `output_dir` | `Path` | Directory where output is written |
| `gateway_client` | `GatewayClient` | Client for LLM gateway access |
| `model_selector` | `ModelSelector` | Resolves model categories to concrete models |
| `logger` | `logging.Logger` | Logger instance for the agent |

## Key Methods

### create_llm_client

```python
llm = context.create_llm_client(ModelCategory.ANALYSIS)
response = llm.invoke([...])
```

Returns a LangChain `BaseChatModel` configured for the given category. In gateway mode, this routes through the LLM gateway with SPIFFE + TraT headers. In direct mode, it uses LiteLLM with API keys.

### load_input

```python
repo_input = context.load_input()
```

Loads agent input from the `REPO_PATH` environment variable. Returns a `RepositoryInput` with the path to the cloned repository.

### write_output

```python
files = context.write_output(sarif_output)
```

Writes agent output to the output directory. Returns a list of filenames written.

## Environment Variables

The context reads the following environment variables during initialization:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUN_ID` | Yes | — | Run identifier |
| `ORG_ID` | Yes | — | Organization identifier |
| `WORK_DIR` | No | `/work` | Working directory |
| `OUTPUT_PATH` | No | `/output` | Output directory |
| `SPIFFE_JWT_PATH` | No | — | Path to SPIFFE JWT file |
| `TRAT_FILE` | No | — | Path to TraT file |
| `AUTH_SIDECAR_SOCKET` | No | — | Auth sidecar Unix socket |
| `LLM_GATEWAY_SOCKET` | No | — | LLM gateway Unix socket |
| `SAGE_SANCTUM_ALLOW_DIRECT` | No | — | Set to `1` for direct mode |
| `SAGE_MODEL` / `OPENAI_MODEL` | No | — | Model override |

See the [Configuration](../guides/configuration.md) guide for the full reference.
