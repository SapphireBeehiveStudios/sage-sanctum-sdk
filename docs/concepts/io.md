# Input & Output

## Input

### RepositoryInput

Agents receive a `RepositoryInput` representing a cloned repository to analyze:

```python
from sage_sanctum.io.inputs import RepositoryInput

repo = context.load_input()

repo.path       # Path("/work/repo") — absolute path to repo root
repo.ref        # "main" — git ref
repo.url        # "https://github.com/org/repo"
```

#### Listing Files

```python
# All files
all_files = repo.list_files()

# Only Python files
py_files = repo.list_files(extensions={".py"})

# Only JavaScript and TypeScript
js_files = repo.list_files(extensions={".js", ".ts"})
```

#### Validation

`RepositoryInput.validate()` checks that:

- The path exists
- The path is a directory
- The path doesn't contain path traversal (`..`)

This is called automatically when loading input.

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `REPO_PATH` | Absolute path to the cloned repository |
| `REPO_REF` | Git ref (branch, tag, commit hash) |
| `REPO_URL` | Remote repository URL |

## Output

### SarifOutput

The primary output format is [SARIF 2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html), the standard for GitHub Code Scanning results.

```python
from sage_sanctum.io.outputs import SarifOutput, Finding, Location, TokenUsage

output = SarifOutput(
    tool_name="my-agent",
    tool_version="1.0.0",
    findings=[
        Finding(
            id="SQL-001",
            title="SQL Injection",
            description="User input passed directly to SQL query",
            severity="high",
            location=Location(
                file="src/db.py",
                start_line=42,
                end_line=42,
                start_column=5,
                end_column=38,
            ),
            cwe="CWE-89",
            remediation="Use parameterized queries",
            confidence="high",
        ),
    ],
    token_usage=[
        TokenUsage(
            model="gpt-4o",
            prompt_tokens=1500,
            completion_tokens=300,
            total_tokens=1800,
        ),
    ],
)
```

### Finding

Each finding represents a single issue discovered by the agent:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique finding identifier (e.g., `"SQL-001"`) |
| `title` | `str` | Short title |
| `description` | `str` | Detailed description |
| `severity` | `str` | `"critical"`, `"high"`, `"medium"`, `"low"`, or `"note"` |
| `location` | `Location` | Source code location |
| `cwe` | `str` | CWE reference (e.g., `"CWE-89"`) |
| `remediation` | `str` | Fix suggestion |
| `confidence` | `str` | `"high"`, `"medium"`, or `"low"` |
| `metadata` | `dict` | Additional data |

### Location

```python
Location(
    file="src/db.py",
    start_line=42,
    end_line=45,
    start_column=5,
    end_column=20,
)
```

### Severity Mapping

SARIF uses a different severity model. The SDK maps automatically:

| SDK Severity | SARIF Level | Security Score |
|-------------|-------------|----------------|
| `critical` | `error` | 9.0 |
| `high` | `error` | 7.0 |
| `medium` | `warning` | 4.0 |
| `low` | `note` | 1.0 |
| `note` | `note` | 0.0 |

### Writing Output

Output is written automatically by `AgentRunner`, or you can write manually:

```python
files = context.write_output(sarif_output)
# ["results.sarif"]
```

### AgentResult

The `run` method wraps output in an `AgentResult`:

```python
return AgentResult(
    output=sarif_output,
    exit_code=0,
    metadata={"files_analyzed": 42},
)
```
