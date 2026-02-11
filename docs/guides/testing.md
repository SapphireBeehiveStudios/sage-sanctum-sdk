# Testing

The SDK provides mock implementations for unit testing agents without real credentials or network access.

## Setup

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

## Mock Classes

### MockGatewayClient

Drop-in replacement for `SpiffeGatewayClient` that returns dummy credentials:

```python
from sage_sanctum.testing.mocks import MockGatewayClient

client = MockGatewayClient()
creds = client.get_credentials()
# GatewayCredentials(spiffe_jwt="mock-jwt", trat="mock-trat")

client.is_gateway_mode  # False (default)
```

Configure gateway behavior:

```python
client = MockGatewayClient(
    is_gateway=True,
    endpoints={"openai": "http://localhost:8080"},
)
```

### MockLLM

Mock LangChain chat model that returns canned responses and tracks calls:

```python
from sage_sanctum.testing.mocks import MockLLM

llm = MockLLM(responses=[
    "This code has a SQL injection vulnerability.",
    "No issues found in this file.",
])

response = llm.invoke([...])  # Returns first response
response = llm.invoke([...])  # Returns second response

# Inspect what was sent
assert len(llm.calls) == 2
assert "SQL" in llm.calls[0][0].content  # Check first call's messages
```

### MockTratClient

Mock Transaction Token client with configurable model allowlists:

```python
from sage_sanctum.testing.mocks import MockTratClient
from sage_sanctum.auth.trat import AllowedModels

client = MockTratClient(allowed_models=AllowedModels(
    triage=["openai:gpt-4o-mini"],
    analysis=["openai:gpt-4o"],
    reasoning=["openai:o1"],
    embeddings=["openai:text-embedding-3-small"],
))

token = client.get_token()
token.tctx.allowed_models.analysis  # ["openai:gpt-4o"]
```

## Pytest Fixtures

The SDK provides ready-made pytest fixtures:

```python
# In your conftest.py or test file
from sage_sanctum.testing.fixtures import mock_gateway, sample_trat, mock_context
```

### mock_gateway

Returns a `MockGatewayClient` instance:

```python
def test_something(mock_gateway):
    creds = mock_gateway.get_credentials()
    assert creds.spiffe_jwt == "mock-jwt"
```

### sample_trat

Returns a `TransactionToken` with standard model allowlists:

```python
def test_trat(sample_trat):
    assert "openai:gpt-4o" in sample_trat.tctx.allowed_models.analysis
```

### mock_context

Returns a fully configured `AgentContext` with mocked gateway and static model selector:

```python
async def test_agent(mock_context):
    agent = MyAgent(mock_context)
    # mock_context.create_llm_client() works but returns a mock-configured client
```

## Testing an Agent End-to-End

```python
import pytest
from sage_sanctum.testing.fixtures import mock_context
from sage_sanctum.io.inputs import RepositoryInput
from my_agent import MyAgent


@pytest.fixture
def agent(mock_context):
    return MyAgent(mock_context)


@pytest.fixture
def repo(tmp_path):
    # Create a minimal test repository
    (tmp_path / "app.py").write_text("print('hello')")
    (tmp_path / "utils.py").write_text("def add(a, b): return a + b")
    return RepositoryInput(path=tmp_path)


async def test_agent_produces_sarif(agent, repo):
    result = await agent.run(repo)

    assert result.exit_code == 0
    assert result.output is not None
    assert result.output.io_type == "sarif"


async def test_agent_handles_empty_repo(agent, tmp_path):
    empty_repo = RepositoryInput(path=tmp_path)
    result = await agent.run(empty_repo)

    assert result.exit_code == 0
    assert len(result.output.findings) == 0
```

## Testing Error Scenarios

```python
from sage_sanctum.errors import ModelNotAvailableError


async def test_agent_handles_model_error(agent, repo, monkeypatch):
    def fail(*args, **kwargs):
        raise ModelNotAvailableError("gpt-4o unavailable")

    monkeypatch.setattr(agent.context, "create_llm_client", fail)

    with pytest.raises(ModelNotAvailableError):
        await agent.run(repo)
```
