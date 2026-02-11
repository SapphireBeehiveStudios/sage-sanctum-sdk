# Quickstart

This guide walks you through building a simple security analysis agent from scratch.

## 1. Create Your Agent

Every agent extends `SageSanctumAgent` and implements three things: a `name`, a `version`, and a `run` method.

```python
# my_agent.py
from sage_sanctum import SageSanctumAgent, AgentResult, AgentRunner
from sage_sanctum.io.inputs import AgentInput, RepositoryInput
from sage_sanctum.io.outputs import SarifOutput, Finding, Location
from sage_sanctum.llm.model_category import ModelCategory


class MySecurityAgent(SageSanctumAgent):
    @property
    def name(self) -> str:
        return "my-security-agent"

    @property
    def version(self) -> str:
        return "0.1.0"

    async def run(self, agent_input: AgentInput) -> AgentResult:
        # agent_input is a RepositoryInput with a path to the cloned repo
        repo = agent_input
        assert isinstance(repo, RepositoryInput)

        # Create an LLM client for the analysis category
        llm = self.context.create_llm_client(ModelCategory.ANALYSIS)

        # List Python files in the repository
        py_files = repo.list_files(extensions={".py"})

        findings = []
        for py_file in py_files[:5]:  # Analyze first 5 files
            code = py_file.read_text()
            response = llm.invoke([
                {"role": "system", "content": "You are a security auditor. Find vulnerabilities."},
                {"role": "user", "content": f"Review this code:\n\n{code}"},
            ])

            # Parse findings from the LLM response...
            # (simplified for this example)

        return AgentResult(
            output=SarifOutput(
                tool_name=self.name,
                tool_version=self.version,
                findings=findings,
            ),
            exit_code=0,
        )
```

## 2. Add an Entry Point

The `AgentRunner` manages the full lifecycle — context initialization, signal handling, input loading, and output writing.

```python
if __name__ == "__main__":
    import sys
    sys.exit(AgentRunner(MySecurityAgent).run())
```

## 3. Run Locally

For local development, set the `SAGE_SANCTUM_ALLOW_DIRECT` flag and provide API keys directly:

```bash
export SAGE_SANCTUM_ALLOW_DIRECT=1
export RUN_ID=local-test-001
export ORG_ID=my-org
export REPO_PATH=/path/to/repo
export OPENAI_API_KEY=sk-...

python my_agent.py
```

Or use the convenience helper:

```python
from sage_sanctum import AgentContext

context = AgentContext.for_local_development(
    work_dir="/tmp/work",
    output_dir="/tmp/output",
    model="openai:gpt-4o",
)
```

## 4. Understand the Output

The agent writes SARIF output to the output directory. SARIF (Static Analysis Results Interchange Format) is the standard format for GitHub Code Scanning:

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "my-security-agent",
        "version": "0.1.0",
        "rules": [...]
      }
    },
    "results": [...]
  }]
}
```

## 5. Write Tests

Use the SDK's built-in testing utilities:

```python
import pytest
from sage_sanctum.testing.mocks import MockGatewayClient, MockLLM
from sage_sanctum.testing.fixtures import mock_context

from my_agent import MySecurityAgent


@pytest.fixture
def agent(mock_context):
    return MySecurityAgent(mock_context)


async def test_agent_run(agent, tmp_path):
    # Create a test repo with a vulnerable file
    test_file = tmp_path / "example.py"
    test_file.write_text("import subprocess\nsubprocess.call(user_input)")

    from sage_sanctum.io.inputs import RepositoryInput
    repo_input = RepositoryInput(path=tmp_path)

    result = await agent.run(repo_input)
    assert result.exit_code == 0
    assert result.output is not None
```

## Next Steps

- [Architecture](../concepts/architecture.md) — Understand the security model
- [Agents](../concepts/agents.md) — Deep dive into agent lifecycle
- [Configuration](../guides/configuration.md) — All environment variables
- [Testing](../guides/testing.md) — Full testing guide
