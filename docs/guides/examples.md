# Examples

## Multi-Model Agent

Use different model categories for different stages of analysis:

```python
from sage_sanctum import SageSanctumAgent, AgentResult
from sage_sanctum.io.inputs import AgentInput, RepositoryInput
from sage_sanctum.io.outputs import SarifOutput, Finding, Location
from sage_sanctum.llm.model_category import ModelCategory


class MultiModelAgent(SageSanctumAgent):
    @property
    def name(self) -> str:
        return "multi-model-agent"

    @property
    def version(self) -> str:
        return "1.0.0"

    async def run(self, agent_input: AgentInput) -> AgentResult:
        repo = agent_input
        assert isinstance(repo, RepositoryInput)

        # Stage 1: Triage — fast, cheap model to identify interesting files
        triage_llm = self.context.create_llm_client(ModelCategory.TRIAGE)
        py_files = repo.list_files(extensions={".py"})
        interesting_files = []

        for f in py_files:
            code = f.read_text()
            resp = triage_llm.invoke([
                {"role": "user", "content": f"Does this file handle user input? Answer YES or NO.\n\n{code}"},
            ])
            if "YES" in resp.content:
                interesting_files.append(f)

        # Stage 2: Analysis — thorough review of flagged files
        analysis_llm = self.context.create_llm_client(ModelCategory.ANALYSIS)
        findings = []

        for f in interesting_files:
            code = f.read_text()
            resp = analysis_llm.invoke([
                {"role": "system", "content": "Find security vulnerabilities. Return JSON."},
                {"role": "user", "content": code},
            ])
            # Parse findings from response...

        return AgentResult(
            output=SarifOutput(
                tool_name=self.name,
                tool_version=self.version,
                findings=findings,
            ),
            exit_code=0,
        )
```

## Local Development Setup

Run your agent locally without the Sage Sanctum platform:

```python
import asyncio
from sage_sanctum import AgentContext
from sage_sanctum.io.inputs import RepositoryInput
from my_agent import MyAgent


async def main():
    context = AgentContext.for_local_development(
        work_dir="/tmp/work",
        output_dir="/tmp/output",
        model="openai:gpt-4o",
    )

    agent = MyAgent(context)
    repo = RepositoryInput(path="/path/to/repo")

    result = await agent.run(repo)
    print(f"Exit code: {result.exit_code}")
    print(f"Findings: {len(result.output.findings)}")

    # Write output
    files = context.write_output(result.output)
    print(f"Written: {files}")


asyncio.run(main())
```

## SARIF Output with Full Detail

Create findings with complete location and metadata:

```python
from sage_sanctum.io.outputs import SarifOutput, Finding, Location, TokenUsage

output = SarifOutput(
    tool_name="vuln-scanner",
    tool_version="2.0.0",
    findings=[
        Finding(
            id="SQLI-001",
            title="SQL Injection via string formatting",
            description=(
                "User-controlled input is passed directly into an SQL query "
                "using f-string formatting. An attacker could modify the query "
                "to extract or modify data."
            ),
            severity="critical",
            location=Location(
                file="src/database/queries.py",
                start_line=42,
                end_line=45,
                start_column=12,
                end_column=58,
            ),
            cwe="CWE-89",
            remediation="Use parameterized queries with cursor.execute(sql, params).",
            confidence="high",
            metadata={"owasp": "A03:2021"},
        ),
        Finding(
            id="XSS-001",
            title="Reflected XSS in template rendering",
            description="User input rendered without escaping in Jinja2 template.",
            severity="high",
            location=Location(
                file="src/web/views.py",
                start_line=88,
                end_line=88,
            ),
            cwe="CWE-79",
            remediation="Use {{ value | e }} or enable autoescape.",
            confidence="medium",
        ),
    ],
    token_usage=[
        TokenUsage(model="gpt-4o", prompt_tokens=5000, completion_tokens=1200, total_tokens=6200),
        TokenUsage(model="gpt-4o-mini", prompt_tokens=800, completion_tokens=50, total_tokens=850),
    ],
)

# Write to disk
sarif_dict = output.to_dict()
files = output.write("/tmp/output")
```

## Custom Agent Input Validation

Add custom validation to your agent:

```python
from sage_sanctum import SageSanctumAgent, AgentResult
from sage_sanctum.io.inputs import AgentInput, RepositoryInput
from sage_sanctum.errors import InputValidationError


class StrictAgent(SageSanctumAgent):
    @property
    def name(self) -> str:
        return "strict-agent"

    @property
    def version(self) -> str:
        return "1.0.0"

    async def run(self, agent_input: AgentInput) -> AgentResult:
        repo = agent_input
        assert isinstance(repo, RepositoryInput)

        py_files = repo.list_files(extensions={".py"})
        if not py_files:
            raise InputValidationError("Repository contains no Python files")

        if len(py_files) > 1000:
            raise InputValidationError(
                f"Repository too large: {len(py_files)} Python files (max 1000)"
            )

        # Proceed with analysis...
        return AgentResult(exit_code=0)
```
