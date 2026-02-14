"""Tests for input/output types."""

import json
from pathlib import Path

import pytest

from sage_sanctum.errors import InputValidationError, PathTraversalError
from sage_sanctum.io.inputs import RepositoryInput
from sage_sanctum.io.outputs import Finding, Location, SarifOutput, TokenUsage


class TestRepositoryInput:
    def test_valid_directory(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        ri = RepositoryInput(path=repo)
        ri.validate()  # Should not raise

    def test_nonexistent_path(self, tmp_path):
        ri = RepositoryInput(path=tmp_path / "missing")
        with pytest.raises(InputValidationError, match="does not exist"):
            ri.validate()

    def test_file_not_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        ri = RepositoryInput(path=f)
        with pytest.raises(InputValidationError, match="not a directory"):
            ri.validate()

    def test_path_traversal(self, tmp_path):
        ri = RepositoryInput(path=tmp_path / ".." / "etc" / "passwd")
        with pytest.raises(PathTraversalError, match="traversal"):
            ri.validate()

    def test_path_with_dotdot_in_name_not_flagged(self, tmp_path):
        """Directory named '..foo' should NOT trigger path traversal."""
        d = tmp_path / "..foo"
        d.mkdir()
        ri = RepositoryInput(path=d)
        ri.validate()  # Should not raise

    def test_path_traversal_middle_component(self, tmp_path):
        """Path with '..' in the middle should be caught."""
        ri = RepositoryInput(path=tmp_path / "a" / ".." / "b")
        with pytest.raises(PathTraversalError, match="traversal"):
            ri.validate()

    def test_list_files(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("print('hello')")
        (repo / "test.js").write_text("console.log('hello')")
        (repo / "sub").mkdir()
        (repo / "sub" / "lib.py").write_text("pass")

        ri = RepositoryInput(path=repo)
        all_files = ri.list_files()
        assert len(all_files) == 3

    def test_list_files_with_extensions(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("print('hello')")
        (repo / "test.js").write_text("console.log('hello')")

        ri = RepositoryInput(path=repo)
        py_files = ri.list_files(extensions={".py"})
        assert len(py_files) == 1
        assert py_files[0] == Path("main.py")

    def test_from_environment(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setenv("REPO_PATH", str(repo))
        monkeypatch.setenv("REPO_REF", "main")
        monkeypatch.setenv("REPO_URL", "https://github.com/test/repo")

        ri = RepositoryInput.from_environment()
        assert ri.path == Path(str(repo))
        assert ri.ref == "main"
        assert ri.url == "https://github.com/test/repo"

    def test_from_environment_missing(self, monkeypatch):
        monkeypatch.delenv("REPO_PATH", raising=False)
        with pytest.raises(InputValidationError, match="REPO_PATH"):
            RepositoryInput.from_environment()

    def test_io_type(self, tmp_path):
        ri = RepositoryInput(path=tmp_path)
        assert ri.io_type == "repository"


class TestSarifOutput:
    def test_empty_sarif(self):
        output = SarifOutput(
            tool_name="test-tool",
            tool_version="1.0.0",
        )
        sarif = output.to_dict()
        assert sarif["version"] == "2.1.0"
        assert sarif["runs"][0]["tool"]["driver"]["name"] == "test-tool"
        assert sarif["runs"][0]["results"] == []

    def test_sarif_with_findings(self):
        output = SarifOutput(
            tool_name="test-tool",
            tool_version="1.0.0",
            findings=[
                Finding(
                    id="SEC-001",
                    title="SQL Injection",
                    description="Unsanitized user input in SQL query",
                    severity="high",
                    location=Location(file="app.py", start_line=42),
                    cwe="89",
                    remediation="Use parameterized queries",
                ),
                Finding(
                    id="SEC-002",
                    title="XSS",
                    description="Reflected XSS in template",
                    severity="medium",
                    location=Location(file="views.py", start_line=15, end_line=20),
                ),
            ],
        )
        sarif = output.to_dict()
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        results = sarif["runs"][0]["results"]

        assert len(rules) == 2
        assert len(results) == 2

        # Check rule
        assert rules[0]["id"] == "SEC-001"
        assert rules[0]["properties"]["security-severity"] == "7.0"
        assert "CWE-89" in rules[0]["properties"]["tags"]

        # Check result
        assert results[0]["ruleId"] == "SEC-001"
        assert results[0]["level"] == "error"
        assert results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "app.py"
        assert results[0]["locations"][0]["physicalLocation"]["region"]["startLine"] == 42

    def test_sarif_severity_levels(self):
        findings = [
            Finding(id="C", title="t", description="d", severity="critical", location=Location(file="f")),
            Finding(id="H", title="t", description="d", severity="high", location=Location(file="f")),
            Finding(id="M", title="t", description="d", severity="medium", location=Location(file="f")),
            Finding(id="L", title="t", description="d", severity="low", location=Location(file="f")),
            Finding(id="N", title="t", description="d", severity="note", location=Location(file="f")),
        ]
        output = SarifOutput(tool_name="t", tool_version="1", findings=findings)
        sarif = output.to_dict()
        levels = [r["level"] for r in sarif["runs"][0]["results"]]
        assert levels == ["error", "error", "warning", "note", "note"]

    def test_write_sarif(self, tmp_path):
        output = SarifOutput(
            tool_name="test-tool",
            tool_version="1.0.0",
            findings=[
                Finding(
                    id="SEC-001",
                    title="Test",
                    description="Test finding",
                    severity="high",
                    location=Location(file="test.py", start_line=1),
                ),
            ],
        )
        files = output.write(tmp_path / "output")
        assert "results.sarif" in files

        sarif_file = tmp_path / "output" / "results.sarif"
        assert sarif_file.exists()

        data = json.loads(sarif_file.read_text())
        assert data["version"] == "2.1.0"

    def test_io_type(self):
        output = SarifOutput(tool_name="t", tool_version="1")
        assert output.io_type == "sarif"


class TestTokenUsage:
    def test_default_values(self):
        usage = TokenUsage(model="gpt-4o")
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_with_values(self):
        usage = TokenUsage(model="gpt-4o", prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150


class TestLocation:
    def test_minimal(self):
        loc = Location(file="test.py")
        assert loc.file == "test.py"
        assert loc.start_line == 0

    def test_full(self):
        loc = Location(file="test.py", start_line=10, end_line=20, start_column=1, end_column=50)
        assert loc.start_line == 10
        assert loc.end_line == 20
