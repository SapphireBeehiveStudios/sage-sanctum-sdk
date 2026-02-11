"""Agent output types including SARIF format."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import OutputWriteError


class AgentOutput(ABC):
    """Base class for agent outputs."""

    @property
    @abstractmethod
    def io_type(self) -> str:
        """Output type identifier."""

    @abstractmethod
    def write(self, output_dir: Path) -> list[str]:
        """Write output files to the output directory.

        Returns:
            List of filenames written.

        Raises:
            OutputWriteError: If writing fails.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize output to a dictionary."""


@dataclass(frozen=True)
class Location:
    """Source code location for a finding.

    Attributes:
        file: File path relative to the repository root.
        start_line: 1-based starting line number.
        end_line: 1-based ending line number (``0`` if single-line).
        start_column: 1-based starting column number (``0`` if not specified).
        end_column: 1-based ending column number (``0`` if not specified).
    """

    file: str
    start_line: int = 0
    end_line: int = 0
    start_column: int = 0
    end_column: int = 0


@dataclass(frozen=True)
class TokenUsage:
    """Token usage tracking for an LLM call.

    Attributes:
        model: Model identifier (e.g. ``"gpt-4o"``).
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens consumed (prompt + completion).
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Finding:
    """A single security or code quality finding.

    Attributes:
        id: Rule identifier (e.g. ``"SQL-001"``). Used as the SARIF ``ruleId``.
        title: Short human-readable title.
        description: Detailed description of the finding.
        severity: Severity level — one of ``"critical"``, ``"high"``,
            ``"medium"``, ``"low"``, or ``"note"``.
        location: Source code location where the finding was identified.
        cwe: CWE identifier (e.g. ``"CWE-89"``). Included as a SARIF tag.
        remediation: Suggested fix or mitigation.
        confidence: Confidence level — ``"high"``, ``"medium"``, or ``"low"``.
        metadata: Arbitrary key-value pairs for additional context.
    """

    id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low, note
    location: Location
    cwe: str = ""
    remediation: str = ""
    confidence: str = "medium"
    metadata: dict = field(default_factory=dict)


@dataclass
class SarifOutput(AgentOutput):
    """SARIF 2.1.0 formatted output.

    The standard format for static analysis results, compatible with
    GitHub Code Scanning and other SARIF consumers.

    Attributes:
        tool_name: Name of the tool that produced the findings.
        tool_version: Version of the tool.
        findings: List of security or quality findings.
        token_usage: LLM token usage records for cost tracking.
    """

    tool_name: str
    tool_version: str
    findings: list[Finding] = field(default_factory=list)
    token_usage: list[TokenUsage] = field(default_factory=list)

    @property
    def io_type(self) -> str:
        return "sarif"

    def write(self, output_dir: Path) -> list[str]:
        """Write SARIF report to the output directory.

        Creates the directory if it doesn't exist and writes
        ``results.sarif`` with JSON-formatted SARIF 2.1.0 content.

        Args:
            output_dir: Directory to write the SARIF file into.

        Returns:
            List containing ``"results.sarif"``.

        Raises:
            OutputWriteError: If writing to disk fails.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        sarif_path = output_dir / "results.sarif"

        try:
            sarif_data = self._to_sarif()
            sarif_path.write_text(json.dumps(sarif_data, indent=2))
        except Exception as e:
            raise OutputWriteError(f"Failed to write SARIF: {e}") from e

        return ["results.sarif"]

    def to_dict(self) -> dict:
        """Serialize to a SARIF 2.1.0 dictionary."""
        return self._to_sarif()

    def _to_sarif(self) -> dict:
        """Convert to SARIF 2.1.0 format."""
        # Build rules from unique finding IDs
        rules_map: dict[str, dict] = {}
        results = []

        severity_to_level = {
            "critical": "error",
            "high": "error",
            "medium": "warning",
            "low": "note",
            "note": "note",
        }

        severity_to_score = {
            "critical": 9.0,
            "high": 7.0,
            "medium": 4.0,
            "low": 1.0,
            "note": 0.0,
        }

        for finding in self.findings:
            # Add rule if not seen
            if finding.id not in rules_map:
                rules_map[finding.id] = {
                    "id": finding.id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                    "properties": {
                        "security-severity": str(
                            severity_to_score.get(finding.severity, 4.0)
                        ),
                    },
                }
                if finding.cwe:
                    rules_map[finding.id]["properties"]["tags"] = [
                        f"CWE-{finding.cwe}" if not finding.cwe.startswith("CWE-") else finding.cwe,
                        "security",
                    ]

            # Build result
            result = {
                "ruleId": finding.id,
                "level": severity_to_level.get(finding.severity, "warning"),
                "message": {"text": finding.description},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": finding.location.file},
                            "region": {
                                "startLine": max(finding.location.start_line, 1),
                            },
                        }
                    }
                ],
            }

            if finding.location.end_line:
                result["locations"][0]["physicalLocation"]["region"]["endLine"] = (
                    finding.location.end_line
                )
            if finding.location.start_column:
                result["locations"][0]["physicalLocation"]["region"]["startColumn"] = (
                    finding.location.start_column
                )
            if finding.location.end_column:
                result["locations"][0]["physicalLocation"]["region"]["endColumn"] = (
                    finding.location.end_column
                )

            if finding.remediation:
                result["fixes"] = [
                    {"description": {"text": finding.remediation}}
                ]

            results.append(result)

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self.tool_name,
                            "version": self.tool_version,
                            "rules": list(rules_map.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }
