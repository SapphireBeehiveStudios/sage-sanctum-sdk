"""Agent input types."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..errors import InputValidationError, PathTraversalError


class AgentInput(ABC):
    """Base class for agent inputs."""

    @property
    @abstractmethod
    def io_type(self) -> str:
        """Input type identifier."""

    @abstractmethod
    def validate(self) -> None:
        """Validate the input.

        Raises:
            InputValidationError: If the input is invalid.
        """


@dataclass
class RepositoryInput(AgentInput):
    """Input representing a cloned repository to analyze.

    Attributes:
        path: Absolute path to the repository root
        ref: Git ref (branch, tag, commit) being analyzed
        url: Remote URL of the repository
    """

    path: Path
    ref: str = ""
    url: str = ""

    @property
    def io_type(self) -> str:
        return "repository"

    def validate(self) -> None:
        """Validate the repository input.

        Raises:
            InputValidationError: If the path doesn't exist or isn't a directory.
            PathTraversalError: If the path contains traversal attempts.
        """
        # Check for path traversal
        resolved = self.path.resolve()
        if ".." in str(self.path):
            raise PathTraversalError(
                f"Path traversal detected in repository path: {self.path}"
            )

        if not resolved.exists():
            raise InputValidationError(
                f"Repository path does not exist: {resolved}"
            )

        if not resolved.is_dir():
            raise InputValidationError(
                f"Repository path is not a directory: {resolved}"
            )

    def list_files(self, extensions: set[str] | None = None) -> list[Path]:
        """List files in the repository.

        Args:
            extensions: Optional set of extensions to filter by (e.g., {'.py', '.js'})

        Returns:
            List of file paths relative to the repository root.
        """
        files = []
        for root, _, filenames in os.walk(self.path):
            root_path = Path(root)
            for name in filenames:
                file_path = root_path / name
                if extensions and file_path.suffix not in extensions:
                    continue
                try:
                    files.append(file_path.relative_to(self.path))
                except ValueError:
                    continue
        return sorted(files)

    @classmethod
    def from_environment(cls) -> RepositoryInput:
        """Create from environment variables.

        Expected env vars:
        - REPO_PATH: Path to the repository
        - REPO_REF: Git ref (optional)
        - REPO_URL: Remote URL (optional)
        """
        repo_path = os.environ.get("REPO_PATH", "")
        if not repo_path:
            raise InputValidationError("REPO_PATH environment variable not set")

        return cls(
            path=Path(repo_path),
            ref=os.environ.get("REPO_REF", ""),
            url=os.environ.get("REPO_URL", ""),
        )
