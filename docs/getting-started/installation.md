# Installation

## Requirements

- Python 3.11 or later
- pip (or any PEP 517 compatible installer)

## Install from PyPI

```bash
pip install sage-sanctum-sdk
```

## Install from Source

```bash
git clone https://github.com/SapphireBeehiveStudios/sage-sanctum-sdk.git
cd sage-sanctum-sdk
pip install -e .
```

## Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `pytest-asyncio`, and `pytest-cov` for running the test suite.

## Dependencies

The SDK installs the following dependencies automatically:

| Package | Purpose |
|---------|---------|
| `pydantic>=2.0.0` | Data validation and settings |
| `urllib3>=2.0.0` | HTTP client utilities |
| `litellm>=1.30.0` | Multi-provider LLM routing |
| `langchain-core>=0.3.0` | LLM abstraction layer |
| `langchain-litellm>=0.2.0` | LiteLLM integration for LangChain |

## Verify Installation

```python
import sage_sanctum
print(sage_sanctum.__version__)
# 0.1.1
```

You can also verify all key imports work:

```python
from sage_sanctum import (
    AgentContext,
    AgentRunner,
    SageSanctumAgent,
    AgentResult,
    ModelCategory,
    SarifOutput,
    Finding,
    Location,
)
print("All imports successful")
```

## Next Steps

Head to the [Quickstart](quickstart.md) to build your first agent.
