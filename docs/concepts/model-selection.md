# Model Selection

The SDK provides a structured approach to model selection through categories, references, and selectors.

## ModelCategory

Agents categorize their LLM usage into four categories:

| Category | Value | Typical Use |
|----------|-------|-------------|
| `TRIAGE` | `"triage"` | Quick classification, routing, filtering |
| `ANALYSIS` | `"analysis"` | Detailed code analysis, vulnerability detection |
| `REASONING` | `"reasoning"` | Complex multi-step reasoning |
| `EMBEDDINGS` | `"embeddings"` | Embedding generation for similarity search |

```python
from sage_sanctum.llm.model_category import ModelCategory

llm = context.create_llm_client(ModelCategory.ANALYSIS)
```

Each category maps to an allowlist of models defined in the Transaction Token. This lets platform operators control cost and capability per use case.

## ModelRef

A `ModelRef` is a canonical reference to a specific model in `provider:model` format:

```python
from sage_sanctum.llm.model_ref import ModelRef

ref = ModelRef.parse("openai:gpt-4o")
ref = ModelRef.parse("anthropic:claude-3-5-sonnet-latest")
ref = ModelRef.parse("google:gemini-2.0-flash")
```

### Auto-Detection

If you omit the provider prefix, the SDK infers it from the model name:

```python
ModelRef.parse("gpt-4o")           # → openai:gpt-4o
ModelRef.parse("claude-3-5-sonnet-latest")  # → anthropic:claude-3-5-sonnet-latest
ModelRef.parse("gemini-2.0-flash")  # → google:gemini-2.0-flash
```

### LiteLLM Formatting

`ModelRef` handles the provider-specific formatting required by LiteLLM:

```python
ref = ModelRef.parse("openai:gpt-4o")
ref.for_litellm  # "gpt-4o"

ref = ModelRef.parse("anthropic:claude-3-5-sonnet-latest")
ref.for_litellm  # "anthropic/claude-3-5-sonnet-latest"

ref = ModelRef.parse("google:gemini-2.0-flash")
ref.for_litellm  # "gemini/gemini-2.0-flash"
```

## ModelSelector

`ModelSelector` resolves categories to concrete models based on the TraT allowlist:

```python
from sage_sanctum.llm.model_selector import ModelSelector

selector = ModelSelector(allowed_models={
    "triage": ["openai:gpt-4o-mini"],
    "analysis": ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-latest"],
    "reasoning": ["openai:o1"],
    "embeddings": ["openai:text-embedding-3-small"],
})

model = selector.select(ModelCategory.ANALYSIS)
# Returns first allowed model: openai:gpt-4o

all_models = selector.select_all(ModelCategory.ANALYSIS)
# Returns all allowed: [openai:gpt-4o, anthropic:claude-3-5-sonnet-latest]

selector.is_allowed(ModelRef.parse("openai:gpt-4o"), ModelCategory.ANALYSIS)
# True
```

### StaticModelSelector

For local development, `StaticModelSelector` returns the same model for every category:

```python
from sage_sanctum.llm.model_selector import StaticModelSelector

selector = StaticModelSelector("openai:gpt-4o")
selector.select(ModelCategory.TRIAGE)      # openai:gpt-4o
selector.select(ModelCategory.REASONING)   # openai:gpt-4o
```

This is what `AgentContext.for_local_development()` uses internally.
