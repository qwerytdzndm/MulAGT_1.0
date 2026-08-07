# Mulagt Core

Mulagt Core keeps only the reusable multi-agent coding workflow and repository
RAG pieces from the original project.

Retained flow:

```text
repository + issue
-> repository RAG indexing
-> evidence retrieval
-> Planner Agent
-> Coder Agent
-> Reviewer Agent
-> human approval checkpoint
-> patch application
-> allowlisted tests
-> verification / rollback / report
```
## Install

Use Python 3.10:

```powershell
cd C:\workplace\mul-main
python -m pip install -e .
```

For rich PDF/Office parsing through Docling:

```powershell
python -m pip install -e ".[documents]"
```

For BGE-M3 embeddings and reranking:

```powershell
python -m pip install -e ".[rag-models]"
```

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

For deterministic local development and tests:

```dotenv
MUL_LLM_MODE=mock
MUL_RAG_EMBEDDING_PROVIDER=deterministic
MUL_RAG_RERANKER_PROVIDER=deterministic
```

For production-grade retrieval, install `.[rag-models]` and set:

```dotenv
MUL_RAG_EMBEDDING_PROVIDER=bge-m3
MUL_RAG_RERANKER_PROVIDER=bge
```

For a real DeepSeek-backed run:

```dotenv
MUL_LLM_MODE=deepseek
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

`MUL_WORKSPACE_ROOTS` limits which repositories may be inspected or
modified. On Windows, separate multiple roots with semicolons.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Three-Agent Code Generation

Run only the RAG + Planner + Coder + Reviewer path, without approval, file
write, or tests:

DeepSeek one-file startup:

```powershell
python start_three_agents.py `
  --mul-path examples\buggy_calculator `
  --instruction "Fix safe_divide so zero divisor raises ValueError."
```

When `--mul-path` or `--instruction` is omitted, the script asks for it
interactively. It writes `result.json`, `transcript.md`, and `proposal.patch`
under `.mulagt/three_agent_runs/<timestamp>/`.

```powershell
python -m mulagt.agent_interface `
  --mul-path examples\buggy_calculator `
  --issue "Fix safe_divide so zero divisor raises ValueError." `
  --llm-mode mock `
  --transcript transcript.json
```

After editable install, the console script is also available:

```powershell
mulagt-three-agents `
  --mul-path examples\buggy_calculator `
  --issue "Fix safe_divide so zero divisor raises ValueError." `
  --llm-mode mock `
  --transcript transcript.json
```

Programmatic call:

```python
from mulagt.agent_interface import run_three_agent_code_generation
from mulagt.models import IssueRequest

result = run_three_agent_code_generation(
    IssueRequest(
        mul_path="examples/buggy_calculator",
        issue="Fix safe_divide so zero divisor raises ValueError.",
        llm_mode="mock",
    )
)

print(result["review"])
print(result["transcript"])
```

The transcript contains each model-facing system prompt, compressed user payload,
assistant JSON response, schema validation result, and runtime step markers.

RAG retrieval evaluation:

```powershell
python scripts\evaluate_retrieval.py
```

Generated or modified repository tests are disabled by default because they run
untrusted code on the host. Enable them only for a trusted local repository:

```dotenv
MUL_ALLOW_HOST_TEST_EXECUTION=1
```
