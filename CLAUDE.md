# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Open-WebUI Developer Toolkit** is a collection of extensions (pipes, filters, tools, and actions) that enhance [Open WebUI](https://github.com/open-webui/open-webui), a self-hosted AI interface. This repository provides production-ready extensions and documentation for building custom Open WebUI components.

**Key principle**: Extensions here are self-contained, copy-pastable Python modules designed to work directly within Open WebUI's plugin system.

## Repository Architecture

### Directory Structure

```
functions/
├── pipes/           # Message transformation pipelines (call APIs, generate content)
├── filters/         # Pre/post processors (modify requests/responses)
└── actions/         # UI actions (buttons in message toolbar)

tools/               # Standalone tool plugins (function-calling capabilities)

external/
├── open-webui/      # Read-only upstream source snapshot (for API reference)
├── openwebui-docs/  # Official documentation clone
└── *_GUIDE.md       # Backend utility documentation

.tests/              # Test suite mirroring extension structure
docs/                # Internal development notes
system_prompts/      # Shared prompt templates
```

### Extension Types

**Pipes** (`functions/pipes/`):
- Transform or generate chat messages
- Have a `pipe()` method that receives request body and returns response
- Examples: API integrations, manifolds, content generators

**Filters** (`functions/filters/`):
- Modify messages before/after processing
- Have `inlet()` (pre-processing) and/or `outlet()` (post-processing) methods
- Examples: toggle filters, validators, sanitizers

**Actions** (`functions/actions/`):
- Add buttons to message UI
- Have an `action()` method triggered by button clicks
- Example: Switch model, regenerate with different settings

**Tools** (`tools/`):
- Provide function-calling capabilities to models
- Define methods that models can invoke during conversation
- Must have type hints for JSON schema generation
- Example: writer_assistant delegates writing to specialized models

## Development Commands

### Setup
```bash
python -m pip install -e .[dev]
```

### Linting
```bash
nox -s lint                    # Run Ruff on all code
ruff check --fix functions     # Auto-fix specific directory
```

### Testing
```bash
nox -s tests                             # Full test suite with coverage
pytest -vv --cov=functions --cov=tools   # Explicit pytest invocation
pytest .tests/pipes/test_specific.py     # Run single test file
```

### Pre-commit
```bash
pre-commit install          # Set up git hooks
pre-commit run --all-files  # Manual run
```

## Key Development Concepts

### Open WebUI Integration Points

**Extension Lifecycle**:
1. User installs extension via Open WebUI admin interface
2. Open WebUI loads module and inspects class structure
3. Extension receives requests with specific parameters (`__user__`, `__event_emitter__`, `__metadata__`, etc.)
4. Extension processes and returns data to Open WebUI

**Critical Parameters** (automatically injected by Open WebUI):
- `__event_emitter__`: Send UI updates (status, messages, citations)
- `__user__`: User info and permissions
- `__metadata__`: Chat/model/request metadata
- `__request__`: HTTP request object (for loading other modules)
- `__event_call__`: User interaction prompts

### Valves Configuration

All extensions support `Valves` (admin settings) and optionally `UserValves` (per-user settings):

```python
class Valves(BaseModel):
    API_KEY: str = Field(default="", description="Your API key")
    DEBUG: bool = Field(default=False, description="Enable debug logging")
```

### Event Emitters

Extensions can emit real-time UI updates:

```python
# Status updates
await __event_emitter__({
    "type": "status",
    "data": {"description": "Processing...", "done": False}
})

# Citations
await __event_emitter__({
    "type": "citation",
    "data": {"document": [...], "metadata": [...], "source": {...}}
})
```

**Important**: Event behavior differs between Default and Native function calling modes. Status and citation events work in both modes; message events only work reliably in Default mode.

### Manifolds

Manifolds are special pipes that provide multiple models under one namespace:

```python
def models(self) -> List[dict]:
    """Return list of available models"""
    return [{"id": "provider.model-name", "name": "Display Name"}]

async def pipe(self, body: dict, __user__: dict = None, ...) -> dict:
    """Handle chat completion for any model in this manifold"""
    model_id = body.get("model")  # Will be "manifold_id.model-name"
    # Process request and return response
```

Example: `openai_responses_manifold` provides access to OpenAI's Responses API models.

### Actions vs Tools

**Actions**:
- UI buttons that appear in chat interface
- Triggered by user clicks
- Can regenerate messages, switch models, etc.
- Return `{"content": "..."}` to replace message

**Tools**:
- Function-calling capabilities for models
- Invoked by the model during conversation
- Must have clear type hints for schema generation
- Return plain strings/data to the model

## Testing Guidelines

### Test Structure
Tests in `.tests/` mirror the structure of `functions/` and `tools/`:

```
.tests/
├── pipes/
│   └── test_openai_responses_manifold.py
├── filters/
│   └── test_extended_thinking_filter.py
└── tools/
    └── test_writer_assistant.py
```

### Pytest Configuration
- Uses `pytest-asyncio` for async tests (mode: `auto`)
- Coverage tracked for `functions/` and `tools/`
- PYTHONPATH includes `external/open-webui/backend` for Open WebUI imports

### Writing Tests
```python
import pytest
from functions.pipes.my_pipe.my_pipe import Pipe

@pytest.mark.asyncio
async def test_pipe_basic_flow():
    """Test description following pattern: test_<component>_<scenario>"""
    pipe = Pipe()
    result = await pipe.pipe(body={"messages": [...]}, __user__={})
    assert result["choices"][0]["message"]["content"]
```

## External Resources Usage

### Read-Only Upstream Reference
`external/open-webui/` contains Open WebUI source code for reference:

```python
# Import Open WebUI utilities for your extension
from open_webui.utils.plugin import get_function_module_from_cache
from open_webui.models.functions import Functions
```

**DO NOT** modify files in `external/open-webui/` - it's a snapshot for API reference only.

### Documentation
- `external/openwebui-docs/docs/` - Official documentation (update with `git pull`)
- `external/*_GUIDE.md` - Backend utility documentation
- Online: https://docs.openwebui.com

## Common Development Patterns

### Loading Other Extensions
Actions and tools can invoke manifolds:

```python
from open_webui.utils.plugin import get_function_module_from_cache
from open_webui.models.functions import Functions

# Load manifold
manifold, _, _ = get_function_module_from_cache(__request__, "manifold_id")

# Load its configuration
valves = Functions.get_function_valves_by_id("manifold_id")
manifold.valves = manifold.Valves(**(valves or {}))

# Call it
result = await manifold.pipe(body=modified_body, __user__=__user__, ...)
```

### Handling Streaming Responses

```python
import inspect

result = await some_async_call()

if inspect.isasyncgen(result):
    # Streaming response
    async for chunk in result:
        # Process chunk
        await __event_emitter__({"type": "message", "data": {"content": chunk}})
else:
    # Non-streaming response
    return result
```

### API Key Security
Store sensitive values in environment variables, not code:

```python
import os

class Valves(BaseModel):
    API_KEY: str = Field(
        default=os.getenv("SERVICE_API_KEY", ""),
        description="API key (or set SERVICE_API_KEY env var)"
    )
```

## Branching Model

- **`main`**: Stable, production-ready code
- **`alpha-preview`**: Release candidates (2-3 week testing period)
- **`development`**: Active development (may be unstable)

Workflow: `development` → `alpha-preview` → `main`

## Code Style

- Python 3.11+ with type hints
- 4-space indentation, 100-char line limit
- Ruff enforces all rules with auto-fix enabled
- Use `snake_case` for functions/variables, `PascalCase` for classes
- Async functions use descriptive verb names: `fetch_data()`, `process_request()`

## Project-Specific Notes

### Writer Assistant Tool
The `writer_assistant` tool demonstrates model-to-model delegation:
- Model calls `write_content(prompt="...")` when user requests writing
- Tool invokes OpenRouter API with specified model
- Returns clean content (docstring instructs model to present verbatim)
- Uses event emitters for "Writing..." / "Writing complete." status

### OpenAI Responses Manifold
Complex manifold integrating OpenAI's Responses API:
- Handles reasoning effort, text verbosity, service tier
- Compatible with Extended Thinking, Verbose, and Priority filters
- Demonstrates advanced manifold patterns and filter interaction

### Filter Interaction
Filters can modify requests before they reach pipes:
- Extended Thinking Filter adds `reasoning_effort` parameter
- Verbose Filter adds `text.verbosity` parameter
- Priority Filter adds `service_tier` parameter
- Filters run in order: inlet chain → pipe → outlet chain
