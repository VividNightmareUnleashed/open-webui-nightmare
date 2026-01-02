# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains extensions (pipes, filters, tools) for [Open WebUI](https://github.com/open-webui/open-webui). Extensions are self-contained Python modules that can be copy-pasted into an Open WebUI instance via the Functions admin panel.

Python 3.11+ required.

## Commands

```bash
# Run all checks (lint + tests)
nox

# Lint only
ruff check functions tools .tests .scripts

# Run tests with coverage
pytest -vv --cov=functions --cov-report=term-missing

# Run a single test file
pytest .tests/test_openai_responses_manifold.py -v

# Pre-commit hooks (ruff + pytest)
pre-commit run --all-files
```

## Architecture

### Extension Types

**Pipes** (`functions/pipes/`) - Transform or generate chat messages. Can call external APIs and stream responses.
```python
class Pipe:
    class Valves(BaseModel):
        # Configurable settings persisted in DB
        SOME_OPTION: str = "default"

    async def pipe(self, body, __user__, __request__, __event_emitter__, ...) -> AsyncGenerator[str, None]:
        # Process request, yield response chunks
```

**Filters** (`functions/filters/`) - Intercept/modify messages before and after pipes.
```python
class Filter:
    class Valves(BaseModel):
        priority: int = 0  # Controls execution order

    async def inlet(self, body, __event_emitter__):
        # Modify incoming request
        return body

    async def outlet(self, body):
        # Modify outgoing response (no event_emitter here)
        return body
```

**Tools** (`tools/`) - Standalone plugins that add new capabilities to the assistant.

### Key Arguments

Pipes and filters receive these injected arguments (declare only what you need):
- `body` - Chat request payload (messages, model, stream options)
- `__user__` - Current user info (id, email, name, role)
- `__request__` - FastAPI Request object
- `__event_emitter__` - Async callback for emitting citations, status, etc.
- `__files__` - Uploaded file metadata
- `__metadata__` - Chat/session IDs, features, variables, model config
- `__tools__` - Available tool definitions with callables

### Extension File Structure

Each extension lives in its own folder:
```
functions/pipes/my_pipe/
├── my_pipe.py      # Main code with docstring metadata
├── README.md       # Usage documentation
└── CHANGELOG.md    # Version history
```

Required docstring format:
```python
"""
title: My Pipe
id: my_pipe
version: 0.1.0
description: What this pipe does
"""
```

### Testing

Tests in `.tests/` use stubs for `open_webui` modules (see `conftest.py`). This allows testing pipes/filters without a full Open WebUI installation.

## Key Documentation

- `docs/pipe_input.md` - Detailed reference for all pipe arguments
- `docs/events.md` - Event emitter system (citations, status, actions)
- `docs/message-execution-path.md` - How messages flow through the system

## Upstream Reference (external/)

The `external/` directory contains a complete read-only copy of the Open WebUI repository. **Do not modify files here** - it exists for reference and testing compatibility.

### Quick Reference Guides (`external/*.md`)

| Guide | Key Insight |
|-------|-------------|
| `MIDDLEWARE_GUIDE.md` | Chat pipeline stages: inlet filters → feature handlers → file retrieval → model execution → outlet filters. Extensions emit events via `__event_emitter__` for real-time browser feedback. |
| `FILTER_GUIDE.md` | Three interception points: `inlet()` (pre-request), `outlet()` (post-response), `stream()` (token-by-token). Priority-based execution. Set `file_handler=True` to claim uploaded files. |
| `TOOLS_GUIDE.md` | Tools auto-convert to OpenAI specs from docstrings/type hints. Use `:param` annotations in docstrings for automatic parameter descriptions. |
| `TASK_GUIDE.md` | Template placeholders (`{{CURRENT_DATE}}`, `{{USER_NAME}}`, `{{prompt:start:N}}`) for background tasks like title generation and RAG queries. |
| `PAYLOAD_GUIDE.md` | Payload normalization between OpenAI/Ollama formats. In-place mutation for system prompts, template variables, and message format translation. |

### Source Code (`external/open-webui/backend/open_webui/`)

**Core Execution:**
- `main.py` - FastAPI app, route registration, middleware, Socket.IO integration
- `functions.py` - `generate_function_chat_completion()` - main pipe execution, streaming, event injection
- `tasks.py` - Background task management

**Extension Loading (`utils/`):**
- `plugin.py` - `load_function_module_by_id()`, `extract_frontmatter()`, module caching
- `filter.py` - `get_sorted_filter_ids()`, `process_filter_functions()` for inlet/outlet/stream
- `tools.py` - `get_tools()`, OpenAI-compatible function calling spec conversion
- `misc.py` - `add_or_update_system_message()`, `get_last_user_message()`, message templates

**Database Models (`models/`):**
- `functions.py` - `Functions.get_functions_by_type()`, `get_function_valves_by_id()`, `get_user_valves_by_id_and_user_id()`
- `tools.py` - `Tools.get_tool_by_id()`, access control support
- `chats.py`, `messages.py`, `users.py` - Chat history, message records, user accounts

**WebSocket (`socket/`):**
- `main.py` - `get_event_emitter()`, `get_event_call()` for real-time status/citation events

**API Routers (`routers/`):**
- `functions.py` - Function CRUD, valve management, load from URL
- `tools.py` - Tool CRUD endpoints
- `chats.py` - Chat endpoints
- `openai.py`, `ollama.py` - LLM API integrations

**RAG/Retrieval (`retrieval/`):**
- `vector/dbs/` - Vector DB implementations (Chroma, Qdrant, pgvector, Milvus, Pinecone, etc.)
- `web/` - Web search providers (Brave, DuckDuckGo, Google PSE, Tavily, etc.)
- `loaders/` - Document loaders (PDF, YouTube, external docs)

## Additional Resources

- `suurt8ll_functions/` - Additional extension collection (separate repo)
- `docs/openwebui/` - Official Open WebUI documentation mirror

## Versioning

Follow semantic versioning. Update `version` in docstring and add entry to `CHANGELOG.md`:
```
## [x.y.z] - YYYY-MM-DD
- What changed
```
