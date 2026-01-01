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
- `external/open-webui/` - Read-only upstream source reference (do not modify)

## Versioning

Follow semantic versioning. Update `version` in docstring and add entry to `CHANGELOG.md`:
```
## [x.y.z] - YYYY-MM-DD
- What changed
```
