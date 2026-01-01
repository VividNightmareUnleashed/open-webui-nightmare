# Anthropic Manifold Companion

Companion filter for the Anthropic Manifold pipe. Provides an Extended Thinking toggle and intercepts OpenWebUI's native web search and code interpreter toggles to use Claude's native tools instead.

## Features

- **Extended Thinking Toggle** (clock icon) - Enable Claude's extended thinking mode for any supported model
- **Web Search Interception** - Routes to Claude's native web search instead of OpenWebUI's
- **Code Interpreter Interception** - Routes to Claude's native code execution instead of OpenWebUI's
- **Files API / RAG Bypass** - Uploads files directly to Anthropic's Files API instead of OpenWebUI's RAG

## Why This Filter Is Needed

OpenWebUI's middleware processes `features["web_search"]` and `features["code_interpreter"]` toggles BEFORE pipes run. Without this filter:

1. OpenWebUI would run its own web search handler
2. OpenWebUI would run its own code interpreter handler
3. Then Claude would also try to use its native tools
4. This causes duplicate processing and status spam

This filter runs in the `inlet` phase (before middleware) and:
1. Provides a thinking toggle (clock icon) for extended thinking
2. Detects when toggles are enabled
3. Disables OpenWebUI's native handling
4. Signals the Anthropic Manifold to use Claude's native tools

## Installation

1. Install this filter in OpenWebUI (Admin > Functions > Add Filter)
2. The filter will automatically detect Anthropic manifold models
3. No additional configuration required

## How It Works

When you enable the web search or code interpreter toggle in the chat interface:

**Without companion filter:**
```
User enables toggle → OpenWebUI middleware runs its handler → Pipe runs
                      ↓
                      OpenWebUI web search/code interpreter runs
                      (Claude's native tools are not used)
```

**With companion filter:**
```
User enables toggle → Filter intercepts → Middleware skips → Pipe runs
                      ↓                                      ↓
                      features["web_search"] = False         anthropic_features["web_search"] = True
                      (OpenWebUI native disabled)            (Claude native enabled)
```

## Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MANIFOLD_PREFIX` | `anthropic.` | Model ID prefix to match |
| `BYPASS_BACKEND_RAG` | `true` | Bypass OpenWebUI's RAG and send files directly to Anthropic Files API |

## Supported Toggles

| Toggle | Icon | Claude Feature |
|--------|------|----------------|
| Extended Thinking | 🕐 (clock) | Extended thinking with `<think>` tags |
| Web Search | 🔍 | `web_search_20250305` |
| Code Interpreter | 💻 | `code_execution_20250825` |

## Requirements

- **Anthropic Manifold** pipe must also be installed
- OpenWebUI 0.6.10+

## License

MIT
