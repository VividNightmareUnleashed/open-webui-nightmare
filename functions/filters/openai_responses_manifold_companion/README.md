# OpenAI Responses Manifold Companion

Companion filter for the OpenAI Responses Manifold pipe. Intercepts OpenWebUI's native feature toggles so the manifold can enable OpenAI-native tools instead.

## Why this filter is needed

OpenWebUI processes `features["web_search"]` and `features["code_interpreter"]` **before** pipes run. Without this filter, enabling those toggles triggers OpenWebUI's built-in handlers instead of (or in addition to) OpenAI Responses tools.

## What it does

- **Web Search**: disables OpenWebUI `web_search` handling and signals the manifold to add OpenAI `web_search_preview`.
- **Code Interpreter**: disables OpenWebUI `code_interpreter` prompt injection and signals the manifold to add OpenAI `code_interpreter`.
- **Optional RAG bypass for files**: when OpenAI file tools are enabled, stashes the original `body["files"]` into `__metadata__["features"]["openai_responses"]["files"]` and clears `body["files"]` so OpenWebUI RAG does not run.

## Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MANIFOLD_PREFIX` | `openai_responses.` | Model ID prefix to match |
| `BYPASS_BACKEND_RAG` | `true` | When OpenAI file tools are enabled, bypass OpenWebUI RAG and let the manifold upload files to OpenAI |

## Requirements

- OpenAI Responses Manifold pipe installed
- OpenWebUI 0.6.10+

## License

MIT

