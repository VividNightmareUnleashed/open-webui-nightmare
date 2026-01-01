# Write Mode Filter

Switches to a writing model and injects a system prompt when enabled.

## Features

- Switches model to `claude` (configurable)
- Injects system prompt directly into messages
- Optional prefix for user messages
- Toggle button in chat UI to enable/disable per conversation

## Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| `MODEL` | `claude` | Model to use when write mode is enabled |
| `SYSTEM_PROMPT` | `You are in EDITOR mode.` | System prompt to inject |
| `USER_PREFIX` | (empty) | Optional prefix to add to user message |
| `priority` | `0` | Filter execution order |

## Usage

1. Install the filter in Open WebUI (Admin > Functions)
2. Enable the filter for your model
3. Set `SYSTEM_PROMPT` in valves to your full writing system prompt
4. Toggle on → switches to claude with injected system prompt
