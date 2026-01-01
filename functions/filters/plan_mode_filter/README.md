# Plan Mode Filter

Switches to a planning model and injects a system prompt when enabled.

## Features

- Switches model to `gemini.dev` (configurable)
- Injects system prompt directly into messages
- Toggle button in chat UI to enable/disable per conversation

## Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| `MODEL` | `gemini.dev` | Model to use when plan mode is enabled |
| `SYSTEM_PROMPT` | `You are in PLANNING mode.` | System prompt to inject |
| `priority` | `0` | Filter execution order |

## Usage

1. Install the filter in Open WebUI (Admin > Functions)
2. Enable the filter for your model
3. Set `SYSTEM_PROMPT` in valves to your full planning system prompt
4. Toggle on → switches to gemini.dev with injected system prompt
