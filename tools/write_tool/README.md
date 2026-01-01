# Write Tool

A writing tool for Open WebUI that calls Claude via OpenRouter to generate prose, documents, emails, and other text content.

## Overview

This tool is designed to be used by an orchestrator model. The orchestrator provides complete writing instructions, and this tool calls Claude to generate the actual content.

## Usage

The orchestrator calls `write(instructions)` with all the context needed:

```
write("Write a formal email to the engineering team about the upcoming deployment.
Key points: deployment scheduled for Friday 3pm, 2-hour maintenance window expected,
all services will be migrated to new infrastructure. Tone: professional but friendly.")
```

The tool will call Claude via OpenRouter and return the generated content.

## Configuration (Valves)

Configure these settings in the Open WebUI admin panel under the tool's settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | (required) |
| `MODEL` | Model to use for writing | `anthropic/claude-sonnet-4` |
| `SYSTEM_PROMPT` | System prompt for the writing model | Expert writer prompt |

## Requirements

- OpenRouter API key
- `aiohttp` (included in Open WebUI)

## How It Works

1. Orchestrator model calls `write(instructions)` with complete writing context
2. Tool sends request to OpenRouter API with the configured model
3. Claude generates the content based on the instructions
4. Generated content is returned to the chat
