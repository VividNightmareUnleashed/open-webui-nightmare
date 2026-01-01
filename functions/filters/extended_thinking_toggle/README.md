# Extended Thinking Toggle

A simple toggle filter that enables Claude's extended thinking mode for Anthropic Manifold models.

## Features

- Clock icon toggle in chat interface
- Enables/disables Claude's extended thinking on demand
- Works with all Claude models that support extended thinking

## Installation

1. Add this filter to your Open WebUI instance
2. Enable the filter for your Anthropic models

## Usage

When enabled (toggle ON):
- Claude will show its reasoning process in `<think>` tags
- Useful for complex problem-solving, math, coding tasks

When disabled (toggle OFF):
- Normal response without visible thinking
- Faster responses for simple queries

## Requirements

- Anthropic Manifold pipe must be installed
- Model must support extended thinking (most Claude 4.x models)

## Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MANIFOLD_PREFIX` | `anthropic.` | Model ID prefix to match |

## How It Works

This filter sets `metadata["features"]["anthropic"]["thinking"] = True` when the toggle is ON. The Anthropic Manifold pipe reads this flag and enables extended thinking for the request.

## Related

- **Anthropic Manifold** - The main pipe that handles Claude API calls
- **Anthropic Manifold Companion** - Intercepts web search and code interpreter toggles
