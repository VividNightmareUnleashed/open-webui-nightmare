# Gemini Deep Research Pipe

Google Gemini Deep Research integration for Open WebUI. Provides autonomous web research capabilities using Google's Deep Research agent via the Interactions API.

## Features

- **Autonomous Research**: Deep Research autonomously searches the web, analyzes sources, and synthesizes comprehensive reports
- **Streaming Progress**: Real-time thinking summaries show research progress as it happens
- **Follow-up Questions**: Continue research with follow-up questions on the same context
- **Long-Running Support**: Handles research sessions up to 60 minutes
- **Stream Resume**: Automatically resumes SSE streaming after disconnects/timeouts (uses `last_event_id`)
- **Fallback Polling**: Gracefully degrades to polling if SSE streaming fails
- **No SDK Required**: Uses raw HTTP requests - works with any Open WebUI installation

## Requirements

- `aiohttp` (included in requirements)
- Google AI API Key with Gemini access

## Installation

1. Copy `gemini_deep_research.py` to Open WebUI Functions
2. Configure the `GOOGLE_API_KEY` valve with your API key
3. Optionally install the companion toggle filter for UI activation

## Configuration (Valves)

| Valve | Default | Description |
|-------|---------|-------------|
| `GOOGLE_API_KEY` | `""` | Google AI API Key (required) |
| `USE_STREAMING` | `true` | Use SSE streaming for real-time thinking summaries |
| `POLLING_INTERVAL` | `10.0` | Seconds between polls (fallback mode only) |
| `MAX_RESEARCH_TIME` | `3600` | Maximum research time in seconds (60 mins) |
| `CONNECTION_TIMEOUT` | `120` | HTTP connection timeout in seconds |
| `DEADLINE_RETRIES` | `10` | Number of SSE reconnection attempts before switching to polling |

## Usage

### Direct Model Selection

Select "Gemini Deep Research" from the model dropdown and send your research query.

### With Toggle Filter

Install the `gemini_deep_research_toggle` filter to enable Deep Research via a UI toggle button.

### Follow-up Questions

After receiving a research report, you can ask follow-up questions. The pipe will continue the research in the context of the previous interaction.

## Example Queries

- "Research the history of quantum computing with focus on recent 2024-2025 breakthroughs"
- "Compare different approaches to sustainable aviation fuel production"
- "Analyze the current state of fusion energy research and commercial viability"

## How It Works

1. Your query is sent to Google's Deep Research agent via the Interactions API
2. The agent autonomously searches the web using `google_search` and `url_context` tools
3. The pipe streams thinking summaries (and output text) over SSE and automatically resumes if the stream is interrupted
4. A comprehensive research report is generated (typically 10-30 minutes)
5. Follow-up questions continue the conversation using `previous_interaction_id`

## Limitations

- No custom tool support (Deep Research uses only built-in tools)
- No file upload support (RAG is bypassed when using Deep Research)
- Research can take up to 60 minutes for complex topics
- Requires stable internet connection

## API Reference

Uses Google's Interactions API:
- Agent: `deep-research-pro-preview-12-2025`
- Endpoint: `POST /v1beta/interactions`
- Streaming: SSE with `alt=sse`, `stream=true`, `agent_config={"thinking_summaries": "auto"}`
