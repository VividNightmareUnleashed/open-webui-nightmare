# Gemini Deep Research Toggle Filter

A companion toggle filter for the Gemini Deep Research pipe. Adds a UI toggle button to enable comprehensive web research on any conversation.

## Features

- **UI Toggle**: Adds a research icon toggle to the chat interface
- **Automatic Routing**: Routes requests to the Deep Research pipe when enabled
- **RAG Bypass**: Bypasses Open WebUI's RAG (Deep Research doesn't support file uploads)
- **File Warning**: Displays a message if files are attached
- **Follow-up Support**: Passes `previous_interaction_id` for follow-up questions

## Requirements

- Gemini Deep Research pipe must be installed
- Open WebUI 0.6.10+

## Installation

1. Copy `gemini_deep_research_toggle.py` to Open WebUI Functions
2. Ensure the Gemini Deep Research pipe is installed and configured
3. The toggle icon will appear in the chat interface

## Configuration (Valves)

| Valve | Default | Description |
|-------|---------|-------------|
| `DEFAULT_MODEL` | `gemini_deep_research.deep-research-pro-preview-12-2025` | Target pipe/model to route requests |
| `priority` | `0` | Filter execution priority |

## Usage

1. Click the research toggle icon (Gemini logo) in the chat interface
2. Send your research query
3. Watch as Deep Research autonomously investigates your topic
4. Receive a comprehensive research report
5. Ask follow-up questions to continue the research

## How It Works

When the toggle is enabled:

1. **Inlet Phase**:
   - Routes the request to the Deep Research pipe
   - Disables OpenWebUI's native `web_search` feature
   - Bypasses RAG via `file_handler = True` and clearing files
   - Shows warning if files are attached
   - Passes `previous_interaction_id` for follow-ups
   - Forces streaming mode for progress updates

2. **Outlet Phase**:
   - Ensures correct model attribution on responses

## Icon

The toggle displays the Gemini sparkle logo, representing Google's Deep Research capabilities.

## Limitations

- Files attached to the chat will be ignored (Deep Research doesn't support file uploads)
- Disables OpenWebUI's built-in web search (Deep Research has its own)
