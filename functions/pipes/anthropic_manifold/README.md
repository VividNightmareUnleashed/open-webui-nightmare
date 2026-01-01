# Anthropic Manifold

Full-featured Anthropic Claude API integration for Open WebUI with extended thinking, web search, native tool calling, and more.

## Features

- **Claude Opus 4.5** - Latest model with full feature support
- **Extended Thinking** - Display Claude's reasoning process with `<think>` tags
- **Interleaved Thinking** - Think-Act-Think-Act pattern for tool use
- **Web Search** - Native Anthropic web search with automatic citations
- **Native Tool Calling** - Execute OpenWebUI tools via Anthropic's tool use API
- **Code Execution** - Python/Bash sandbox execution (intercepts OpenWebUI code interpreter toggle)
- **Files API** - Upload files to Anthropic's Files API for direct access in code execution sandbox
- **Document Citations** - Cite exact passages from uploaded documents with source attribution
- **MCP Connector** - Connect to remote MCP servers
- **Prompt Caching** - Automatic cache control for reduced costs

## Supported Models

| Model ID | Display Name | Features |
|----------|--------------|----------|
| `claude-opus-4-5-20251101` | claude-4.5-opus | All features |
| `claude-sonnet-4-5-20250929` | claude-4.5-sonnet | All features |
| `claude-sonnet-4-5-20250929-think` | claude-4.5-sonnet-thinking | Extended thinking |
| `claude-opus-4-1-20250805` | claude-4.1-opus | Tools, web search |
| `claude-haiku-4-5-20251001` | claude-4.5-haiku | Fast, tools |
| `claude-3-5-haiku-latest` | claude-3.5-haiku | Basic |

## Requirements

### Companion Filter (for toggle detection)

To use OpenWebUI's web search and code interpreter toggles with Claude's native tools, install the **Anthropic Manifold Companion** filter alongside this pipe.

Without the companion filter:
- Valves work (enable `ENABLE_WEB_SEARCH` or `ENABLE_CODE_EXECUTION`)
- UI toggles do NOT work (OpenWebUI runs its own handlers instead)

With the companion filter:
- Both valves AND UI toggles work correctly
- Claude's native web search and code execution tools are used

## Configuration

### Required

| Valve | Description |
|-------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |

### Extended Thinking

| Valve | Default | Description |
|-------|---------|-------------|
| `ENABLE_THINKING` | `true` | Enable extended thinking |
| `THINKING_BUDGET` | `16000` | Token budget (1024-32000) |
| `DISPLAY_THINKING` | `true` | Show thinking in chat |
| `ENABLE_INTERLEAVED_THINKING` | `true` | Think between tool calls |
| `EFFORT_LEVEL` | `not_set` | `low`/`medium`/`high` (overrides budget) |

### Web Search

| Valve | Default | Description |
|-------|---------|-------------|
| `ENABLE_WEB_SEARCH` | `false` | Enable web search |
| `WEB_SEARCH_MAX_USES` | `5` | Max searches per request |
| `WEB_SEARCH_ALLOWED_DOMAINS` | `""` | Comma-separated allowed domains |
| `WEB_SEARCH_BLOCKED_DOMAINS` | `""` | Comma-separated blocked domains |
| `WEB_SEARCH_USER_LOCATION` | `""` | JSON location for localized results |

Web search also activates when users enable the OpenWebUI web search toggle (requires companion filter).

### Tool Calling

| Valve | Default | Description |
|-------|---------|-------------|
| `ENABLE_NATIVE_TOOLS` | `true` | Use Anthropic native tool calling |
| `MAX_TOOL_LOOPS` | `10` | Max tool call iterations |
| `PARALLEL_TOOL_CALLS` | `true` | Execute tools in parallel |

### Advanced

| Valve | Default | Description |
|-------|---------|-------------|
| `ENABLE_CODE_EXECUTION` | `false` | Enable Python sandbox |
| `ENABLE_CITATIONS` | `false` | Enable document citations |
| `MCP_SERVERS_JSON` | `""` | JSON array of MCP servers |
| `ENABLE_CONTEXT_MANAGEMENT` | `false` | Auto-clear stale tool calls |
| `BETA_FEATURES` | `""` | Additional beta headers |

## Usage Examples

### Basic Chat

Just select any Anthropic model and chat normally.

### Extended Thinking

Select the `claude-4.5-sonnet-thinking` model for complex reasoning tasks. The thinking process will be displayed in `<think>` tags.

### Web Search

1. Enable the `ENABLE_WEB_SEARCH` valve, **OR**
2. Use the OpenWebUI web search toggle in the chat interface

Claude will automatically search the web when needed and cite sources.

### Tool Calling

OpenWebUI tools are automatically converted to Anthropic's tool format and executed natively. The manifold handles:
- Tool discovery from `__tools__`
- Input schema transformation
- Parallel execution
- Result feeding back to Claude

### Code Execution

1. Enable the `ENABLE_CODE_EXECUTION` valve, **OR**
2. Use the OpenWebUI code interpreter toggle in the chat interface

Claude can:
- Run Python code in a sandboxed environment
- Execute Bash commands
- Create, view, and edit files
- Generate visualizations and charts

Results are displayed as formatted code blocks with stdout/stderr output.

**Supported models**: Claude Opus 4.5, Claude Sonnet 4.5

### Files API

When you upload files in OpenWebUI with the companion filter installed:

1. Files are uploaded to Anthropic's Files API (bypassing OpenWebUI's RAG)
2. Files become available in the code execution sandbox
3. Claude can access files directly: `pd.read_csv('/mnt/user/data.csv')`

**File types and how they're handled:**

| File Type | With Code Execution | Without Code Execution |
|-----------|---------------------|------------------------|
| CSV, Excel, JSON | `container_upload` (sandbox access) | Inline in context |
| PDF, Text | `document` block (Claude reads directly) | `document` block |
| Images | `image` block (Claude sees directly) | `image` block |

**Benefits:**
- Reduced context usage (file content not embedded in messages)
- Proper file operations (pandas, image processing, binary files)
- Same experience as claude.ai

**Note:** Requires the companion filter with `BYPASS_BACKEND_RAG` enabled (default: true).

### Document Citations

When you enable `ENABLE_CITATIONS` and upload documents (PDFs, text files), Claude will cite exact passages from those documents:

1. Enable the `ENABLE_CITATIONS` valve
2. Upload a PDF or text file
3. Ask Claude questions about the document

Claude's response will include citations that appear in the OpenWebUI sidebar, showing:
- The document title
- The exact quoted text
- The location (page number for PDFs, character range for text)

**Citation types:**

| Document Type | Citation Format |
|---------------|-----------------|
| PDF | Page number (e.g., "Page 5") |
| Plain text | Character range (e.g., "Characters 100-250") |

**Notes:**
- Citations only work with `document` blocks (PDFs, text files without code execution)
- All Claude models support citations except Haiku 3
- `cited_text` doesn't count toward output tokens (cost savings)
- Not compatible with code execution sandbox files (`container_upload`)

### MCP Servers

Configure remote MCP servers via the `MCP_SERVERS_JSON` valve:

```json
[
  {
    "url": "https://mcp.example.com",
    "auth_token": "your-token"
  }
]
```

## Differences from Original

This manifold is a complete rewrite with:

1. **Async Architecture** - Uses `aiohttp` instead of `requests` for better performance
2. **Full Pipe Signature** - Accepts all injected arguments (`__tools__`, `__metadata__`, etc.)
3. **Event Emission** - Proper status updates and citation events
4. **Web Search Integration** - Intercepts OpenWebUI's toggle for native search
5. **Tool Execution Loop** - Multi-turn tool calling with result handling
6. **Interleaved Thinking** - Beta feature for tool use reasoning

## API Reference

### Anthropic Beta Headers Used

| Feature | Header |
|---------|--------|
| Interleaved Thinking | `interleaved-thinking-2025-05-14` |
| Code Execution | `code-execution-2025-08-25` |
| Files API | `files-api-2025-04-14` |
| Context Management | `context-management-2025-06-27` |
| MCP Connector | `mcp-client-2025-04-04` |

### Web Search Tool Type

```json
{
  "type": "web_search_20250305",
  "name": "web_search",
  "max_uses": 5
}
```

### Code Execution Tool Type

```json
{
  "type": "code_execution_20250825",
  "name": "code_execution"
}
```

## Troubleshooting

### Web search not working

- Verify your Anthropic organization has web search enabled in Console
- Check that the model supports web search (Opus 4.5, Sonnet 4.5, etc.)
- Enable `ENABLE_WEB_SEARCH` valve or use OpenWebUI toggle

### Tools not executing

- Ensure `ENABLE_NATIVE_TOOLS` is `true`
- Check that the model supports tool use
- Verify tools are properly defined in OpenWebUI

### Thinking not displayed

- Use a `-think` model variant, or
- Ensure `ENABLE_THINKING` and `DISPLAY_THINKING` are `true`
- Model must support extended thinking

### Code execution not working

- Verify your Anthropic organization has code execution enabled
- Use a supported model (Opus 4.5 or Sonnet 4.5)
- Enable `ENABLE_CODE_EXECUTION` valve or use OpenWebUI code interpreter toggle

### Files not being uploaded to Anthropic

- Ensure the companion filter is installed and enabled
- Check that `BYPASS_BACKEND_RAG` is `true` in the companion filter
- Temporary chats ("local" chat_id) don't support file upload - use a saved chat
- Check logs for file upload errors

### Citations not appearing

- Enable `ENABLE_CITATIONS` valve
- Upload a PDF or text file (not images or code execution files)
- Code execution sandbox files (`container_upload`) don't support citations
- Haiku 3 doesn't support citations
- Claude Sonnet 3.7 may need explicit prompting: "Use citations to back up your answer"

## License

MIT
