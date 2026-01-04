# Changelog

All notable changes to this project will be documented in this file.

## [1.2.4] - 2026-01-04

### Fixed

- **Tool Result Errors** - Ensure `tool_result.content` is never empty when `is_error=true` (prevents Anthropic API 400s on empty exception messages).
- **Python 3.11 Compatibility** - Avoid nested f-strings with backslash escapes inside f-string expressions (SyntaxError on Python <=3.11).

## [1.2.2] - 2026-01-02

### Fixed

- **Code Fence Safety** - Tool outputs and file contents are now wrapped in dynamic-length fenced code blocks to prevent nested ``` fences from breaking formatting.

## [1.2.1] - 2026-01-02

### Fixed

- **Streaming Stability** - Reduced HTML/markdown breakage by buffering streamed deltas and avoiding mixed `yield` + `message` event emission.
- **Status Lifecycle** - Tool status updates now mark completion (`done=True`) so progress indicators clear correctly.

## [1.2.0] - 2025-01-01

### Fixed

- **Interleaved Thinking with Tool Use** - Fixed thinking block preservation during tool call loops
  - Thinking blocks are now accumulated during streaming
  - Properly prepended to assistant messages before tool_use blocks
  - Captures signature via `signature_delta` event (not at block start)
  - Fixes API error: "Expected `thinking` or `redacted_thinking`, but found `tool_use`"
  - Fixes API error: "Invalid `signature` in `thinking` block"
  - Enables external tools (like Nano Banana Pro) to work with extended thinking enabled

## [1.1.0] - 2025-01-01

### Added

- **Document Citations** - Cite exact passages from uploaded documents
  - Enable with `ENABLE_CITATIONS` valve
  - Works with PDF and text files (document blocks)
  - Citations appear in OpenWebUI sidebar
  - Shows document title, quoted text, and location (page or character range)
  - Handles `citations_delta` streaming events
  - Not compatible with code execution sandbox files (`container_upload`)

## [1.0.0] - 2025-01-01

### Added

- **Claude Opus 4.5 Support** - Added `claude-opus-4-5-20251101` to model list
- **Code Execution Tool** - Native Anthropic sandbox execution
  - Intercepts OpenWebUI's code interpreter toggle (`features["code_interpreter"]`)
  - Uses `code_execution_20250825` tool type
  - Beta header: `code-execution-2025-08-25`
  - Handles `bash_code_execution_tool_result` and `text_editor_code_execution_tool_result` blocks
  - Displays stdout/stderr as formatted code blocks
- **Async Architecture** - Complete rewrite using `aiohttp` for better performance
- **Full Pipe Signature** - Now accepts `__tools__`, `__metadata__`, `__event_emitter__`, etc.
- **Web Search Tool** - Native Anthropic web search with `web_search_20250305`
  - Intercepts OpenWebUI's web search toggle
  - Automatic citation emission
  - Domain filtering support
  - User location for localized results
- **Native Tool Calling** - OpenWebUI tools converted to Anthropic format
  - Parallel tool execution
  - Multi-turn tool loops
  - Error handling with `is_error` flag
- **Interleaved Thinking** - Beta feature for think-act-think-act pattern
  - Uses `interleaved-thinking-2025-05-14` header
  - Thinking blocks between tool calls
- **Effort Parameter** - Control thinking budget with `low`/`medium`/`high`
- **MCP Connector** - Connect to remote MCP servers
  - Uses `mcp-client-2025-04-04` beta header
- **Context Management** - Automatic tool call clearing
  - Uses `context-management-2025-06-27` beta header
- **Event Emission** - Proper status and citation events
  - Status updates during tool execution
  - Citation events from web search
  - Error events with structured data
- **Feature Support Dictionary** - Model capability detection
- **Configurable Model List** - Via `MODEL_IDS` valve

### Changed

- Moved to `functions/pipes/anthropic_manifold/` directory structure
- Renamed from `anthropic manifold.py` to `anthropic_manifold.py`
- Extended thinking now requires `-think` model suffix
- Cache control improvements with TTL support

### Fixed

- Proper handling of `web_search_tool_result` blocks
- Thinking tag closure in all scenarios
- Image size validation for URL images

## [0.3.2] - Previous Version

- Original implementation by justinh-rahb et al.
- Basic extended thinking support
- Synchronous requests library
- Cache control on messages
