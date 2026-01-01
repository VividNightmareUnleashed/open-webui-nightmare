# Changelog

## [0.3.1] - 2026-01-01

### Fixed
- SSE event parsing: `event_type` is inside JSON data, not SSE `event:` line
- Thought summaries now correctly extracted from `content.delta` with `delta.type="thought_summary"`
- Removed stuck timer from streaming mode (no timer in SSE, kept in polling)

## [0.3.0] - 2026-01-01

### Added
- SSE streaming with real-time thinking summaries
- Follow-up questions support via `previous_interaction_id`
- `USE_STREAMING` valve to toggle between SSE and polling modes
- Automatic fallback to polling if SSE fails

### Changed
- Increased connection timeout default to 120 seconds
- Timer now updates on every poll in fallback mode

## [0.2.0] - 2026-01-01

### Changed
- Switched from SDK to raw HTTP requests (aiohttp)
- No longer requires google-genai SDK
- Works with any Open WebUI installation

### Removed
- google-genai SDK dependency

## [0.1.0] - 2025-01-01

### Added
- Initial release
- Polling-based research with status updates
- Configurable research timeout (default 60 mins)
- Error handling with user-friendly messages
