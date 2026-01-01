# Changelog

## [0.2.0] - 2026-01-01

### Added
- RAG bypass via `file_handler = True`
- File upload warning message
- Follow-up questions support via `previous_interaction_id`

### Changed
- Now clears `body["files"]` to ensure RAG bypass

## [0.1.0] - 2025-01-01

### Added
- Initial release
- UI toggle with research icon
- Automatic routing to Deep Research pipe
- Feature flag management for pipe communication
- Model attribution in responses
