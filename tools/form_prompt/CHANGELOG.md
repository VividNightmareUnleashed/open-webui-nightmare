## [0.1.1] - 2026-01-04
- Catch `__event_call__` failures and return a non-empty error payload (prevents downstream tool_result validation errors).
- Send both `code` and `script` fields for better Open WebUI version compatibility.

## [0.1.0] - 2026-01-04
- Initial release: render a modal form in the browser and return filled values via tool calling.
