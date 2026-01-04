## [0.1.4] - 2026-01-04
- Rename tool to `AskUserQuestion` (keeps `prompt_form` as a backwards-compatible alias).

## [0.1.3] - 2026-01-04
- Restyle the injected form modal to match Open WebUI’s Tailwind UI (Modal/InputVariablesModal classes).

## [0.1.2] - 2026-01-04
- Accept `key` as an alias for `name` on fields (common LLM output).
- Coerce select `options` provided as `{label,value}` objects into strings.
- Accept the `schema` argument as a JSON string (and parse it).

## [0.1.1] - 2026-01-04
- Catch `__event_call__` failures and return a non-empty error payload (prevents downstream tool_result validation errors).
- Send both `code` and `script` fields for better Open WebUI version compatibility.

## [0.1.0] - 2026-01-04
- Initial release: render a modal form in the browser and return filled values via tool calling.
