## [0.1.10] - 2026-01-04
- Make the `schema` argument strict-schema compatible for OpenAI Responses native tool calling (prevents `invalid_function_parameters` 400s).

## [0.1.11] - 2026-01-04
- Capitalize select/multiselect option labels by default for readability (leaves identifiers like `gpt-5.2` unchanged).

## [0.1.12] - 2026-01-04
- Add an optional “additional info” popover on Confirm; included in the tool result only when filled.

## [0.1.6] - 2026-01-04
- Adjust cancel `refusal` message wording to be more explicit and consistent.

## [0.1.7] - 2026-01-04
- Send a user notification when a form is opened.

## [0.1.8] - 2026-01-04
- Make “model would like you to answer some questions” notifications non-optional.

## [0.1.9] - 2026-01-04
- Prefer the selected chat model name from `__metadata__["model"]` for notifications (avoids showing the tool-calling task model).

## [0.1.5] - 2026-01-04
- Avoid WebSocket call timeouts by opening the UI immediately and polling for results (`timeout_seconds`, `poll_interval_ms`).
- Add a Cancel hover popover (“Explain model what to do instead”) that optionally returns a `refusal` string.
- Accept `id` as an alias for field `name`.

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
