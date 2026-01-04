# Form Prompt Tool

Interactive, **structured user input** for Open WebUI tool calling.

This tool uses Open WebUI’s built-in `__event_call__` + `execute` event to render a modal form in the browser (text inputs, selects, checkboxes, etc.) and returns the filled values to the model as JSON.
The injected modal uses the same Tailwind classnames as Open WebUI’s native modal/input-variables UI, so it blends in visually.

It also notifies the user that the model is waiting for input (toast + best-effort system notification when supported).

## Install

1. Open WebUI → **Admin Panel** → **Tools**
2. Create a new tool with:
   - **id**: `AskUserQuestion`
   - paste the contents of `tools/form_prompt/form_prompt.py`

## How To Use

Enable the tool for your chat, then ask the model to call `AskUserQuestion`.

Example prompt you can give the model:

> Use the `AskUserQuestion` tool to collect the missing details. Use a checkbox for optional add-ons, and a select for pace.

Example tool call schema:

```json
{
  "title": "Trip Planner",
  "description": "Fill this out so I can generate a tailored itinerary.",
  "submit_label": "Generate plan",
  "cancel_label": "Cancel",
  "fields": [
    {"name": "destination", "label": "Destination", "type": "text", "required": true, "placeholder": "e.g. Tokyo"},
    {"name": "days", "label": "Number of days", "type": "number", "required": true, "min": 1, "max": 30, "step": 1, "default": 5},
    {"name": "pace", "label": "Pace", "type": "select", "options": ["Relaxed", "Balanced", "Packed"], "default": "Balanced"},
    {"name": "include_museums", "label": "Include museums", "type": "checkbox", "default": true},
    {"name": "notes", "label": "Anything else?", "type": "textarea", "placeholder": "Preferences, constraints, etc."}
  ]
}
```

Notes:
- Fields should use `name`, but `key` and `id` are also accepted.
- For `select` / `multiselect`, `options` should be an array of strings; `{ "label": "...", "value": "..." }` objects are accepted and coerced to strings.

Return value (to the model):

```json
{"cancelled": false, "values": {"destination": "Tokyo", "days": 5, "pace": "Balanced", "include_museums": true, "notes": ""}}
```

Cancel behavior:
- If the user cancels: `{"cancelled": true}`
- If the user cancels and fills “Explain model what to do instead”: `{"cancelled": true, "refusal": "User cancelled the form and declined to answer. Additional info, if the user provided: <note>"}` (the `refusal` field is omitted if empty)

Timeout behavior:
- If the user doesn’t respond within `timeout_seconds`: `{"timeout": true, "error": "Timed out ..."}` (no `cancelled`/`refusal`)
- Set `timeout_seconds=0` to wait indefinitely

## Supported Field Types

- `text`, `textarea`
- `number` (supports `min`, `max`, `step`)
- `select` (requires `options`)
- `multiselect` (requires `options`, returns a list of strings)
- `checkbox` (returns boolean)
- `email`, `url`, `date`, `time`

## Safety Notes

- This tool relies on Open WebUI’s **client-side `execute` event** (JavaScript execution in the browser). Treat it as a privileged capability and only enable it in environments you trust.
