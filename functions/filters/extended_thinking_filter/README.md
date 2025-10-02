# Extended Thinking Filter

Adds reasoning effort to requests without changing the selected model.

## Description

This filter adds a `reasoning_effort` parameter to your chat requests while keeping your currently selected model. Unlike the `reason_toggle_filter`, this does not switch models - it only enhances the reasoning capability of whatever model you have selected.

## Features

- **Toggle button in compose UI** - Appears as a lightbulb icon next to other feature toggles
- **No model switching** - Keeps your selected model, only adds reasoning effort
- **Configurable effort level** - Default is "high", but can be changed via Valves

## Valves

- `REASONING_EFFORT`: Sets the reasoning effort level. Options:
  - `"minimal"` - Minimal reasoning
  - `"low"` - Low reasoning
  - `"medium"` - Medium reasoning
  - `"high"` - High reasoning (default)
  - `"not set"` - Disabled, no reasoning effort added

- `priority`: Priority level for filter operations (default: 0)

## Installation

1. Copy the contents of `extended_thinking_filter.py`
2. In Open WebUI, go to **Workspace → Functions**
3. Click **Create New Function**
4. Select **Filter** type
5. Paste the code
6. Save and enable the filter

## Usage

Once installed:
1. A lightbulb toggle button will appear in the message input area
2. Click to **enable** - adds `reasoning_effort: "high"` to your requests
3. Click again to **disable** - removes reasoning effort

The filter works with any model that supports the `reasoning_effort` parameter (like OpenAI's o1/o3 models).

## Notes

- This filter does NOT change your selected model
- If your model doesn't support `reasoning_effort`, the parameter will be ignored
- You can adjust the default effort level in the Valves configuration
