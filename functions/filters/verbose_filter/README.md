# Verbose Filter

Adds verbosity parameter to requests for more detailed responses.

## Description

This filter adds a `text.verbosity` parameter to your chat requests, controlling how detailed the model's responses should be. Unlike the `extended_thinking_filter` which adds reasoning effort, this filter specifically controls response verbosity.

**Note**: The `text.verbosity` parameter is only supported by **GPT-5 models** (gpt-5, gpt-5-mini, gpt-5-nano). Other models will ignore this parameter.

## Features

- **Toggle button in compose UI** - Appears as a document/lines icon next to other feature toggles
- **No model switching** - Keeps your selected model, only adds verbosity parameter
- **Configurable verbosity level** - Default is "high", but can be changed via Valves

## Valves

- `VERBOSITY`: Sets the verbosity level. Options:
  - `"low"` - More concise responses
  - `"medium"` - Balanced verbosity
  - `"high"` - More detailed responses (default)

- `priority`: Priority level for filter operations (default: 0)

## Installation

1. Copy the contents of `verbose_filter.py`
2. In Open WebUI, go to **Workspace → Functions**
3. Click **Create New Function**
4. Select **Filter** type
5. Paste the code
6. Save and enable the filter

## Usage

Once installed:
1. A document/lines toggle button will appear in the message input area
2. Click to **enable** - adds `text.verbosity: "high"` to your requests
3. Click again to **disable** - removes verbosity parameter

The filter works only with GPT-5 family models that support the `text.verbosity` parameter.

## Combining with Other Filters

This filter can be used **together with** the `extended_thinking_filter`:
- **Extended Thinking** (clock icon) → adds `reasoning_effort` for reasoning models
- **Verbose** (document icon) → adds `text.verbosity` for GPT-5 models

Both filters can be enabled simultaneously if you want both reasoning and verbosity control.

## Technical Details

The filter sets a nested parameter structure:
```python
body["text"]["verbosity"] = "high"  # or "low" or "medium"
```

This follows OpenAI's Responses API pattern where certain text generation parameters are nested under a `text` object. See the [OpenAI Responses Manifold documentation](https://github.com/jrkropp/open-webui-developer-toolkit/tree/main/functions/pipes/openai_responses_manifold) for more details on supported features.

## Notes

- This filter does NOT change your selected model
- Only GPT-5 models support `text.verbosity` - other models will ignore it
- You can adjust the default verbosity level in the Valves configuration
- Works best with the OpenAI Responses Manifold pipe
