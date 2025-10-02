# Priority Filter

Enables OpenAI priority processing for faster responses with latency SLAs.

## Description

This filter adds a `service_tier` parameter to your chat requests, controlling OpenAI's processing speed and priority level. When enabled with "priority" tier, requests are processed faster with guaranteed latency SLAs (at premium pricing).

**Note**: The `service_tier` parameter works with all OpenAI models through the Responses API.

## Features

- **Toggle button in compose UI** - Appears as a lightning bolt icon next to other feature toggles
- **No model switching** - Keeps your selected model, only changes processing tier
- **Configurable service tier** - Choose priority, default, or flex via Valves

## Valves

- `SERVICE_TIER`: Sets the processing tier. Options:
  - `"priority"` - Premium pricing with latency SLAs (faster) (default)
  - `"default"` - Standard pricing and performance
  - `"flex"` - 50% cheaper with increased latency (only for o3, o4-mini, gpt-5)

- `priority`: Priority level for filter operations (default: 0)

## Installation

1. Copy the contents of `priority_filter.py`
2. In Open WebUI, go to **Workspace → Functions**
3. Click **Create New Function**
4. Select **Filter** type
5. Paste the code
6. Save and enable the filter

## Usage

Once installed:
1. A lightning bolt toggle button will appear in the message input area
2. Click to **enable** - adds `service_tier: "priority"` for faster responses
3. Click again to **disable** - removes service tier (uses default)

The filter works with all OpenAI models that support the Responses API.

## Combining with Other Filters

This filter can be used **together with**:
- **Extended Thinking** (clock icon) → adds `reasoning_effort` for reasoning models
- **Verbose** (text lines icon) → adds `text.verbosity` for GPT-5 models

All filters can be enabled simultaneously for maximum control over model behavior.

## Service Tier Details

### Priority Processing
- **Cost**: Premium pricing (higher than default)
- **Benefit**: Latency SLAs, faster response times
- **Use case**: When speed is critical

### Default Processing
- **Cost**: Standard pricing
- **Benefit**: Normal performance
- **Use case**: Regular usage

### Flex Processing
- **Cost**: 50% cheaper than default
- **Benefit**: Lower cost
- **Drawback**: Increased latency (slower)
- **Models**: Only works with o3, o4-mini, gpt-5
- **Use case**: When cost matters more than speed

## Technical Details

The filter sets a top-level parameter:
```python
body["service_tier"] = "priority"  # or "default" or "flex"
```

This is similar to `reasoning_effort` (top-level) and different from `text.verbosity` (nested under `text`).

For more information, see:
- [OpenAI Priority Processing](https://openai.com/api-priority-processing/)
- [Priority Processing FAQ](https://help.openai.com/en/articles/11647665-priority-processing-faq)

## Notes

- This filter does NOT change your selected model
- Priority processing costs more but provides faster responses
- Flex processing is cheaper but slower (and only works on certain models)
- You can adjust the default service tier in the Valves configuration
- Works best with the OpenAI Responses Manifold pipe
