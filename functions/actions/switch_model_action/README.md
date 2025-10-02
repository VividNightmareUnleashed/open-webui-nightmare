# Pro Mode Action

Adds a button underneath assistant messages to regenerate the response using a pro/premium model by invoking the manifold directly.

## Description

This action function creates a button in the message toolbar that allows you to regenerate any assistant response using a more powerful model. When clicked, it **invokes the OpenAI Responses Manifold directly** with your configured pro model, ensuring all manifold logic and active filters are respected.

**Key advantage**: Unlike simple model switching, this action calls the manifold's pipe method directly, which means all your active filters (Extended Thinking, Verbose, Priority) apply to the regenerated response, and all OpenAI Responses API features (reasoning_effort, text.verbosity, service_tier) work correctly.

## Features

- **Button in message toolbar** - Appears underneath each assistant message with Pro Mode icon
- **Invokes manifold directly** - Calls the OpenAI Responses Manifold pipe method, not just simple model switching
- **Respects all filters** - Extended Thinking, Verbose, and Priority filters apply to regeneration
- **Full manifold compatibility** - All OpenAI Responses API features work (reasoning_effort, text.verbosity, service_tier)
- **Streaming support** - Handles both streaming and non-streaming responses
- **Error handling** - Shows clear notifications if manifold loading or generation fails
- **Minimal feedback** - Shows quick "Pro Mode" status, then regenerates

## Valves

- `TARGET_MODEL`: The model to switch to when regenerating (default: `"openai_responses.gpt-5-thinking-high"`)
  - Use full manifold model ID like `"openai_responses.gpt-5-thinking-high"`
  - Supports all OpenAI Responses Manifold model IDs

- `MANIFOLD_ID`: The ID of the manifold pipe to invoke (default: `"openai_responses"`)
  - Change this if your manifold has a different ID
  - The action will load this pipe and call its `pipe()` method

## Installation

1. Copy the contents of `switch_model_action.py`
2. In Open WebUI, go to **Workspace → Functions**
3. Click **Create New Function**
4. Select **Action** type
5. Paste the code
6. Save and enable the action

## Usage

Once installed:
1. Send a message and get a response from any model
2. Look for the **Pro Mode** button underneath the assistant's message
3. Click the button to regenerate the response with your configured pro model
4. You'll see a quick "Pro Mode" status, then the message regenerates

## Configuration

### Required Setup

**This action requires the OpenAI Responses Manifold to be installed.** It invokes the manifold directly to regenerate responses.

Configure the target model using the full manifold model ID format:

```python
TARGET_MODEL: "openai_responses.gpt-5-thinking-high"  # Pro thinking model
# or
TARGET_MODEL: "openai_responses.o3-mini"  # O3 mini
# or
TARGET_MODEL: "openai_responses.gpt-5-chat-latest"  # Fast chat model
```

If your manifold has a different ID (not `openai_responses`), also configure:

```python
MANIFOLD_ID: "your_manifold_id"
```

## Use Cases

### Quick Model Upgrade
- Get response from fast model (e.g., `gpt-5-chat-latest`)
- Click "Pro Mode" to regenerate with premium thinking model (e.g., `gpt-5-thinking-high`)
- Compare quality and reasoning depth

### Model Fallback
- If response from one model is unsatisfactory
- Switch to pro model for better results
- Useful when a model doesn't fully understand the context

### Cost Optimization
- Start with cheaper fast model for simple queries
- Use Pro Mode button only when you need deeper thinking
- Pay premium only when necessary

## Advanced: Multiple Model Actions

You can create variants of this action for different target models:

**Example: Create multiple action files:**
1. `pro_mode_thinking.py` - Switches to `openai_responses.gpt-5-thinking-high`
2. `pro_mode_o3.py` - Switches to `openai_responses.o3-mini`
3. `fast_mode.py` - Switches back to `openai_responses.gpt-5-chat-latest`

Each will appear as a separate button in the message toolbar.

## Technical Details

### How It Works

Unlike simple model switching, this action **invokes the manifold pipe directly**:

1. **Load manifold module**: Uses `get_function_module_from_cache()` to load the OpenAI Responses Manifold
2. **Load manifold valves**: Fetches the manifold's configuration from the database
3. **Change model**: Sets `body["model"]` to the target pro model
4. **Call manifold pipe**: Invokes `manifold_module.pipe()` with all parameters
5. **Handle streaming**: Collects streamed chunks from the manifold's async generator
6. **Return content**: Returns `{"content": "full response"}` to display the result

```python
# Load the manifold
manifold_module, _, _ = get_function_module_from_cache(__request__, "openai_responses")

# Set up its valves
valves = Functions.get_function_valves_by_id("openai_responses")
manifold_module.valves = manifold_module.Valves(**(valves if valves else {}))

# Change model
body["model"] = "openai_responses.gpt-5-thinking-high"

# Call the manifold's pipe
result = await manifold_module.pipe(
    body=body,
    __user__=__user__,
    __request__=__request__,
    __event_emitter__=__event_emitter__,
    __metadata__=__metadata__ or {},
    __tools__=__tools__,
    __task__=__task__,
    __task_body__=__task_body__,
    __event_call__=__event_call__,
)

# Handle streaming response
if inspect.isasyncgen(result):
    full_response = ""
    async for chunk in result:
        full_response += str(chunk)
    return {"content": full_response}
```

### Why This Approach?

**Calling the manifold directly ensures**:
- ✅ **All filters apply**: Extended Thinking, Verbose, Priority filters modify body before manifold sees it
- ✅ **Full manifold logic**: reasoning_effort, text.verbosity, service_tier parameters work correctly
- ✅ **OpenAI Responses API**: All API features (thinking tokens, reasoning summaries, etc.) work
- ✅ **Proper streaming**: Manifold handles streaming via `__event_emitter__` internally

**Without this approach**:
- ❌ Filters would be bypassed
- ❌ Manifold parameters wouldn't work
- ❌ OpenAI Responses API features would fail

### Status Feedback

Shows minimal status:
```python
await __event_emitter__({
    "type": "status",
    "data": {"description": "Pro Mode", "done": True}
})
```

This provides quick feedback without verbose messages.

## Notes

- **Requires OpenAI Responses Manifold** - This action specifically invokes the manifold pipe
- The action button appears for ALL assistant messages (cannot be model-specific)
- Regeneration uses the same conversation context and all active filters
- Original message is replaced (not preserved)
- All manifold parameters (reasoning_effort, text.verbosity, service_tier) work correctly
- Active filters (Extended Thinking, Verbose, Priority) apply to the regenerated response
- Minimal UI feedback - just shows "Pro Mode" briefly
- Error notifications shown if manifold loading or generation fails

## Compatibility with Filters

This action works seamlessly with all filter toggle buttons:

- **Extended Thinking Filter** ✅ - If enabled, adds reasoning_effort to regeneration
- **Verbose Filter** ✅ - If enabled, adds text.verbosity to regeneration
- **Priority Filter** ✅ - If enabled, adds service_tier to regeneration

All filters modify the body **before** the action runs, so the manifold receives the fully configured request.
