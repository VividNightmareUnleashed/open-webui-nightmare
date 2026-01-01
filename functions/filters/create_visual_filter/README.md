# Create Visual Filter

Filter that intercepts `<create_visual>` tags in model output and generates images inline using Google's Gemini API.

## How It Works

**Inlet (before model):** Automatically appends usage instructions to the system prompt, teaching the model how to use `<create_visual>` tags.

**Outlet (after model):** When a model outputs text containing `<create_visual>description</create_visual>` tags, this filter:

1. Extracts the description from each tag
2. Calls Gemini API to generate an image
3. Uploads the image to Open WebUI storage
4. Replaces the tag with an inline `<img>` element

This allows models to generate multiple images per message without native function calling round-trips. No manual system prompt editing required.

## Tag Syntax

```
<create_visual>Your image description here</create_visual>
<create_visual ratio="1:1">Square image description</create_visual>
```

Supported ratios: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`

## Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| GEMINI_API_KEY | (required) | Google AI Studio API key |
| MODEL | gemini-2.5-flash-image | Gemini model |
| DEFAULT_ASPECT_RATIO | 16:9 | Default aspect ratio |
| MAX_WIDTH | 512 | Max image display width (px) |

## Setup

1. Add the filter via Functions admin panel
2. Configure `GEMINI_API_KEY`
3. Enable the filter globally or for specific models
4. Update model system prompts to use the `<create_visual>` syntax
