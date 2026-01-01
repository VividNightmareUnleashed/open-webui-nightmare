# Nano Banana Pro

AI-callable tool for generating illustrative images using Google's Gemini API.

## Purpose

This tool allows AI models to autonomously generate images when explaining concepts. The model decides when a visual would be helpful - it's not triggered by user requests.

## Use Cases

- Diagrams showing processes or relationships
- Illustrations of abstract concepts
- Visual examples of objects or scenes
- Educational illustrations

## Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| GEMINI_API_KEY | (required) | Google AI Studio API key |
| MODEL | gemini-2.5-flash-image | Model (or gemini-3-pro-image-preview) |
| ASPECT_RATIO | 16:9 | Default aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4) |

## Setup

1. Get an API key from [Google AI Studio](https://aistudio.google.com/)
2. Add the tool to Open WebUI via the Functions admin panel
3. Configure the `GEMINI_API_KEY` valve
4. Enable the tool for models that should have image generation capability

## Tool Method

### `create_visual(description, aspect_ratio?)`

Generates an image from a text description.

- **description**: Detailed description of the image to generate
- **aspect_ratio**: Optional override (1:1, 16:9, 9:16, 4:3, 3:4)
- **Returns**: Markdown image with inline base64 data URI

## Notes

- Images are returned as inline base64 data URIs for immediate rendering
- Content filtering may block certain prompts
- The model is guided via docstring to use this for educational/explanatory purposes
