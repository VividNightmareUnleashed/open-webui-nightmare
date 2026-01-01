# Changelog

## [0.4.1] - 2026-01-01

- Fixed image display size: now uses embeds with styled img tag (max 512px)
- Images now have rounded corners for cleaner appearance

## [0.4.0] - 2026-01-01

- Fixed 200k token issue: now uploads image to storage and emits via event_emitter
- Image renders inline for user, model only sees "Image generated and displayed"
- Prevents base64 blob from being round-tripped through tool call mechanism

## [0.3.0] - 2025-01-01

- Rewrote to use google-genai SDK instead of raw HTTP
- Fixes response parsing issues (field name casing)
- Changed default model to gemini-2.5-flash-image
- Simplified error handling
- Removed API_BASE_URL and TIMEOUT_SECONDS valves (SDK handles these)

## [0.2.1] - 2025-01-01

- Fixed: responseModalities must include both TEXT and IMAGE (IMAGE-only not supported by Gemini)

## [0.2.0] - 2025-01-01

- Improved docstring with prompting best practices and examples
- Model now receives guidance on natural language prompting, style, composition, colors, and labels

## [0.1.0] - 2025-01-01

- Initial release
- Text-to-image generation via Gemini API
- Configurable aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4)
- Returns inline base64 markdown images
