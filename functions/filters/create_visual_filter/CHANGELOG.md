# Changelog

## [0.7.0] - 2026-01-01

- Renamed to "Illustrations"
- Simplified toggle (removed UserValves, uses self.toggle)
- Updated icon to Material Symbols document image

## [0.6.0] - 2026-01-01

- Added toggle button in chat compose area (UserValves)
- Users can enable/disable visual generation per-user

## [0.5.0] - 2026-01-01

- Parallel image generation using asyncio.gather (much faster for multiple images)
- Status shows "Generating N images in parallel..."

## [0.4.0] - 2026-01-01

- Tags now immediately replaced with "*Generating image N...*" placeholders (hides XML from user)
- Images render inline where tags were using markdown syntax
- Real-time content updates as each image completes via chat:message events

## [0.3.0] - 2026-01-01

- Fixed image rendering: now uses embeds event instead of raw HTML in content
- Added status feedback: shows "Generating image 1/N..." during processing
- Images display properly via embeds event

## [0.2.0] - 2026-01-01

- Added inlet method to auto-inject usage instructions into system prompt
- No manual system prompt editing required - filter teaches model the syntax

## [0.1.0] - 2026-01-01

- Initial release
- Outlet filter intercepts `<create_visual>` tags in model output
- Generates images via Gemini API
- Uploads to Open WebUI storage for persistence
- Replaces tags with styled inline `<img>` elements
- Supports optional aspect ratio via `ratio` attribute
