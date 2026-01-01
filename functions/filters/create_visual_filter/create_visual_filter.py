"""
title: Illustrations
id: create_visual_filter
version: 0.7.0
description: Allow the model to generate diagrams, illustrations, and visuals to help explain things.
license: MIT
requirements: google-genai>=1.0.0
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import uuid
from typing import Any, Awaitable, Callable, Literal

from google import genai
from google.genai import types
from open_webui.models.files import FileForm, Files
from open_webui.storage.provider import Storage
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Pattern to match <create_visual>...</create_visual> tags
CREATE_VISUAL_PATTERN = re.compile(
    r"<create_visual(?:\s+ratio=[\"']([^\"']+)[\"'])?\s*>(.*?)</create_visual>",
    re.DOTALL | re.IGNORECASE,
)


class Filter:
    """Filter that executes <create_visual> tags inline during model output."""

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter priority (lower runs first)",
        )
        GEMINI_API_KEY: str = Field(
            default="",
            description="Google AI Studio API key for Gemini image generation",
        )
        MODEL: str = Field(
            default="gemini-2.5-flash-image",
            description="Gemini model for image generation",
        )
        DEFAULT_ASPECT_RATIO: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = Field(
            default="16:9",
            description="Default aspect ratio for generated images",
        )
        MAX_WIDTH: int = Field(
            default=512,
            description="Maximum image width in pixels for display",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAtOTYwIDk2MCA5NjAiIHdpZHRoPSIyNHB4IiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwYXRoIGQ9Ik0xMjAtMTIwcS04IDAtMTUuNS0zLjVUOTItMTMycS0xMi0xMS0xMi0yNy41VDkxLTE4OHExOS0yMCA0NS41LTI0dDUzLjUgMXE4IDIgMTUgNHQxMi0zcTYtNiA0LjUtMTV0LTMuNS0xN3EtNC0yNy0xLTUzLjV0MjItNDYuNXExOS0yMCA0NS41LTI0dDUzLjUgMXE4IDIgMTUuNSA0dDEyLjUtM3E2LTYgNC0xNXQtNC0xN3EtNC0yNy0xLTUzLjV0MjItNDYuNXExOS0yMCA0NS41LTI0dDUzLjUgMXE4IDIgMTUuNSA0dDEyLjUtM3E2LTYgNC0xNXQtNC0xN3EtNC0yNy0uNS01My41VDUzNi02NTBxMTktMjAgNDUuNS0yNHQ1My41IDFxOCAyIDE1LjUgNHQxMi41LTNxNi02IDQtMTV0LTQtMTdxLTQtMjctLjUtNTMuNVQ2ODUtODA0cTE5LTIwIDQ1LjUtMjR0NTMuNSAxcTggMiAxNS41IDMuNVQ4MTItODI3cTExLTEyIDI3LjUtMTJ0MjguNSAxMXExMiAxMSAxMiAyNy41VDg2OS03NzJxLTE5IDIwLTQ1LjUgMjQuNVQ3NzAtNzQ4cS04LTItMTUuNS00dC0xMi41IDNxLTYgNi00IDE1dDQgMTdxNCAyNyAuNSA1My41VDcyMC02MTdxLTE5IDIwLTQ1LjUgMjR0LTUzLjUtMXEtOC0yLTE1LTR0LTEyIDNxLTYgNi00LjUgMTV0My41IDE3cTQgMjcgMSA1My41VDU3Mi00NjNxLTE5IDE5LTQ1LjUgMjMuNVQ0NzMtNDQwcS04LTItMTUtMy41dC0xMiAzLjVxLTYgNi00LjUgMTQuNVQ0NDUtNDA5cTQgMjcgMSA1My41VDQyNC0zMDlxLTE5IDIwLTQ2IDI0dC01NC0xcS04LTItMTUtMy41dC0xMiAzLjVxLTYgNi00IDE0LjV0NCAxNi41cTQgMjcgLjUgNTMuNVQyNzUtMTU1cS0xOSAyMC00NS41IDI0dC01My41LTFxLTgtMi0xNS0zLjV0LTEyIDMuNXEtNiA2LTEzLjUgOXQtMTUuNSAzWm0xMjAtNDAwcS04MyAwLTE0MS41LTU4LjVUNDAtNzIwcTAtODQgNTguNS0xNDJUMjQwLTkyMHE4NCAwIDE0MiA1OHQ1OCAxNDJxMCA4My01OCAxNDEuNVQyNDAtNTIwWm0wLTgwcTUxIDAgODUuNS0zNXQzNC41LTg1cTAtNTEtMzQuNS04NS41VDI0MC04NDBxLTUwIDAtODUgMzQuNVQxMjAtNzIwcTAgNTAgMzUgODV0ODUgMzVaTTY0MC00MHEtMzMgMC01Ni41LTIzLjVUNTYwLTEyMHYtMjAwcTAtMzMgMjMuNS01Ni41VDY0MC00MDBoMjAwcTMzIDAgNTYuNSAyMy41VDkyMC0zMjB2MjAwcTAgMzMtMjMuNSA1Ni41VDg0MC00MEg2NDBabTAtODBoMjAwdi0yMDBINjQwdjIwMFptMTAwLTEwMFpNMjQwLTcyMFoiLz48L3N2Zz4="

    async def inlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Inject create_visual instructions into system prompt."""
        if not self.valves.GEMINI_API_KEY:
            return body

        instructions = """

## Illustrations

You have access to an inline image generation capability. Use it when visuals would help explain concepts.

### When to Use
- Diagrams showing relationships, processes, or architectures
- Illustrations of abstract concepts that benefit from visualization
- Visual examples of objects, scenes, or UI mockups
- Educational illustrations, charts, or infographics

DO NOT USE FOR: user-requested artwork, portraits of real people, or inappropriate content.

### Syntax
```
<create_visual>Your detailed image description</create_visual>
<create_visual ratio="1:1">Square image description</create_visual>
```
Supported ratios: 1:1, 16:9 (default), 9:16, 4:3, 3:4

### Prompting Best Practices
- Write natural, descriptive sentences (not keywords)
- Specify visual style: "clean minimalist diagram", "colorful illustration", "technical schematic"
- Describe composition: "centered", "left-to-right flow", "hierarchical tree structure"
- Include colors if important: "blue boxes connected by gray arrows"
- Mention text/labels to include: "labeled with 'Input', 'Process', 'Output'"

### Examples
- `<create_visual>A clean flowchart showing input, processing, and output stages connected by arrows, using blue and gray colors on white background</create_visual>`
- `<create_visual ratio="1:1">A friendly icon of a lightbulb with a gear inside, representing innovation</create_visual>`

You can include multiple images in one response by using multiple tags."""

        messages = body.get("messages", [])
        if not messages:
            return body

        # Find or create system message
        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                break

        if system_idx is not None:
            # Append to existing system message
            current_content = messages[system_idx].get("content", "")
            messages[system_idx]["content"] = current_content + instructions
        else:
            # Insert new system message at beginning
            messages.insert(0, {"role": "system", "content": instructions.strip()})

        body["messages"] = messages
        return body

    async def outlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Process model output, replacing <create_visual> tags with generated images."""
        if not self.valves.GEMINI_API_KEY:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Get last assistant message
        last_message = messages[-1]
        if last_message.get("role") != "assistant":
            return body

        content = last_message.get("content", "")
        if not isinstance(content, str):
            return body

        # Find all <create_visual> tags
        matches = list(CREATE_VISUAL_PATTERN.finditer(content))
        if not matches:
            return body

        log.info(f"Found {len(matches)} <create_visual> tags to process")
        total = len(matches)

        # Step 1: Immediately replace all tags with placeholders to hide XML
        placeholder_content = content
        for i, match in enumerate(reversed(matches)):
            placeholder_content = (
                placeholder_content[: match.start()]
                + f"*Generating image {total - i}...*"
                + placeholder_content[match.end() :]
            )

        # Emit updated content and status
        if __event_emitter__:
            await __event_emitter__(
                {"type": "chat:message", "data": {"content": placeholder_content}}
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Generating {total} image{'s' if total > 1 else ''} in parallel...",
                        "done": False,
                    },
                }
            )

        # Step 2: Build tag info and generate all images in parallel
        tag_info = []
        tasks = []
        for match in matches:
            ratio = match.group(1) or self.valves.DEFAULT_ASPECT_RATIO
            description = match.group(2).strip()
            tag_info.append((match.start(), match.end(), description))
            if description:
                tasks.append(self._generate_image(description, ratio, __user__))

        # Generate all images in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Replace all tags with results
        new_content = content
        offset = 0
        result_idx = 0

        for start, end, description in tag_info:
            if not description:
                # Empty tag - just remove it
                replacement = ""
            else:
                result = results[result_idx]
                result_idx += 1
                if isinstance(result, Exception):
                    replacement = f"[Image error: {str(result)[:50]}]"
                else:
                    replacement = str(result)

            new_content = (
                new_content[: start + offset]
                + replacement
                + new_content[end + offset :]
            )
            offset += len(replacement) - (end - start)

        # Emit final content
        if __event_emitter__:
            await __event_emitter__(
                {"type": "chat:message", "data": {"content": new_content}}
            )

        # Clear status
        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": "", "done": True}}
            )

        # Update message content
        messages[-1]["content"] = new_content
        body["messages"] = messages

        return body

    async def _generate_image(
        self,
        description: str,
        aspect_ratio: str,
        user: dict[str, Any] | None,
    ) -> str:
        """Generate image and return markdown image syntax for inline display."""
        # Validate aspect ratio
        valid_ratios = ["1:1", "16:9", "9:16", "4:3", "3:4"]
        if aspect_ratio not in valid_ratios:
            aspect_ratio = self.valves.DEFAULT_ASPECT_RATIO

        # Truncate description
        if len(description) > 2000:
            description = description[:2000]

        try:
            client = genai.Client(api_key=self.valves.GEMINI_API_KEY)

            config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            )

            response = await client.aio.models.generate_content(
                model=self.valves.MODEL,
                contents=description,
                config=config,
            )

            if not response.candidates:
                return f"[Image generation failed: no response]"

            candidate = response.candidates[0]

            if candidate.finish_reason and candidate.finish_reason not in [
                types.FinishReason.STOP,
                types.FinishReason.MAX_TOKENS,
            ]:
                return f"[Image blocked: {candidate.finish_reason}]"

            # Find image in response
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.inline_data and part.inline_data.data:
                        mime_type = part.inline_data.mime_type or "image/png"
                        image_bytes = part.inline_data.data

                        if not isinstance(image_bytes, bytes):
                            import base64
                            image_bytes = base64.b64decode(image_bytes)

                        # Upload to storage
                        file_id = str(uuid.uuid4())
                        ext = "png" if "png" in mime_type else "jpg"
                        filename = f"visual_{file_id[:8]}.{ext}"
                        stored_filename = f"{file_id}_{filename}"

                        _, file_path = Storage.upload_file(
                            io.BytesIO(image_bytes), stored_filename, {}
                        )

                        # Register in database
                        if user:
                            Files.insert_new_file(
                                user.get("id", ""),
                                FileForm(
                                    id=file_id,
                                    filename=filename,
                                    path=file_path,
                                    meta={
                                        "name": filename,
                                        "content_type": mime_type,
                                        "size": len(image_bytes),
                                    },
                                ),
                            )

                        # Return markdown image for inline display
                        # Escape brackets in alt text for markdown
                        alt_text = description[:80].replace("[", "").replace("]", "")
                        return f"![{alt_text}](/api/v1/files/{file_id}/content)"

            return "[Image generation failed: no image in response]"

        except Exception as e:
            log.exception("Error generating image")
            return f"[Image generation error: {str(e)[:100]}]"
