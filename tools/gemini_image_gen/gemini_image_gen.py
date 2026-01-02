"""
title: Nano Banana Pro
id: nano_banana_pro
version: 0.5.0
description: Leverage Google's latest image generation to create images on request.
license: MIT
requirements: google-genai>=1.0.0
"""

from __future__ import annotations

import io
import logging
import uuid
from typing import Any, Awaitable, Callable, Literal

from google import genai
from google.genai import types
from open_webui.models.files import FileForm, Files
from open_webui.storage.provider import Storage
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class Tools:
    """Nano Banana Pro - image generation tool."""

    class Valves(BaseModel):
        GEMINI_API_KEY: str = Field(
            default="",
            description="Your Google AI Studio API key for Gemini image generation",
        )
        MODEL: str = Field(
            default="gemini-2.5-flash-image",
            description="Gemini model to use (gemini-2.5-flash-image or gemini-3-pro-image-preview)",
        )
        ASPECT_RATIO: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = Field(
            default="16:9",
            description="Default aspect ratio for generated images",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def generate_image(
        self,
        description: str,
        aspect_ratio: str | None = None,
        __user__: dict[str, Any] | None = None,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        """
        Generate an image from a text description.

        PROMPTING BEST PRACTICES:
        - Write natural, descriptive sentences (not keywords)
        - Specify visual style: "oil painting", "photorealistic", "cartoon", "watercolor", "pencil sketch"
        - Describe composition: "centered", "close-up", "wide angle", "from above"
        - Include colors if important: "vibrant sunset colors", "muted earth tones"
        - Mention lighting: "golden hour lighting", "dramatic shadows", "soft diffused light"
        - State the mood/tone: "serene", "energetic", "mysterious", "whimsical"

        EXAMPLE PROMPTS:
        - "A cozy coffee shop interior with warm lighting, wooden furniture, and rain visible through large windows, photorealistic style"
        - "A majestic mountain landscape at sunset with a lake reflection, vibrant orange and purple sky, oil painting style"

        :param description: Natural language description of the image. Be specific about style, composition, colors, and mood.
        :param aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4). Default 16:9.
        :return: Confirmation message.
        """
        # Validate API key
        if not self.valves.GEMINI_API_KEY:
            return "Error: Gemini API key not configured. Please set GEMINI_API_KEY in the tool's Valves settings."

        # Truncate description to avoid token limits (image prompts should be concise)
        MAX_PROMPT_CHARS = 2000
        if len(description) > MAX_PROMPT_CHARS:
            description = description[:MAX_PROMPT_CHARS]
            log.warning(f"Description truncated to {MAX_PROMPT_CHARS} chars")

        # Validate aspect ratio
        valid_ratios = ["1:1", "16:9", "9:16", "4:3", "3:4"]
        selected_ratio = (
            aspect_ratio if aspect_ratio in valid_ratios else self.valves.ASPECT_RATIO
        )

        try:
            # Create client
            client = genai.Client(api_key=self.valves.GEMINI_API_KEY)

            # Build config
            config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=selected_ratio),
            )

            # Generate content
            response = await client.aio.models.generate_content(
                model=self.valves.MODEL,
                contents=description,
                config=config,
            )

            # Check for candidates
            if not response.candidates:
                return "Error: No response generated. The model may have declined the request."

            candidate = response.candidates[0]

            # Check finish reason
            if candidate.finish_reason and candidate.finish_reason not in [
                types.FinishReason.STOP,
                types.FinishReason.MAX_TOKENS,
            ]:
                reason_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
                return f"Image generation blocked ({reason_name}). Try rephrasing your description."

            # Find image in response parts
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.inline_data and part.inline_data.data:
                        mime_type = part.inline_data.mime_type or "image/png"
                        image_bytes = part.inline_data.data
                        if not isinstance(image_bytes, bytes):
                            # If it's a base64 string, decode it
                            import base64

                            image_bytes = base64.b64decode(image_bytes)

                        # Upload to storage and emit event (if we have event_emitter)
                        if __event_emitter__ and __user__:
                            file_id = str(uuid.uuid4())
                            ext = "png" if "png" in mime_type else "jpg"
                            filename = f"nano_banana_{file_id[:8]}.{ext}"
                            stored_filename = f"{file_id}_{filename}"

                            # Upload file to storage
                            _, file_path = Storage.upload_file(
                                io.BytesIO(image_bytes), stored_filename, {}
                            )

                            # Register in database
                            Files.insert_new_file(
                                __user__.get("id", ""),
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

                            # Emit embed with sized image
                            alt_text = description[:80].replace('"', "'")
                            await __event_emitter__(
                                {
                                    "type": "embeds",
                                    "data": {
                                        "embeds": [
                                            f'<img src="/api/v1/files/{file_id}/content" style="max-width: 512px; max-height: 512px; border-radius: 8px;" alt="{alt_text}">'
                                        ]
                                    },
                                }
                            )

                            return "Image generated and displayed."
                        else:
                            # Fallback: return base64 markdown (for testing without event_emitter)
                            import base64

                            b64_data = base64.b64encode(image_bytes).decode("utf-8")
                            alt_text = description[:100].replace('"', "'")
                            return f"![{alt_text}](data:{mime_type};base64,{b64_data})"

            return "Error: Response received but no image data found."

        except Exception as e:
            log.exception("Error during image generation")
            error_msg = str(e)
            if "API key" in error_msg or "401" in error_msg:
                return "Error: Invalid API key. Please check your GEMINI_API_KEY."
            if "quota" in error_msg.lower() or "429" in error_msg:
                return "Error: API quota exceeded. Please try again later."
            return f"Error: {error_msg}"
