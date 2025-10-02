"""
title: Writer Assistant
id: writer_assistant
author: Open WebUI Developer Toolkit
type: tool
description: Allows the model to delegate writing tasks to another specialized model via OpenRouter for content generation, editing, or rewriting.
git_url: https://github.com/jrkropp/open-webui-developer-toolkit.git
required_open_webui_version: 0.6.10
version: 0.1.0
"""

from __future__ import annotations
import aiohttp
import logging
import json
import os
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(
            default=os.getenv("OPENROUTER_API_KEY", ""),
            description="Your OpenRouter API key. Can also be set via OPENROUTER_API_KEY environment variable."
        )
        OPENROUTER_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="OpenRouter API base URL"
        )
        TARGET_MODEL: str = Field(
            default="anthropic/claude-3.5-sonnet",
            description="The model to use for writing tasks (OpenRouter model ID, e.g., 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o')"
        )
        SYSTEM_PROMPT: str = Field(
            default="You are a professional writing assistant. Follow the user's instructions precisely and provide high-quality written content. Be concise and focused.",
            description="System prompt for the writer model"
        )
        DEBUG: bool = Field(
            default=False,
            description="Show debug information in logs (useful when troubleshooting)"
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_or_create_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session for API calls."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def write_content(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> str:
        """
        Delegate a writing task to a specialized writing model.

        CRITICAL: Only call this tool when you have COMPLETE information about what to write.
        You can use context from earlier in the conversation. If the user's request is still
        vague or incomplete after considering the full conversation, ask clarifying questions FIRST.

        Examples:
        - "Write me an email" with NO context → Ask: What should the email be about?
        - "Write me an email" AFTER discussing a topic → Use conversation context, call tool ✓
        - "Can you write something?" → Ask: What would you like me to write?
        - "Rewrite this to be better" → Ask: What style or tone would you prefer?

        Use this tool ONLY when the user clearly wants you to produce:
        - Emails, letters, or formal correspondence
        - Creative writing (stories, poems, scripts, articles)
        - Content rewriting, paraphrasing, or style adjustments
        - Marketing copy, product descriptions, or announcements
        - Professional documents that require polished writing

        Do NOT use this tool for:
        - Casual conversation about writing (e.g., "How do I write better?")
        - Explaining how to write something (e.g., "What should I write in an email?")
        - Code or technical implementation (even if they say "write code")
        - Analysis, research, or explanations
        - Quick replies or conversational responses
        - Answering questions or providing information
        - When the user is discussing writing but hasn't asked you to write anything

        The tool returns professionally written content. You MUST present the
        returned content to the user VERBATIM without modification.

        Args:
            prompt: Clear writing instruction. Include all necessary context,
                   desired style, tone, and any content to work with.
                   Examples:
                   - "Write a professional thank you email to a client"
                   - "Rewrite this in a casual tone: [text]"
                   - "Create a product announcement for our new app, casual and friendly style"
            __event_emitter__: Optional event emitter for status updates

        Returns:
            str: The written content from the specialized model, followed by
                 instructions to present it verbatim to the user.
        """

        # Validate API key
        if not self.valves.OPENROUTER_API_KEY:
            error_msg = "OPENROUTER_API_KEY is not set. Please configure it in the tool valves or environment variables."
            self.logger.error(error_msg)
            return f"Error: {error_msg}"

        if self.valves.DEBUG:
            self.logger.info("Writer Assistant invoked with prompt: %s", prompt[:100])
            self.logger.info("Target model: %s", self.valves.TARGET_MODEL)

        # Build request body for OpenRouter
        request_body = {
            "model": self.valves.TARGET_MODEL,
            "messages": [
                {"role": "system", "content": self.valves.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
        }

        if self.valves.DEBUG:
            self.logger.debug("Request body: %s", json.dumps(request_body, indent=2))

        # Show "Writing..." status to user
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "Writing...",
                    "done": False
                }
            })

        # Make API call to OpenRouter
        try:
            session = await self._get_or_create_session()
            headers = {
                "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/jrkropp/open-webui-developer-toolkit",  # Optional but recommended
                "X-Title": "Open WebUI Writer Assistant"  # Optional but recommended
            }

            url = f"{self.valves.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"

            async with session.post(url, json=request_body, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"OpenRouter API error (status {response.status}): {error_text}"
                    self.logger.error(error_msg)

                    # Show "Writing failed." status to user
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {
                                "description": "Writing failed.",
                                "done": True
                            }
                        })

                    return f"Error: {error_msg}"

                response_data = await response.json()

                if self.valves.DEBUG:
                    self.logger.debug("Response data: %s", json.dumps(response_data, indent=2))

                # Extract the content from the response
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    message = response_data["choices"][0].get("message", {})
                    content_result = message.get("content", "")

                    # Show "Writing complete." status to user
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {
                                "description": "Writing complete.",
                                "done": True
                            }
                        })

                    return content_result.strip()
                else:
                    error_msg = "No content in response from OpenRouter"
                    self.logger.error(error_msg)

                    # Show "Writing failed." status to user
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {
                                "description": "Writing failed.",
                                "done": True
                            }
                        })

                    return f"Error: {error_msg}"

        except aiohttp.ClientError as e:
            error_msg = f"Network error calling OpenRouter: {e}"
            self.logger.error(error_msg)

            # Show "Writing failed." status to user
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Writing failed.",
                        "done": True
                    }
                })

            return f"Error: {error_msg}"
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse OpenRouter response: {e}"
            self.logger.error(error_msg)

            # Show "Writing failed." status to user
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Writing failed.",
                        "done": True
                    }
                })

            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            self.logger.error(error_msg)
            self.logger.exception("Full traceback:")

            # Show "Writing failed." status to user
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Writing failed.",
                        "done": True
                    }
                })

            return f"Error: {error_msg}"

    async def __del__(self):
        """Cleanup aiohttp session when tool is destroyed."""
        if self.session and not self.session.closed:
            await self.session.close()
