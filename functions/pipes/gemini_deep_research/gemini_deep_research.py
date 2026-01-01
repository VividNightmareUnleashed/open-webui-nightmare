"""
title: Gemini Deep Research
id: gemini_deep_research
description: Google Gemini Deep Research integration with real-time thinking summaries. Uses SSE streaming for live research progress.
author: openwebuidev
author_url: https://github.com/openwebuidev
license: MIT
required_open_webui_version: 0.6.10
version: 0.3.1
requirements: aiohttp
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Literal

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

# Setting auditable=False avoids duplicate output for log levels
log = logger.bind(auditable=False)

# API Configuration
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"

# Interaction statuses (lowercase as returned by API)
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


class Pipe:
    """
    Gemini Deep Research Pipe.

    Integrates Google's Deep Research agent via the Interactions API.
    Features:
    - SSE streaming with real-time thinking summaries
    - Follow-up questions using previous_interaction_id
    - Fallback to polling if SSE fails
    """

    class Valves(BaseModel):
        """Configuration valves for the Deep Research pipe."""

        GOOGLE_API_KEY: str = Field(
            default="",
            description="Google AI API Key for Gemini Deep Research. Required.",
        )
        POLLING_INTERVAL: float = Field(
            default=10.0,
            description="Seconds between status polls (fallback mode only).",
        )
        MAX_RESEARCH_TIME: int = Field(
            default=3600,
            description="Maximum research time in seconds (default: 60 minutes).",
        )
        CONNECTION_TIMEOUT: int = Field(
            default=120,
            description="HTTP connection timeout in seconds.",
        )
        USE_STREAMING: bool = Field(
            default=True,
            description="Use SSE streaming for real-time thinking summaries. Disable to use polling.",
        )
        LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"] = Field(
            default="INFO",
            description="Logging verbosity level.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        """Return available Deep Research models."""
        return [
            {
                "id": DEEP_RESEARCH_AGENT,
                "name": "Gemini Deep Research",
            }
        ]

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        __metadata__: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process a Deep Research request and stream the response.

        Args:
            body: Request payload containing messages
            __user__: Current user info
            __event_emitter__: Async callback for emitting events to frontend
            __metadata__: Request metadata including features and previous_interaction_id

        Yields:
            Response text chunks
        """
        # Validate API key
        api_key = self.valves.GOOGLE_API_KEY
        if not api_key:
            error_msg = "Google API Key not configured. Please set GOOGLE_API_KEY in the pipe valves."
            await self._emit_error(__event_emitter__, error_msg)
            yield f"\n\n**Error:** {error_msg}"
            return

        # Extract research query from messages
        try:
            query = self._extract_research_query(body.get("messages", []))
        except ValueError as e:
            await self._emit_error(__event_emitter__, str(e))
            yield f"\n\n**Error:** {e}"
            return

        # Check for previous interaction (follow-up support)
        previous_interaction_id = None
        if __metadata__:
            features = __metadata__.get("features", {})
            deep_research_features = features.get("gemini_deep_research", {})
            previous_interaction_id = deep_research_features.get("previous_interaction_id")

        log.info(f"Starting Deep Research for query: {query[:100]}...")
        if previous_interaction_id:
            log.info(f"Follow-up on interaction: {previous_interaction_id}")

        # Emit initial status
        await self._emit_status(__event_emitter__, "Initializing Deep Research...", done=False)

        # Run research
        interaction_id = None
        try:
            if self.valves.USE_STREAMING:
                # Try SSE streaming first
                async for chunk in self._run_research_streaming(
                    query=query,
                    api_key=api_key,
                    event_emitter=__event_emitter__,
                    previous_interaction_id=previous_interaction_id,
                ):
                    if chunk.startswith("__INTERACTION_ID__:"):
                        interaction_id = chunk.split(":", 1)[1]
                    else:
                        yield chunk
            else:
                # Use polling mode
                async for chunk in self._run_research_polling(
                    query=query,
                    api_key=api_key,
                    event_emitter=__event_emitter__,
                    previous_interaction_id=previous_interaction_id,
                ):
                    if chunk.startswith("__INTERACTION_ID__:"):
                        interaction_id = chunk.split(":", 1)[1]
                    else:
                        yield chunk

            # Store interaction_id for follow-ups via metadata
            if interaction_id and __metadata__:
                features = __metadata__.setdefault("features", {})
                deep_research = features.setdefault("gemini_deep_research", {})
                deep_research["last_interaction_id"] = interaction_id

            # Emit completion status
            await self._emit_status(__event_emitter__, "Research complete", done=True)

        except asyncio.CancelledError:
            log.info("Research cancelled by user")
            await self._emit_status(__event_emitter__, "Research cancelled", done=True)
            raise

        except Exception as e:
            log.exception(f"Deep Research failed: {e}")
            await self._emit_error(__event_emitter__, f"Research failed: {e}")
            yield f"\n\n**Error:** Research failed: {e}"

    def _extract_research_query(self, messages: list[dict[str, Any]]) -> str:
        """
        Extract the research query from conversation messages.

        Uses the last user message as the primary research topic.
        Optionally includes system message as context.
        """
        system_content = ""
        last_user_message = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle multimodal content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = " ".join(text_parts)

            if role == "system":
                system_content = content
            elif role == "user":
                last_user_message = content

        if not last_user_message:
            raise ValueError("No user message found for research query")

        # Combine system context with user query if present
        if system_content:
            return f"Context: {system_content}\n\nResearch topic: {last_user_message}"

        return last_user_message

    async def _run_research_streaming(
        self,
        query: str,
        api_key: str,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        previous_interaction_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Run Deep Research with SSE streaming and thinking summaries.

        Args:
            query: Research query
            api_key: Google API key
            event_emitter: Event emitter callback
            previous_interaction_id: ID of previous interaction for follow-ups

        Yields:
            Response text chunks (prefixed with __INTERACTION_ID__: for ID)
        """
        url = f"{BASE_URL}/interactions"
        params = {"key": api_key, "alt": "sse"}

        payload: dict[str, Any] = {
            "input": query,
            "agent": DEEP_RESEARCH_AGENT,
            "background": True,
            "store": True,
            "stream": True,
            "agent_config": {
                "type": "deep-research",
                "thinking_summaries": "auto",
            },
        }

        # Add previous interaction for follow-ups
        if previous_interaction_id:
            payload["previous_interaction_id"] = previous_interaction_id

        log.debug(f"Creating streaming interaction: {url}")

        # Use longer timeout for SSE - research can take up to 60 minutes
        timeout = aiohttp.ClientTimeout(
            total=self.valves.MAX_RESEARCH_TIME,
            connect=self.valves.CONNECTION_TIMEOUT,
        )

        interaction_id = None

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, params=params, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"Failed to create streaming interaction: {resp.status} - {error_text}")

                    await self._emit_status(event_emitter, "Research started", done=False)

                    # Parse SSE stream - Google puts event_type inside JSON data
                    async for line in resp.content:
                        line_str = line.decode("utf-8").strip()

                        # Skip empty lines and non-data lines
                        if not line_str or not line_str.startswith("data:"):
                            continue

                        json_str = line_str[5:].strip()  # Remove "data:" prefix
                        if not json_str:
                            continue

                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError:
                            log.debug(f"Skipping non-JSON SSE line: {json_str[:100]}")
                            continue

                        event_type = data.get("event_type", "")
                        log.debug(f"SSE event: {event_type}")

                        if event_type == "interaction.start":
                            # Capture interaction ID for follow-ups
                            interaction_id = data.get("interaction", {}).get("id", "")
                            if interaction_id:
                                log.info(f"Research started: {interaction_id}")
                                yield f"__INTERACTION_ID__:{interaction_id}"
                            await self._emit_status(event_emitter, "Research in progress", done=False)

                        elif event_type == "content.delta":
                            delta = data.get("delta", {})
                            delta_type = delta.get("type", "")

                            if delta_type == "thought_summary":
                                # Research progress updates - try multiple locations
                                summary = (
                                    delta.get("content", {}).get("text", "")
                                    or delta.get("text", "")
                                    or delta.get("summary", "")
                                    or ""
                                )
                                # Filter out "..." placeholders, require actual text
                                # Strip markdown, whitespace, and trailing dots
                                cleaned = summary.replace("**", "").strip().rstrip(".")
                                if cleaned and any(c.isalnum() for c in cleaned):
                                    await self._emit_status(event_emitter, cleaned[:100], done=False)

                            else:
                                # Text content - try multiple locations
                                text = (
                                    delta.get("text", "")
                                    or delta.get("content", {}).get("text", "")
                                    or delta.get("message", "")
                                    or ""
                                )
                                if text:
                                    yield text

                        elif event_type == "interaction.complete":
                            log.info("Research completed via SSE")
                            # Try to extract final content from outputs
                            interaction = data.get("interaction", {})
                            outputs = interaction.get("outputs", [])
                            for output in outputs:
                                if isinstance(output, dict):
                                    text = output.get("text", "")
                                    if text:
                                        yield text
                                elif isinstance(output, str):
                                    yield output

                        elif event_type == "error":
                            error_msg = data.get("error", {}).get("message", "Unknown error")
                            raise Exception(f"Research error: {error_msg}")

                        elif event_type:
                            # Log unhandled event types for debugging
                            log.debug(f"Unhandled event type: {event_type}")

        except aiohttp.ClientError as e:
            log.warning(f"SSE streaming failed, falling back to polling: {e}")
            # Fallback to polling
            async for chunk in self._run_research_polling(
                query=query,
                api_key=api_key,
                event_emitter=event_emitter,
                previous_interaction_id=previous_interaction_id,
            ):
                yield chunk

    async def _run_research_polling(
        self,
        query: str,
        api_key: str,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        previous_interaction_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Run Deep Research with polling (fallback mode).

        Args:
            query: Research query
            api_key: Google API key
            event_emitter: Event emitter callback
            previous_interaction_id: ID of previous interaction for follow-ups

        Yields:
            Response text chunks
        """
        timeout = aiohttp.ClientTimeout(total=self.valves.CONNECTION_TIMEOUT)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Create the research interaction
            await self._emit_status(event_emitter, "Starting research...", done=False)

            url = f"{BASE_URL}/interactions"
            params = {"key": api_key}
            payload: dict[str, Any] = {
                "input": query,
                "agent": DEEP_RESEARCH_AGENT,
                "background": True,
                "store": True,
            }

            # Add previous interaction for follow-ups
            if previous_interaction_id:
                payload["previous_interaction_id"] = previous_interaction_id

            log.debug(f"Creating polling interaction: {url}")

            async with session.post(url, params=params, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Failed to create interaction: {resp.status} - {error_text}")
                interaction = await resp.json()

            interaction_id = interaction.get("id")
            if not interaction_id:
                raise Exception("No interaction ID returned from API")

            log.info(f"Research started with interaction_id: {interaction_id}")
            yield f"__INTERACTION_ID__:{interaction_id}"
            await self._emit_status(event_emitter, "Research in progress...", done=False)

            # Poll for completion
            start_time = time.monotonic()
            poll_count = 0

            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > self.valves.MAX_RESEARCH_TIME:
                    raise Exception("Research exceeded maximum time limit")

                # Get current status
                get_url = f"{BASE_URL}/interactions/{interaction_id}"
                try:
                    async with session.get(get_url, params=params) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise Exception(f"Failed to get interaction: {resp.status} - {error_text}")
                        interaction = await resp.json()
                except Exception as e:
                    log.warning(f"Polling error (will retry): {e}")
                    await asyncio.sleep(self.valves.POLLING_INTERVAL)
                    continue

                status = interaction.get("status", "").lower()
                poll_count += 1

                # Update status on every poll (timer update)
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                status_msg = f"Researching... ({minutes}m {seconds}s)"
                await self._emit_status(event_emitter, status_msg, done=False)
                log.debug(f"Status: {status} (poll #{poll_count}, elapsed: {elapsed:.0f}s)")

                # Check terminal states
                if status == STATUS_COMPLETED:
                    log.info(f"Research completed after {elapsed:.0f}s")

                    # Extract output text
                    outputs = interaction.get("outputs", [])
                    if outputs:
                        for output in outputs:
                            if isinstance(output, dict):
                                text = output.get("text", "")
                                if text:
                                    yield text
                            elif isinstance(output, str):
                                yield output
                    else:
                        # Try alternative response format
                        response = interaction.get("response", {})
                        if isinstance(response, dict):
                            text = response.get("text", "")
                            if text:
                                yield text

                    return

                elif status == STATUS_FAILED:
                    error = interaction.get("error", {})
                    error_msg = error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
                    raise Exception(f"Research failed: {error_msg}")

                elif status == STATUS_CANCELLED:
                    raise Exception("Research was cancelled")

                # Wait before next poll
                await asyncio.sleep(self.valves.POLLING_INTERVAL)

    async def _emit_status(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        description: str,
        done: bool = False,
    ) -> None:
        """Emit status update to frontend."""
        if not event_emitter:
            return

        log.debug(f"Status: {description} (done={done})")

        try:
            await event_emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )
        except Exception as e:
            log.warning(f"Failed to emit status: {e}")

    async def _emit_error(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        error_message: str,
    ) -> None:
        """Emit error event to frontend."""
        if not event_emitter:
            return

        log.error(f"Emitting error: {error_message}")

        try:
            await event_emitter(
                {
                    "type": "chat:completion",
                    "data": {
                        "content": "",
                        "done": True,
                        "error": {"message": error_message},
                    },
                }
            )
        except Exception as e:
            log.warning(f"Failed to emit error: {e}")
