"""
title: Gemini Deep Research
id: gemini_deep_research
description: Google Gemini Deep Research integration with real-time thinking summaries. Uses SSE streaming for live research progress.
author: openwebuidev
author_url: https://github.com/openwebuidev
license: MIT
required_open_webui_version: 0.6.10
version: 0.3.3
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

SSE_REPLAY_PARAM = "last_event_id"


class GeminiAPIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status} - {message}")
        self.status = status
        self.message = message


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
        DEADLINE_RETRIES: int = Field(
            default=10,
            description=(
                "Number of reconnection attempts when the SSE stream is interrupted "
                "(e.g., Google stream deadline expiry). Set to 0 to disable."
            ),
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

        interaction_id = None
        try:
            if self.valves.USE_STREAMING:
                # Stream via SSE with automatic reconnection/resume.
                async for chunk in self._run_research_streaming(
                    query=query,
                    api_key=api_key,
                    event_emitter=__event_emitter__,
                    previous_interaction_id=previous_interaction_id,
                    metadata=__metadata__,
                ):
                    if chunk.startswith("__INTERACTION_ID__:"):
                        interaction_id = chunk.split(":", 1)[1]
                    else:
                        yield chunk
            else:
                # Use polling mode.
                async for chunk in self._run_research_polling(
                    query=query,
                    api_key=api_key,
                    event_emitter=__event_emitter__,
                    previous_interaction_id=previous_interaction_id,
                    metadata=__metadata__,
                ):
                    if chunk.startswith("__INTERACTION_ID__:"):
                        interaction_id = chunk.split(":", 1)[1]
                    else:
                        yield chunk

            # Emit completion status and notification
            await self._emit_status(__event_emitter__, "Research complete", done=True)
            await self._emit_notification(__event_emitter__, "Deep Research complete", level="success")
            return

        except asyncio.CancelledError:
            log.info("Research cancelled by user")
            await self._emit_status(__event_emitter__, "Research cancelled", done=True)
            raise

        except Exception as e:
            log.exception(f"Deep Research failed: {e}")
            error_str = str(e)
            if self._is_deadline_error(error_str):
                error_msg = (
                    "Research timed out on Google's side.\n\n"
                    "**Try:**\n"
                    "- Breaking into smaller, focused questions\n"
                    "- Using follow-up questions for additional depth\n"
                    "- Simplifying the research scope"
                )
                await self._emit_error(__event_emitter__, "Research timed out")
                yield f"\n\n**Error:** {error_msg}"
                return

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
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Run Deep Research with SSE streaming and thinking summaries.

        This implementation uses background execution and streams events via
        `GET /interactions/{id}?stream=true&alt=sse`, allowing automatic reconnection
        and resumption using `last_event_id` if the stream is interrupted.

        Args:
            query: Research query
            api_key: Google API key
            event_emitter: Event emitter callback
            previous_interaction_id: ID of previous interaction for follow-ups

        Yields:
            Response text chunks (prefixed with __INTERACTION_ID__: for ID)
        """
        # Use long timeouts for SSE - research can take up to 60 minutes.
        # Important: set sock_read to avoid default per-read timeouts during long gaps.
        timeout = aiohttp.ClientTimeout(
            total=self.valves.MAX_RESEARCH_TIME,
            connect=self.valves.CONNECTION_TIMEOUT,
            sock_read=self.valves.MAX_RESEARCH_TIME,
        )

        start_time = time.monotonic()
        last_event_id: str | None = None
        streamed_tail = ""
        consecutive_interrupts = 0

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1) Start the research task in the background and capture interaction_id immediately.
            interaction_id = await self._create_interaction(
                session,
                api_key=api_key,
                query=query,
                previous_interaction_id=previous_interaction_id,
            )

            log.info(f"Research started with interaction_id: {interaction_id}")
            yield f"__INTERACTION_ID__:{interaction_id}"
            self._persist_last_interaction_id(metadata, interaction_id)
            await self._emit_status(event_emitter, "Research started", done=False)

            # 2) Stream progress and content; reconnect/resume on disconnect.
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > self.valves.MAX_RESEARCH_TIME:
                    raise Exception("Research exceeded maximum time limit")

                try:
                    completed = False
                    async for data in self._stream_interaction_sse(
                        session,
                        api_key=api_key,
                        interaction_id=interaction_id,
                        last_event_id=last_event_id,
                    ):
                        consecutive_interrupts = 0
                        event_id = data.get("event_id") or data.get("eventId")
                        if event_id is not None:
                            last_event_id = str(event_id)

                        event_type = data.get("event_type", "")
                        log.debug(f"SSE event: {event_type}")

                        if event_type == "interaction.start":
                            await self._emit_status(event_emitter, "Research in progress", done=False)
                            continue

                        if event_type == "content.delta":
                            delta = data.get("delta", {})
                            delta_type = delta.get("type", "")

                            if delta_type == "thought_summary":
                                summary = (
                                    delta.get("content", {}).get("text", "")
                                    or delta.get("text", "")
                                    or delta.get("summary", "")
                                    or ""
                                )
                                cleaned = summary.replace("**", "").strip().rstrip(".")
                                if cleaned and any(c.isalnum() for c in cleaned):
                                    await self._emit_status(event_emitter, cleaned[:100], done=False)
                                continue

                            text = (
                                delta.get("text", "")
                                or delta.get("content", {}).get("text", "")
                                or delta.get("message", "")
                                or ""
                            )
                            if text:
                                streamed_tail = (streamed_tail + text)[-8000:]
                                yield text
                            continue

                        if event_type == "interaction.complete":
                            log.info("Research completed via SSE")
                            completed = True
                            break

                        if event_type == "error":
                            error_msg = data.get("error", {}).get("message", "Unknown error")
                            # Some "deadline" errors are stream interruptions; verify status before failing.
                            if self._is_deadline_error(error_msg):
                                log.warning(f"SSE reported deadline error, will resume: {error_msg}")
                                break
                            raise Exception(f"Research error: {error_msg}")

                    if completed:
                        interaction = await self._get_interaction(session, api_key=api_key, interaction_id=interaction_id)
                        async for text in self._yield_final_outputs(interaction, streamed_tail=streamed_tail):
                            yield text
                        return

                    # Stream ended or broke (disconnect/interruption). Check current status.
                    interaction = await self._get_interaction(session, api_key=api_key, interaction_id=interaction_id)
                    status = interaction.get("status", "").lower()
                    if status == STATUS_COMPLETED:
                        async for text in self._yield_final_outputs(interaction, streamed_tail=streamed_tail):
                            yield text
                        return
                    if status == STATUS_FAILED:
                        error = interaction.get("error", {})
                        error_msg = error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
                        raise Exception(f"Research failed: {error_msg}")
                    if status == STATUS_CANCELLED:
                        raise Exception("Research was cancelled")

                    consecutive_interrupts += 1
                    if consecutive_interrupts > self.valves.DEADLINE_RETRIES:
                        log.warning("SSE stream repeatedly interrupted; switching to polling.")
                        await self._emit_status(event_emitter, "Stream interrupted; switching to polling...", done=False)
                        interaction = await self._poll_interaction_until_complete(
                            session,
                            api_key=api_key,
                            interaction_id=interaction_id,
                            event_emitter=event_emitter,
                            start_time=start_time,
                        )
                        async for text in self._yield_final_outputs(interaction, streamed_tail=streamed_tail):
                            yield text
                        return

                    await self._emit_status(event_emitter, "Connection lost; resuming...", done=False)
                    await asyncio.sleep(min(2**consecutive_interrupts, 10))

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    consecutive_interrupts += 1
                    log.warning(f"SSE stream interrupted ({consecutive_interrupts}/{self.valves.DEADLINE_RETRIES}): {e}")
                    if consecutive_interrupts > self.valves.DEADLINE_RETRIES:
                        await self._emit_status(event_emitter, "Stream interrupted; switching to polling...", done=False)
                        interaction = await self._poll_interaction_until_complete(
                            session,
                            api_key=api_key,
                            interaction_id=interaction_id,
                            event_emitter=event_emitter,
                            start_time=start_time,
                        )
                        async for text in self._yield_final_outputs(interaction, streamed_tail=streamed_tail):
                            yield text
                        return

                    await self._emit_status(event_emitter, "Stream interrupted; retrying...", done=False)
                    await asyncio.sleep(min(2**consecutive_interrupts, 10))

    async def _run_research_polling(
        self,
        query: str,
        api_key: str,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        previous_interaction_id: str | None = None,
        metadata: dict[str, Any] | None = None,
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
            start_time = time.monotonic()
            # Create the research interaction
            await self._emit_status(event_emitter, "Starting research...", done=False)
            interaction_id = await self._create_interaction(
                session,
                api_key=api_key,
                query=query,
                previous_interaction_id=previous_interaction_id,
            )
            log.info(f"Research started with interaction_id: {interaction_id}")
            yield f"__INTERACTION_ID__:{interaction_id}"
            self._persist_last_interaction_id(metadata, interaction_id)

            interaction = await self._poll_interaction_until_complete(
                session,
                api_key=api_key,
                interaction_id=interaction_id,
                event_emitter=event_emitter,
                start_time=start_time,
            )
            async for text in self._yield_final_outputs(interaction, streamed_tail=""):
                yield text
            return

    async def _create_interaction(
        self,
        session: aiohttp.ClientSession,
        *,
        api_key: str,
        query: str,
        previous_interaction_id: str | None,
    ) -> str:
        url = f"{BASE_URL}/interactions"
        params = {"key": api_key}
        payload: dict[str, Any] = {
            "input": query,
            "agent": DEEP_RESEARCH_AGENT,
            "background": True,
            "store": True,
            # Ensures real-time thought summaries are enabled for streaming/replay.
            "agent_config": {
                "type": "deep-research",
                "thinking_summaries": "auto",
            },
        }

        if previous_interaction_id:
            payload["previous_interaction_id"] = previous_interaction_id

        log.debug(f"Creating interaction: {url}")

        async with session.post(url, params=params, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise GeminiAPIError(resp.status, f"Failed to create interaction: {error_text}")
            interaction = await resp.json()

        interaction_id = interaction.get("id")
        if not interaction_id:
            raise Exception("No interaction ID returned from API")

        return str(interaction_id)

    async def _stream_interaction_sse(
        self,
        session: aiohttp.ClientSession,
        *,
        api_key: str,
        interaction_id: str,
        last_event_id: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        url = f"{BASE_URL}/interactions/{interaction_id}"
        params: dict[str, str] = {"key": api_key, "alt": "sse", "stream": "true"}
        if last_event_id:
            params[SSE_REPLAY_PARAM] = last_event_id

        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise GeminiAPIError(resp.status, f"Failed to stream interaction: {error_text}")

            buffer = ""
            async for chunk in resp.content:
                buffer += chunk.decode("utf-8")

                while "\n" in buffer:
                    line_str, buffer = buffer.split("\n", 1)
                    line_str = line_str.strip()

                    if not line_str or not line_str.startswith("data:"):
                        continue

                    json_str = line_str[5:].strip()
                    if not json_str:
                        continue

                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        log.debug(f"Skipping non-JSON SSE line: {json_str[:100]}")
                        continue

                    if isinstance(data, dict):
                        yield data

    async def _get_interaction(
        self,
        session: aiohttp.ClientSession,
        *,
        api_key: str,
        interaction_id: str,
    ) -> dict[str, Any]:
        url = f"{BASE_URL}/interactions/{interaction_id}"
        params = {"key": api_key}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise GeminiAPIError(resp.status, f"Failed to get interaction: {error_text}")
            data = await resp.json()
            if not isinstance(data, dict):
                raise Exception("Invalid interaction response")
            return data

    async def _poll_interaction_until_complete(
        self,
        session: aiohttp.ClientSession,
        *,
        api_key: str,
        interaction_id: str,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        start_time: float,
    ) -> dict[str, Any]:
        poll_count = 0
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > self.valves.MAX_RESEARCH_TIME:
                raise Exception("Research exceeded maximum time limit")

            try:
                interaction = await self._get_interaction(session, api_key=api_key, interaction_id=interaction_id)
            except Exception as e:
                log.warning(f"Polling error (will retry): {e}")
                await asyncio.sleep(self.valves.POLLING_INTERVAL)
                continue

            status = str(interaction.get("status", "")).lower()
            poll_count += 1

            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            status_msg = f"Researching... ({minutes}m {seconds}s)"
            await self._emit_status(event_emitter, status_msg, done=False)
            log.debug(f"Status: {status} (poll #{poll_count}, elapsed: {elapsed:.0f}s)")

            if status == STATUS_COMPLETED:
                log.info(f"Research completed after {elapsed:.0f}s")
                return interaction
            if status == STATUS_FAILED:
                error = interaction.get("error", {})
                error_msg = error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
                raise Exception(f"Research failed: {error_msg}")
            if status == STATUS_CANCELLED:
                raise Exception("Research was cancelled")

            await asyncio.sleep(self.valves.POLLING_INTERVAL)

    async def _yield_final_outputs(
        self,
        interaction: dict[str, Any],
        *,
        streamed_tail: str,
    ) -> AsyncGenerator[str, None]:
        texts: list[str] = []
        outputs = interaction.get("outputs", [])
        if isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, dict):
                    text = output.get("text", "")
                    if text:
                        texts.append(str(text))
                elif isinstance(output, str):
                    texts.append(output)

        if not texts:
            response = interaction.get("response", {})
            if isinstance(response, dict):
                text = response.get("text", "")
                if text:
                    texts.append(str(text))

        if not texts:
            return

        final_text = "\n".join(texts)

        if not streamed_tail:
            yield final_text
            return

        # Prefer the earliest match to avoid accidentally skipping content if the tail repeats.
        idx = final_text.find(streamed_tail)
        if idx != -1:
            remainder = final_text[idx + len(streamed_tail) :]
            if remainder:
                yield remainder
            return

        # Fallback: if we can't reliably dedupe, yield the full output.
        yield final_text

    def _persist_last_interaction_id(self, metadata: dict[str, Any] | None, interaction_id: str) -> None:
        if not metadata:
            return
        features = metadata.setdefault("features", {})
        deep_research = features.setdefault("gemini_deep_research", {})
        deep_research["last_interaction_id"] = interaction_id

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

    async def _emit_notification(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        content: str,
        *,
        level: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        """Emit a toast notification to the UI."""
        if not event_emitter:
            return

        try:
            await event_emitter({
                "type": "notification",
                "data": {"type": level, "content": content}
            })
        except Exception as e:
            log.warning(f"Failed to emit notification: {e}")

    def _is_deadline_error(self, error_msg: str) -> bool:
        """Check if error is a Google deadline timeout."""
        msg = error_msg.lower()
        return "deadline expired" in msg or "deadline exceeded" in msg
