"""
title: Anthropic Manifold
id: anthropic_manifold
author: Based on original by justinh-rahb, christian-taillon, Mark Kazakov, Vincent, NIK-NUB, Snav
description: Full-featured Anthropic Claude API integration with extended thinking, web search, tool calling, and more.
required_open_webui_version: 0.6.10
version: 1.2.2
license: MIT
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import datetime
import inspect
import json
import logging
import os
import re
import sys
from collections import defaultdict, deque
from contextvars import ContextVar
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from urllib.parse import urlparse

import aiohttp
from fastapi import Request
from pydantic import BaseModel, Field

from open_webui.utils.misc import pop_system_message
from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.storage.provider import Storage


# ─────────────────────────────────────────────────────────────────────────────
# 2. Constants & Global Configuration
# ─────────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_FILES_API_URL = "https://api.anthropic.com/v1/files"
FILES_API_BETA_HEADER = "files-api-2025-04-14"

# Feature support by model
FEATURE_SUPPORT = {
    "web_search": {
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    },
    "interleaved_thinking": {
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
    },
    "extended_thinking": {
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    },
    "native_tools": {
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
        "claude-3-5-haiku-latest",
    },
    "code_execution": {
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
    },
    "mcp": {
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
    },
}

# Effort level to thinking budget mapping
EFFORT_TO_BUDGET = {
    "low": 4096,
    "medium": 16000,
    "high": 32000,
}

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "claude-3-5-haiku-latest": "claude-3.5-haiku",
    "claude-opus-4-1-20250805": "claude-4.1-opus",
    "claude-opus-4-5-20251101": "claude-4.5-opus",
    "claude-sonnet-4-5-20250929": "claude-4.5-sonnet",
    "claude-sonnet-4-5-20250929-think": "claude-4.5-sonnet-thinking",
    "claude-haiku-4-5-20251001": "claude-4.5-haiku",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Logging Infrastructure
# ─────────────────────────────────────────────────────────────────────────────
class SessionLogger:
    """Session-aware logger for debugging and troubleshooting."""

    session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
    log_level: ContextVar[int] = ContextVar("log_level", default=logging.INFO)
    logs: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

    @classmethod
    def get_logger(cls, name: str = __name__) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.filters.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        def log_filter(record):
            record.session_id = cls.session_id.get()
            return record.levelno >= cls.log_level.get()

        logger.addFilter(log_filter)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            logging.Formatter("[%(levelname)s] [%(session_id)s] %(message)s")
        )
        logger.addHandler(console)

        return logger


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Pipe Class
# ─────────────────────────────────────────────────────────────────────────────
class Pipe:
    """Anthropic Claude API manifold with full feature support."""

    CACHE_TTL = "1h"
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image

    # ─────────────────────────────────────────────────────────────────────────
    # 4.1 Valves (Configuration)
    # ─────────────────────────────────────────────────────────────────────────
    class Valves(BaseModel):
        """Global configuration options."""

        # API Configuration
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Anthropic API Key",
        )
        MODEL_IDS: str = Field(
            default="claude-3-5-haiku-latest,claude-opus-4-1-20250805,claude-opus-4-5-20251101,claude-sonnet-4-5-20250929,claude-sonnet-4-5-20250929-think",
            description="Comma-separated list of model IDs to expose",
        )

        # Extended Thinking
        ENABLE_THINKING: bool = Field(
            default=True,
            description="Enable extended thinking for supported models",
        )
        THINKING_BUDGET: int = Field(
            default=16000,
            description="Maximum tokens for thinking (1024-32000)",
        )
        DISPLAY_THINKING: bool = Field(
            default=True,
            description="Display thinking process in chat",
        )
        ENABLE_INTERLEAVED_THINKING: bool = Field(
            default=True,
            description="Enable think-act-think-act pattern for tool use",
        )
        EFFORT_LEVEL: Literal["low", "medium", "high", "not_set"] = Field(
            default="not_set",
            description="Thinking effort level (overrides THINKING_BUDGET)",
        )

        # Temperature/Sampling
        CLAUDE_45_USE_TEMPERATURE: bool = Field(
            default=True,
            description="For Claude 4.5: Use temperature (True) or top_p (False)",
        )

        # Web Search
        ENABLE_WEB_SEARCH: bool = Field(
            default=False,
            description="Enable web search (also activates when OpenWebUI toggle is on)",
        )
        WEB_SEARCH_MAX_USES: int = Field(
            default=5,
            description="Maximum web searches per request",
        )
        WEB_SEARCH_ALLOWED_DOMAINS: str = Field(
            default="",
            description="Comma-separated allowed domains (empty = all)",
        )
        WEB_SEARCH_BLOCKED_DOMAINS: str = Field(
            default="",
            description="Comma-separated blocked domains",
        )
        WEB_SEARCH_USER_LOCATION: str = Field(
            default="",
            description='JSON location: {"type":"approximate","country":"US","city":"NYC"}',
        )

        # Tool Calling
        ENABLE_NATIVE_TOOLS: bool = Field(
            default=True,
            description="Use Anthropic native tool calling",
        )
        MAX_TOOL_LOOPS: int = Field(
            default=10,
            description="Maximum tool call iterations per request",
        )
        PARALLEL_TOOL_CALLS: bool = Field(
            default=True,
            description="Execute multiple tool calls in parallel",
        )

        # Code Execution
        ENABLE_CODE_EXECUTION: bool = Field(
            default=False,
            description="Enable Python code execution sandbox",
        )

        # Citations
        ENABLE_CITATIONS: bool = Field(
            default=False,
            description="Enable document citations (cite exact passages from uploaded files)",
        )

        # MCP Connector
        MCP_SERVERS_JSON: str = Field(
            default="",
            description='JSON: [{"url":"...","auth_token":"..."}]',
        )

        # Context Management
        ENABLE_CONTEXT_MANAGEMENT: bool = Field(
            default=False,
            description="Enable automatic context management (clears stale tool calls)",
        )

        # Beta Features (Manual)
        BETA_FEATURES: str = Field(
            default="",
            description="Additional beta headers (comma-separated)",
        )

        # Logging
        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
            default="INFO",
            description="Logging verbosity",
        )

    class UserValves(BaseModel):
        """Per-user overrides."""

        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "INHERIT"] = Field(
            default="INHERIT",
            description="Logging level ('INHERIT' uses global)",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4.2 Constructor
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        self.session: aiohttp.ClientSession | None = None
        self.logger = SessionLogger.get_logger(__name__)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.3 Model Discovery
    # ─────────────────────────────────────────────────────────────────────────
    def pipes(self) -> List[dict]:
        """Return list of available models."""
        model_ids = [m.strip() for m in self.valves.MODEL_IDS.split(",") if m.strip()]
        models = []
        for model_id in model_ids:
            display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
            models.append({"id": model_id, "name": display_name})
        return models

    # ─────────────────────────────────────────────────────────────────────────
    # 4.4 Main Entry Point
    # ─────────────────────────────────────────────────────────────────────────
    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __metadata__: dict[str, Any],
        __tools__: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Process request and stream response."""

        valves = self._merge_valves(
            self.valves,
            self.UserValves.model_validate(__user__.get("valves", {})),
        )

        # Setup session logging
        SessionLogger.session_id.set(__metadata__.get("session_id"))
        SessionLogger.log_level.set(getattr(logging, valves.LOG_LEVEL.upper(), logging.INFO))

        # Get Anthropic-specific features set by companion filter
        # The companion filter intercepts OpenWebUI toggles and stores them here
        anthropic_features = __metadata__.get("features", {}).get("anthropic", {})

        # Resolve __tools__ if it's a coroutine
        if inspect.isawaitable(__tools__):
            __tools__ = await __tools__

        try:
            # Parse request
            system_message, messages = pop_system_message(body["messages"])

            # Extract model info
            raw_model = body.get("model", "")
            model_name = raw_model[raw_model.find(".") + 1:] if "." in raw_model else raw_model
            is_thinking_model = model_name.endswith("-think")
            api_model_name = model_name.replace("-think", "") if is_thinking_model else model_name

            # Check if model supports features
            model_supports_thinking = api_model_name in FEATURE_SUPPORT["extended_thinking"]
            model_supports_tools = api_model_name in FEATURE_SUPPORT["native_tools"]
            model_supports_web_search = api_model_name in FEATURE_SUPPORT["web_search"]
            model_supports_interleaved = api_model_name in FEATURE_SUPPORT["interleaved_thinking"]

            # Determine if thinking is enabled
            # Thinking activates when: model supports it AND (valve is on OR companion filter signals it OR -think model)
            will_enable_thinking = model_supports_thinking and (
                valves.ENABLE_THINKING
                or anthropic_features.get("thinking", False)
                or is_thinking_model
            )

            # Check for web search (valve OR companion filter signal)
            # The companion filter intercepts OpenWebUI's toggle and disables native handling
            web_search_enabled = valves.ENABLE_WEB_SEARCH or anthropic_features.get("web_search", False)

            # Check for code execution (valve OR companion filter signal)
            # The companion filter intercepts OpenWebUI's toggle and disables native handling
            model_supports_code_execution = api_model_name in FEATURE_SUPPORT["code_execution"]
            code_execution_enabled = valves.ENABLE_CODE_EXECUTION or anthropic_features.get("code_execution", False)

            # Check if companion filter signaled to upload documents (bypassing OpenWebUI RAG)
            upload_documents = anthropic_features.get("upload_documents", False)
            uploaded_files: list[dict] = []

            if upload_documents:
                chat_id = __metadata__.get("chat_id", "")
                user_id = __user__.get("id", "")

                if chat_id and chat_id != "local" and user_id:
                    session = await self._get_or_init_session()
                    uploaded_files = await self._process_and_upload_files(
                        session=session,
                        chat_id=chat_id,
                        user_id=user_id,
                        api_key=valves.ANTHROPIC_API_KEY,
                        event_emitter=__event_emitter__,
                    )

                    if uploaded_files:
                        self.logger.info(f"Uploaded {len(uploaded_files)} files to Anthropic Files API")

            # Process messages
            processed_messages = self._process_messages(messages)

            # Add file content blocks to last user message if files were uploaded
            if uploaded_files:
                # Find last user message
                for msg in reversed(processed_messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, str):
                            content = [{"type": "text", "text": content}]
                            msg["content"] = content

                        # Prepend file blocks (so they appear before the text)
                        for file_info in reversed(uploaded_files):
                            block = self._get_file_content_block(
                                file_id=file_info["file_id"],
                                mime_type=file_info["mime_type"],
                                code_execution_enabled=code_execution_enabled and model_supports_code_execution,
                                citations_enabled=valves.ENABLE_CITATIONS,
                                filename=file_info["filename"],
                            )
                            content.insert(0, block)
                        break

            # Prepare system blocks
            system_blocks = self._prepare_system_blocks(system_message)

            # Apply cache control to last message
            self._apply_cache_control_to_last_message(processed_messages)

            # Build payload
            payload = self._build_payload(
                model=api_model_name,
                messages=processed_messages,
                system_blocks=system_blocks,
                body=body,
                valves=valves,
                will_enable_thinking=will_enable_thinking,
                is_thinking_model=is_thinking_model,
            )

            # Build headers
            headers = self._build_headers(
                valves=valves,
                model_id=api_model_name,
                will_enable_thinking=will_enable_thinking,
                has_tools=bool(__tools__) or web_search_enabled or code_execution_enabled,
                model_supports_interleaved=model_supports_interleaved,
                code_execution_enabled=code_execution_enabled and model_supports_code_execution,
            )

            # Add Files API beta header if files were uploaded
            if uploaded_files:
                existing_beta = headers.get("anthropic-beta", "")
                if existing_beta:
                    headers["anthropic-beta"] = f"{existing_beta},{FILES_API_BETA_HEADER}"
                else:
                    headers["anthropic-beta"] = FILES_API_BETA_HEADER

            # Add tools to payload
            tools_list = []

            # Add web search tool if enabled
            if web_search_enabled and model_supports_web_search:
                web_search_tool = self._build_web_search_tool(valves)
                tools_list.append(web_search_tool)

            # Add code execution tool if enabled
            if code_execution_enabled and model_supports_code_execution:
                tools_list.append({
                    "type": "code_execution_20250825",
                    "name": "code_execution",
                })

            # Add OpenWebUI tools if enabled
            if __tools__ and valves.ENABLE_NATIVE_TOOLS and model_supports_tools:
                anthropic_tools = self.transform_tools(__tools__)
                tools_list.extend(anthropic_tools)

            if tools_list:
                payload["tools"] = tools_list

            # Add MCP connector if configured
            mcp_config = self._parse_mcp_config(valves)
            if mcp_config:
                payload["mcp"] = mcp_config

            self.logger.debug("Request payload: %s", json.dumps(payload, indent=2, default=str))

            # Stream response with tool loop
            if body.get("stream", True):
                async for chunk in self._run_streaming_loop(
                    payload=payload,
                    headers=headers,
                    valves=valves,
                    tools=__tools__ if isinstance(__tools__, dict) else {},
                    event_emitter=__event_emitter__,
                    metadata=__metadata__,
                ):
                    yield chunk
            else:
                result = await self._run_non_streaming(
                    payload=payload,
                    headers=headers,
                    valves=valves,
                )
                yield result

        except Exception as e:
            self.logger.exception("Error in pipe: %s", e)
            await self._emit_error(__event_emitter__, str(e))
            yield f"Error: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # 4.5 Message Processing
    # ─────────────────────────────────────────────────────────────────────────
    def _process_messages(self, messages: List[dict]) -> List[dict]:
        """Convert OpenWebUI messages to Anthropic format."""
        processed = []
        total_image_size = 0

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            processed_content = []

            if isinstance(content, list):
                for item in content:
                    item_type = item.get("type", "text")

                    if item_type == "text":
                        processed_content.append({
                            "type": "text",
                            "text": item.get("text", ""),
                        })

                    elif item_type == "image_url":
                        processed_image = self._process_image(item)
                        processed_content.append(processed_image)

                        # Track image size
                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                            total_image_size += image_size
                            if total_image_size > 100 * 1024 * 1024:
                                raise ValueError("Total image size exceeds 100MB limit")

                    elif item_type == "thinking" and "signature" in item:
                        processed_content.append({
                            "type": "thinking",
                            "thinking": item.get("thinking", ""),
                            "signature": item.get("signature", ""),
                        })

                    elif item_type == "redacted_thinking" and "data" in item:
                        processed_content.append({
                            "type": "redacted_thinking",
                            "data": item.get("data", ""),
                        })
            else:
                processed_content = [{"type": "text", "text": str(content)}]

            processed.append({
                "role": role,
                "content": processed_content,
            })

        return processed

    def _process_image(self, image_data: dict) -> dict:
        """Process image data with size validation."""
        url = image_data.get("image_url", {}).get("url", "")

        if url.startswith("data:image"):
            mime_type, base64_data = url.split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            image_size = len(base64_data) * 3 / 4
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(f"Image exceeds 5MB: {image_size / (1024*1024):.2f}MB")

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 4.6 Cache Control
    # ─────────────────────────────────────────────────────────────────────────
    def _attach_cache_control(self, block: dict) -> dict:
        """Attach cache control to a content block."""
        if not isinstance(block, dict):
            return block

        if block.get("type") in {"thinking", "redacted_thinking"}:
            return block

        if not block.get("type"):
            block["type"] = "text"
            if "text" not in block:
                block["text"] = ""

        cache_control = dict(block.get("cache_control", {}))
        cache_control["type"] = "ephemeral"
        cache_control["ttl"] = self.CACHE_TTL
        block["cache_control"] = cache_control
        return block

    def _normalize_content_blocks(self, raw_content) -> List[dict]:
        """Normalize content to list of blocks."""
        blocks = []

        items = raw_content if isinstance(raw_content, list) else [raw_content]

        for item in items:
            if isinstance(item, dict) and item.get("type"):
                blocks.append(dict(item))
            elif isinstance(item, dict) and "content" in item:
                blocks.extend(self._normalize_content_blocks(item["content"]))
            elif item is not None:
                blocks.append({"type": "text", "text": str(item)})

        return blocks

    def _prepare_system_blocks(self, system_message) -> Optional[List[dict]]:
        """Prepare system message blocks with cache control."""
        if not system_message:
            return None

        content = (
            system_message.get("content")
            if isinstance(system_message, dict) and "content" in system_message
            else system_message
        )

        normalized = self._normalize_content_blocks(content)
        cached = [self._attach_cache_control(block) for block in normalized]

        return cached if cached else None

    def _apply_cache_control_to_last_message(self, messages: List[dict]) -> None:
        """Apply cache control to last user message."""
        if not messages:
            return

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return

        for block in reversed(last_message.get("content", [])):
            if isinstance(block, dict) and block.get("type") not in {"thinking", "redacted_thinking"}:
                self._attach_cache_control(block)
                break

    # ─────────────────────────────────────────────────────────────────────────
    # 4.7 Payload & Headers Building
    # ─────────────────────────────────────────────────────────────────────────
    def _build_payload(
        self,
        model: str,
        messages: List[dict],
        system_blocks: Optional[List[dict]],
        body: dict,
        valves: "Pipe.Valves",
        will_enable_thinking: bool,
        is_thinking_model: bool,
    ) -> dict:
        """Build the API request payload."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": body.get("max_tokens", 64000),
            "stream": body.get("stream", True),
        }

        if body.get("stop"):
            payload["stop_sequences"] = body["stop"]

        if system_blocks:
            payload["system"] = system_blocks

        # Add thinking configuration
        if will_enable_thinking:
            thinking_budget = self._get_thinking_budget(valves)
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Thinking models require temperature = 1.0
            payload["temperature"] = 1.0
        else:
            # Add sampling parameters
            if not will_enable_thinking:
                payload["top_k"] = body.get("top_k", 40)

            if model.startswith("claude-sonnet-4-5") or model.startswith("claude-opus-4-5"):
                if valves.CLAUDE_45_USE_TEMPERATURE:
                    payload["temperature"] = body.get("temperature", 0.8)
                else:
                    payload["top_p"] = body.get("top_p", 0.9)
            else:
                payload["temperature"] = body.get("temperature", 0.8)
                payload["top_p"] = body.get("top_p", 0.9)

        return payload

    def _build_headers(
        self,
        valves: "Pipe.Valves",
        model_id: str,
        will_enable_thinking: bool,
        has_tools: bool,
        model_supports_interleaved: bool,
        code_execution_enabled: bool = False,
    ) -> dict:
        """Build API request headers."""
        headers = {
            "x-api-key": valves.ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

        beta_headers = []

        # User-specified beta features
        if valves.BETA_FEATURES:
            beta_headers.extend([h.strip() for h in valves.BETA_FEATURES.split(",") if h.strip()])

        # Interleaved thinking for tool use
        if (
            valves.ENABLE_INTERLEAVED_THINKING
            and will_enable_thinking
            and has_tools
            and model_supports_interleaved
        ):
            beta_headers.append("interleaved-thinking-2025-05-14")

        # Code execution
        if code_execution_enabled:
            beta_headers.append("code-execution-2025-08-25")

        # Context management
        if valves.ENABLE_CONTEXT_MANAGEMENT:
            beta_headers.append("context-management-2025-06-27")

        # MCP connector
        if valves.MCP_SERVERS_JSON:
            beta_headers.append("mcp-client-2025-04-04")

        if beta_headers:
            headers["anthropic-beta"] = ",".join(set(beta_headers))

        return headers

    def _get_thinking_budget(self, valves: "Pipe.Valves") -> int:
        """Get thinking budget based on effort level or valve."""
        if valves.EFFORT_LEVEL != "not_set":
            return EFFORT_TO_BUDGET.get(valves.EFFORT_LEVEL, 16000)
        return max(1024, min(63999, valves.THINKING_BUDGET))

    # ─────────────────────────────────────────────────────────────────────────
    # 4.8 Web Search
    # ─────────────────────────────────────────────────────────────────────────
    def _build_web_search_tool(self, valves: "Pipe.Valves") -> dict:
        """Build web search tool configuration."""
        tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": valves.WEB_SEARCH_MAX_USES,
        }

        if valves.WEB_SEARCH_ALLOWED_DOMAINS:
            tool["allowed_domains"] = [
                d.strip() for d in valves.WEB_SEARCH_ALLOWED_DOMAINS.split(",") if d.strip()
            ]

        if valves.WEB_SEARCH_BLOCKED_DOMAINS:
            tool["blocked_domains"] = [
                d.strip() for d in valves.WEB_SEARCH_BLOCKED_DOMAINS.split(",") if d.strip()
            ]

        if valves.WEB_SEARCH_USER_LOCATION:
            try:
                tool["user_location"] = json.loads(valves.WEB_SEARCH_USER_LOCATION)
            except json.JSONDecodeError:
                self.logger.warning("Invalid WEB_SEARCH_USER_LOCATION JSON")

        return tool

    # ─────────────────────────────────────────────────────────────────────────
    # 4.9 Tool Transformation
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def transform_tools(tools: dict | list | None) -> List[dict]:
        """Convert OpenWebUI tools to Anthropic format.

        OpenWebUI format:
            {"tool_name": {"spec": {...}, "callable": fn}}

        Anthropic format:
            {"name": "...", "description": "...", "input_schema": {...}}
        """
        if not tools:
            return []

        iterable = tools.values() if isinstance(tools, dict) else tools
        anthropic_tools = []

        for item in iterable:
            if not isinstance(item, dict):
                continue

            # Handle __tools__ entry with spec
            if "spec" in item:
                spec = item["spec"]
                if isinstance(spec, dict):
                    anthropic_tools.append({
                        "name": spec.get("name", ""),
                        "description": spec.get("description", ""),
                        "input_schema": spec.get("parameters", {"type": "object"}),
                    })

            # Handle already-Anthropic format
            elif "input_schema" in item:
                anthropic_tools.append(item)

        return anthropic_tools

    # ─────────────────────────────────────────────────────────────────────────
    # 4.10 MCP Configuration
    # ─────────────────────────────────────────────────────────────────────────
    def _parse_mcp_config(self, valves: "Pipe.Valves") -> Optional[dict]:
        """Parse MCP server configuration."""
        if not valves.MCP_SERVERS_JSON:
            return None

        try:
            servers = json.loads(valves.MCP_SERVERS_JSON)
            if not isinstance(servers, list):
                servers = [servers]

            return {"servers": servers}
        except json.JSONDecodeError:
            self.logger.warning("Invalid MCP_SERVERS_JSON")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # 4.11 HTTP Session
    # ─────────────────────────────────────────────────────────────────────────
    async def _get_or_init_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling."""
        if self.session is not None and not self.session.closed:
            return self.session

        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=10,
            keepalive_timeout=75,
            ttl_dns_cache=300,
        )

        timeout = aiohttp.ClientTimeout(
            connect=30,
            sock_connect=30,
            sock_read=3600,  # 1 hour for long thinking
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=json.dumps,
        )

        return self.session

    # ─────────────────────────────────────────────────────────────────────────
    # 4.12 Streaming Loop with Tool Execution
    # ─────────────────────────────────────────────────────────────────────────
    async def _run_streaming_loop(
        self,
        payload: dict,
        headers: dict,
        valves: "Pipe.Valves",
        tools: dict[str, Any],
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
        metadata: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Stream response with tool call handling."""

        session = await self._get_or_init_session()

        for loop_idx in range(valves.MAX_TOOL_LOOPS):
            accumulated_text = ""
            tool_calls: List[dict] = []
            current_tool_call: Optional[dict] = None
            current_tool_input_json = ""

            in_thinking = False
            has_yielded_think_close = False
            server_tool_blocks: dict[int, dict] = {}  # Track server_tool_use blocks for input accumulation

            # Track thinking blocks for interleaved thinking with tool use
            accumulated_thinking_blocks: List[dict] = []
            current_thinking_content = ""
            current_thinking_signature = ""

            # Buffer streamed output to reduce tiny SSE chunks and avoid splitting HTML tags.
            stream_buffer = ""

            OUTPUT_CHUNK_TARGET = 96
            OUTPUT_CHUNK_MAX = 32768
            CODE_FENCE_RE = re.compile(r"(?m)^[ \t]*```")
            CODE_FENCE_TICKS_RE = re.compile(r"(?m)^[ \t]{0,3}(`{3,})")

            def _has_unclosed_html_tag(text: str) -> bool:
                last_lt = text.rfind("<")
                if last_lt == -1:
                    return False
                last_gt = text.rfind(">")
                return last_gt < last_lt

            def _ends_inside_code_fence(text: str) -> bool:
                return len(CODE_FENCE_RE.findall(text)) % 2 == 1

            def _tail_outside_code_fences(text: str) -> str:
                if _ends_inside_code_fence(text):
                    return ""
                last_fence = None
                for match in CODE_FENCE_RE.finditer(text):
                    last_fence = match
                if not last_fence:
                    return text
                fence_line_end = text.find("\n", last_fence.start())
                if fence_line_end == -1:
                    return ""
                return text[fence_line_end + 1 :]

            def _pop_flushable_chunk(*, force: bool = False) -> str | None:
                nonlocal stream_buffer
                if not stream_buffer:
                    return None

                if force:
                    chunk = stream_buffer
                    stream_buffer = ""
                    return chunk

                if len(stream_buffer) < OUTPUT_CHUNK_TARGET and "\n" not in stream_buffer:
                    return None

                # Prefer splitting at line boundaries when possible, but never while inside a fenced code block.
                if "\n" in stream_buffer:
                    for idx in range(len(stream_buffer) - 1, -1, -1):
                        if stream_buffer[idx] != "\n":
                            continue
                        candidate_end = idx + 1
                        candidate = stream_buffer[:candidate_end]
                        if _ends_inside_code_fence(candidate):
                            continue
                        if _has_unclosed_html_tag(_tail_outside_code_fences(candidate)):
                            continue
                        chunk = candidate
                        stream_buffer = stream_buffer[candidate_end:]
                        return chunk

                if stream_buffer.endswith("`"):
                    return None

                if _ends_inside_code_fence(stream_buffer):
                    return None

                if _has_unclosed_html_tag(_tail_outside_code_fences(stream_buffer)):
                    return None

                if (
                    len(stream_buffer) >= OUTPUT_CHUNK_MAX
                    or stream_buffer[-1].isspace()
                    or stream_buffer[-1] in ".!?,;:"
                ):
                    chunk = stream_buffer
                    stream_buffer = ""
                    return chunk

                return None

            def _append_to_stream(text: str) -> List[str]:
                nonlocal stream_buffer
                if not text:
                    return []
                stream_buffer += text
                chunks: list[str] = []
                while True:
                    chunk = _pop_flushable_chunk()
                    if not chunk:
                        break
                    chunks.append(chunk)
                return chunks

            def _format_code_block(content: str, language: str | None = None) -> str:
                """Wrap arbitrary content in a fenced code block that won't be closed by inner fences."""
                content = (content or "").rstrip("\n")
                info = (language or "").strip().splitlines()[0]
                max_ticks = 0
                for match in CODE_FENCE_TICKS_RE.finditer(content):
                    max_ticks = max(max_ticks, len(match.group(1)))
                fence = "`" * max(3, max_ticks + 1)
                return f"\n{fence}{info}\n{content}\n{fence}\n"

            try:
                async with session.post(
                    ANTHROPIC_API_URL,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        try:
                            error_json = json.loads(error_text)
                            error_msg = error_json.get("error", {}).get("message", error_text)
                        except:
                            error_msg = error_text
                        raise Exception(f"HTTP {response.status}: {error_msg}")

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type", "")

                        # ─── Content Block Start ───
                        if event_type == "content_block_start":
                            block = data.get("content_block", {})
                            block_type = block.get("type", "")

                            if block_type == "thinking":
                                # Reset for new thinking block (signature comes via signature_delta)
                                current_thinking_content = ""
                                current_thinking_signature = ""
                                if valves.DISPLAY_THINKING and not in_thinking:
                                    in_thinking = True
                                    has_yielded_think_close = False  # Reset for interleaved thinking
                                    # Flush any pending output before opening a reasoning tag.
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield "<think>"

                            elif block_type == "text":
                                # Save completed thinking block before transitioning
                                if current_thinking_content:
                                    accumulated_thinking_blocks.append({
                                        "type": "thinking",
                                        "thinking": current_thinking_content,
                                        "signature": current_thinking_signature,
                                    })
                                    current_thinking_content = ""

                                if in_thinking and not has_yielded_think_close:
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield "</think>"
                                    has_yielded_think_close = True
                                    in_thinking = False

                                # Yield initial text if any
                                if block.get("text"):
                                    accumulated_text += block["text"]
                                    for chunk in _append_to_stream(block["text"]):
                                        yield chunk

                            elif block_type == "tool_use":
                                # Save completed thinking block before transitioning
                                if current_thinking_content:
                                    accumulated_thinking_blocks.append({
                                        "type": "thinking",
                                        "thinking": current_thinking_content,
                                        "signature": current_thinking_signature,
                                    })
                                    current_thinking_content = ""

                                if in_thinking and not has_yielded_think_close:
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield "</think>"
                                    has_yielded_think_close = True
                                    in_thinking = False

                                current_tool_call = {
                                    "id": block.get("id", ""),
                                    "name": block.get("name", ""),
                                    "input": {},
                                }
                                current_tool_input_json = ""

                                await self._emit_status(
                                    event_emitter,
                                    f"Calling {block.get('name', 'tool')}...",
                                    done=False,
                                )

                            elif block_type == "server_tool_use":
                                # Save completed thinking block before transitioning
                                if current_thinking_content:
                                    accumulated_thinking_blocks.append({
                                        "type": "thinking",
                                        "thinking": current_thinking_content,
                                        "signature": current_thinking_signature,
                                    })
                                    current_thinking_content = ""

                                # Close thinking before tool execution output
                                if in_thinking and not has_yielded_think_close:
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield "</think>"
                                    has_yielded_think_close = True
                                    in_thinking = False

                                # Server-side tool use (web search, code execution)
                                block_index = data.get("index", 0)
                                tool_name = block.get("name", "")
                                tool_input = block.get("input", {})

                                # Track this block for input accumulation from deltas
                                server_tool_blocks[block_index] = {
                                    "name": tool_name,
                                    "input_json": ""
                                }

                                # Emit status for server tools (code display happens in content_block_stop)
                                if tool_name == "web_search":
                                    await self._emit_status(
                                        event_emitter,
                                        "Searching the web...",
                                        done=False,
                                    )
                                elif tool_name == "bash_code_execution":
                                    await self._emit_status(
                                        event_emitter,
                                        "Executing code...",
                                        done=False,
                                    )
                                elif tool_name == "text_editor_code_execution":
                                    await self._emit_status(
                                        event_emitter,
                                        "File operation...",
                                        done=False,
                                    )

                            elif block_type == "redacted_thinking":
                                if valves.DISPLAY_THINKING:
                                    if not in_thinking:
                                        while (chunk := _pop_flushable_chunk(force=True)):
                                            yield chunk
                                        yield "<think>"
                                        in_thinking = True
                                    for chunk in _append_to_stream("[Redacted thinking content]"):
                                        yield chunk

                            elif block_type == "web_search_tool_result":
                                await self._handle_web_search_results(block, event_emitter)

                            elif block_type == "bash_code_execution_tool_result":
                                # Close thinking before output
                                if in_thinking and not has_yielded_think_close:
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield "</think>"
                                    has_yielded_think_close = True
                                    in_thinking = False

                                result_content = block.get("content", {})
                                stdout = result_content.get("stdout", "")
                                stderr = result_content.get("stderr", "")
                                return_code = result_content.get("return_code", 0)

                                if stdout:
                                    output_text = _format_code_block(stdout, "output")
                                    accumulated_text += output_text
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield output_text
                                if stderr:
                                    stderr_text = _format_code_block(stderr, "stderr")
                                    accumulated_text += stderr_text
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield stderr_text

                                await self._emit_status(
                                    event_emitter,
                                    f"Code executed (exit: {return_code})",
                                    done=True,
                                )

                            elif block_type == "text_editor_code_execution_tool_result":
                                # Close thinking before output
                                if in_thinking and not has_yielded_think_close:
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield "</think>"
                                    has_yielded_think_close = True
                                    in_thinking = False

                                result_content = block.get("content", {})
                                file_content = result_content.get("content", "")
                                file_type = result_content.get("file_type", "text")

                                if file_content:
                                    content_text = _format_code_block(file_content, file_type)
                                    accumulated_text += content_text
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield content_text

                                await self._emit_status(
                                    event_emitter,
                                    "File operation complete",
                                    done=True,
                                )

                        # ─── Content Block Delta ───
                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            delta_type = delta.get("type", "")

                            if delta_type == "thinking_delta":
                                thinking_text = delta.get("thinking", "")
                                current_thinking_content += thinking_text
                                if valves.DISPLAY_THINKING and in_thinking:
                                    for chunk in _append_to_stream(thinking_text):
                                        yield chunk

                            elif delta_type == "text_delta":
                                text = delta.get("text", "")
                                accumulated_text += text
                                for chunk in _append_to_stream(text):
                                    yield chunk

                            elif delta_type == "input_json_delta":
                                partial = delta.get("partial_json", "")
                                current_tool_input_json += partial
                                # Also accumulate for server_tool_use blocks
                                block_index = data.get("index", 0)
                                if block_index in server_tool_blocks:
                                    server_tool_blocks[block_index]["input_json"] += partial

                            elif delta_type == "citations_delta":
                                # Document citations from uploaded files
                                citation = delta.get("citation", {})
                                if citation:
                                    await self._emit_document_citation(event_emitter, citation)

                            elif delta_type == "signature_delta":
                                # Capture thinking block signature (sent just before block_stop)
                                current_thinking_signature = delta.get("signature", "")

                        # ─── Content Block Stop ───
                        elif event_type == "content_block_stop":
                            block_index = data.get("index", 0)

                            # Handle server_tool_use block completion - display the code
                            if block_index in server_tool_blocks:
                                tool_info = server_tool_blocks.pop(block_index)
                                tool_name = tool_info["name"]
                                try:
                                    tool_input = json.loads(tool_info["input_json"]) if tool_info["input_json"] else {}
                                except json.JSONDecodeError:
                                    tool_input = {}

                                # Display the code being executed
                                if tool_name == "bash_code_execution":
                                    command = tool_input.get("command", "")
                                    if command:
                                        lang = "python" if command.strip().startswith("python") else "bash"
                                        code_block = _format_code_block(command, lang)
                                        accumulated_text += code_block
                                        while (chunk := _pop_flushable_chunk(force=True)):
                                            yield chunk
                                        yield code_block

                                elif tool_name == "text_editor_code_execution":
                                    cmd = tool_input.get("command", "")
                                    path = tool_input.get("path", "")
                                    content = tool_input.get("file_text", "") or tool_input.get("new_str", "")

                                    if cmd == "create" and content:
                                        ext = path.split(".")[-1] if "." in path else "text"
                                        code_block = f"\n**Creating `{path}`:**{_format_code_block(content, ext)}"
                                        accumulated_text += code_block
                                        while (chunk := _pop_flushable_chunk(force=True)):
                                            yield chunk
                                        yield code_block
                                    elif cmd == "str_replace":
                                        old_str = tool_input.get("old_str", "")
                                        new_str = tool_input.get("new_str", "")
                                        if old_str or new_str:
                                            edit_block = f"\n**Editing `{path}`:**{_format_code_block(f'- {old_str}\n+ {new_str}', 'diff')}"
                                            accumulated_text += edit_block
                                            while (chunk := _pop_flushable_chunk(force=True)):
                                                yield chunk
                                            yield edit_block
                                    elif cmd == "view":
                                        view_msg = f"\n**Viewing `{path}`**\n"
                                        accumulated_text += view_msg
                                        while (chunk := _pop_flushable_chunk(force=True)):
                                            yield chunk
                                        yield view_msg

                            if current_tool_call:
                                # Parse accumulated JSON input
                                try:
                                    current_tool_call["input"] = json.loads(current_tool_input_json) if current_tool_input_json else {}
                                except json.JSONDecodeError:
                                    current_tool_call["input"] = {}

                                tool_calls.append(current_tool_call)
                                current_tool_call = None
                                current_tool_input_json = ""

                            if in_thinking and not has_yielded_think_close:
                                while (chunk := _pop_flushable_chunk(force=True)):
                                    yield chunk
                                yield "</think>"
                                has_yielded_think_close = True
                                in_thinking = False

                        # ─── Message Stop ───
                        elif event_type == "message_stop":
                            if in_thinking and not has_yielded_think_close:
                                while (chunk := _pop_flushable_chunk(force=True)):
                                    yield chunk
                                yield "</think>"
                            while (chunk := _pop_flushable_chunk(force=True)):
                                yield chunk
                            break

                        # ─── Message Delta (usage, stop reason) ───
                        elif event_type == "message_delta":
                            stop_reason = data.get("delta", {}).get("stop_reason")
                            if stop_reason == "end_turn":
                                if in_thinking and not has_yielded_think_close:
                                    while (chunk := _pop_flushable_chunk(force=True)):
                                        yield chunk
                                    yield "</think>"
                                while (chunk := _pop_flushable_chunk(force=True)):
                                    yield chunk
                                break

            except aiohttp.ClientError as e:
                self.logger.error("HTTP error: %s", e)
                await self._emit_error(event_emitter, str(e))
                yield f"Error: {e}"
                return

            # ─── Handle Tool Calls ───
            if not tool_calls:
                # No tool calls, we're done
                return

            # Execute tool calls
            tool_results = await self._execute_tool_calls(
                tool_calls,
                tools,
                event_emitter,
                valves,
            )

            # Build assistant content with thinking and tool use blocks
            # Thinking blocks must precede tool_use blocks for interleaved thinking
            assistant_content = []
            for thinking_block in accumulated_thinking_blocks:
                assistant_content.append(thinking_block)
            if accumulated_text:
                assistant_content.append({"type": "text", "text": accumulated_text})
            for call in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["input"],
                })

            # Add assistant message and tool results to payload
            payload["messages"].append({
                "role": "assistant",
                "content": assistant_content,
            })
            payload["messages"].append({
                "role": "user",
                "content": tool_results,
            })

            self.logger.debug("Tool loop %d: %d calls executed", loop_idx + 1, len(tool_calls))

        # Max loops reached
        await self._emit_status(event_emitter, "Max tool iterations reached", done=True)

    async def _handle_web_search_results(
        self,
        block: dict,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Process web search results and emit citations."""
        results = block.get("content", [])

        for result in results:
            if result.get("type") == "web_search_result":
                url = result.get("url", "")
                title = result.get("title", "")
                domain = urlparse(url).netloc.lower().lstrip("www.")

                citation_payload = {
                    "source": {"name": domain, "url": url},
                    "document": [title],
                    "metadata": [{
                        "source": url,
                        "date_accessed": datetime.date.today().isoformat(),
                    }],
                }

                await event_emitter({"type": "source", "data": citation_payload})

        # Emit source count status
        source_count = sum(1 for r in results if r.get("type") == "web_search_result")
        if source_count > 0:
            await self._emit_status(
                event_emitter,
                f"Found {source_count} sources",
                done=True,
            )

    async def _emit_document_citation(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        citation: dict,
    ) -> None:
        """Emit a document citation event for OpenWebUI sidebar."""
        if not event_emitter:
            return

        cite_type = citation.get("type", "")
        cited_text = citation.get("cited_text", "")
        doc_title = citation.get("document_title", "Document")

        # Format location based on citation type
        if cite_type == "page_location":
            location = f"Page {citation.get('start_page_number', '?')}"
        elif cite_type == "char_location":
            start = citation.get("start_char_index", 0)
            end = citation.get("end_char_index", 0)
            location = f"Characters {start}-{end}"
        else:
            location = f"Block {citation.get('start_block_index', 0)}"

        # Truncate long cited text for display
        display_text = cited_text[:200] + "..." if len(cited_text) > 200 else cited_text

        # Match existing web search citation format
        citation_payload = {
            "source": {"name": doc_title},
            "document": [display_text],
            "metadata": [{
                "source": doc_title,
                "location": location,
                "type": cite_type,
            }],
        }

        await event_emitter({"type": "source", "data": citation_payload})

    # ─────────────────────────────────────────────────────────────────────────
    # 4.13 Tool Execution
    # ─────────────────────────────────────────────────────────────────────────
    async def _execute_tool_calls(
        self,
        calls: List[dict],
        tools: dict[str, Any],
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
        valves: "Pipe.Valves",
    ) -> List[dict]:
        """Execute tool calls and return results."""

        async def execute_one(call: dict) -> dict:
            tool_name = call["name"]
            tool_config = tools.get(tool_name)

            if not tool_config:
                return {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": f"Tool '{tool_name}' not found",
                    "is_error": True,
                }

            fn = tool_config.get("callable")
            if not fn:
                return {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": f"Tool '{tool_name}' has no callable",
                    "is_error": True,
                }

            args = call["input"]

            try:
                await self._emit_status(
                    event_emitter,
                    f"Running {tool_name}...",
                    done=False,
                )

                if inspect.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    result = await asyncio.to_thread(fn, **args)

                await self._emit_status(
                    event_emitter,
                    f"Finished {tool_name}",
                    done=True,
                )

                return {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": str(result),
                }

            except Exception as e:
                self.logger.error("Tool %s error: %s", tool_name, e)
                await self._emit_status(
                    event_emitter,
                    f"Error running {tool_name}",
                    done=True,
                )
                return {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": str(e),
                    "is_error": True,
                }

        if valves.PARALLEL_TOOL_CALLS and len(calls) > 1:
            tasks = [execute_one(call) for call in calls]
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for call in calls:
                result = await execute_one(call)
                results.append(result)

        return list(results)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.14 Non-Streaming Response
    # ─────────────────────────────────────────────────────────────────────────
    async def _run_non_streaming(
        self,
        payload: dict,
        headers: dict,
        valves: "Pipe.Valves",
    ) -> str:
        """Handle non-streaming response."""
        payload["stream"] = False
        session = await self._get_or_init_session()

        async with session.post(
            ANTHROPIC_API_URL,
            headers=headers,
            json=payload,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")

            res = await response.json()

        content_blocks = res.get("content", [])
        if not content_blocks:
            return ""

        result_parts = []
        thinking_parts = []

        for block in content_blocks:
            block_type = block.get("type", "")

            if block_type == "thinking" and valves.DISPLAY_THINKING:
                thinking_parts.append(block.get("thinking", ""))

            elif block_type == "redacted_thinking" and valves.DISPLAY_THINKING:
                thinking_parts.append("[Redacted thinking content]")

            elif block_type == "text":
                result_parts.append(block.get("text", ""))

        result = ""
        if thinking_parts:
            result += f"<think>{''.join(thinking_parts)}</think>"
        result += "".join(result_parts)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 4.15 Event Emitters
    # ─────────────────────────────────────────────────────────────────────────
    async def _emit_status(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        description: str,
        *,
        done: bool = False,
    ) -> None:
        """Emit status update."""
        if event_emitter:
            await event_emitter({
                "type": "status",
                "data": {"description": description, "done": done},
            })

    async def _emit_error(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        error_message: str,
    ) -> None:
        """Emit error event."""
        if event_emitter:
            await event_emitter({
                "type": "chat:completion",
                "data": {
                    "error": {"message": error_message},
                    "done": True,
                },
            })

    async def _emit_message_delta(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        content: str,
    ) -> None:
        """Emit message content atomically (avoids SSE chunking issues)."""
        if event_emitter:
            await event_emitter({
                "type": "message",
                "data": {"content": content},
            })

    # ─────────────────────────────────────────────────────────────────────────
    # 4.16 Files API Support
    # ─────────────────────────────────────────────────────────────────────────
    async def _get_file_content(
        self,
        file_id: str,
    ) -> tuple[bytes | None, str | None, str | None]:
        """
        Fetch file content from OpenWebUI storage.

        Returns:
            Tuple of (content_bytes, filename, mime_type) or (None, None, None) on error
        """
        log = SessionLogger.get_logger(__name__)

        try:
            # Get file metadata from database
            file_model = await asyncio.to_thread(Files.get_file_by_id, file_id)
            if not file_model:
                log.warning(f"File not found in database: {file_id}")
                return None, None, None

            filename = file_model.filename
            mime_type = file_model.meta.get("content_type", "application/octet-stream") if file_model.meta else "application/octet-stream"

            # Get file path from database
            file_path = file_model.path
            if not file_path:
                log.warning(f"File has no path: {file_id}")
                return None, None, None

            # Resolve actual path via storage provider (handles S3, GCS, local, etc.)
            resolved_path = Storage.get_file(file_path)

            # Read actual file content from disk
            try:
                with open(resolved_path, "rb") as f:
                    content = f.read()
            except FileNotFoundError:
                log.warning(f"File not found on disk: {resolved_path}")
                return None, None, None
            except Exception as e:
                log.error(f"Error reading file {resolved_path}: {e}")
                return None, None, None

            return content, filename, mime_type

        except Exception as e:
            log.error(f"Error fetching file content for {file_id}: {e}")
            return None, None, None

    async def _upload_to_anthropic_files(
        self,
        session: aiohttp.ClientSession,
        content: bytes,
        filename: str,
        mime_type: str,
        api_key: str,
    ) -> str | None:
        """
        Upload file to Anthropic Files API.

        Returns:
            file_id on success, None on failure
        """
        log = SessionLogger.get_logger(__name__)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "anthropic-beta": FILES_API_BETA_HEADER,
        }

        form = aiohttp.FormData()
        form.add_field(
            "file",
            content,
            filename=filename,
            content_type=mime_type,
        )

        try:
            async with session.post(
                ANTHROPIC_FILES_API_URL,
                headers=headers,
                data=form,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    file_id = data.get("id")
                    log.info(f"Uploaded file to Anthropic: {filename} -> {file_id}")
                    return file_id
                else:
                    error = await resp.text()
                    log.error(f"File upload failed ({resp.status}): {error}")
                    return None
        except Exception as e:
            log.error(f"Exception during file upload: {e}")
            return None

    def _get_file_content_block(
        self,
        file_id: str,
        mime_type: str,
        code_execution_enabled: bool,
        citations_enabled: bool = False,
        filename: str | None = None,
    ) -> dict:
        """
        Get the appropriate content block for a file based on type.

        Args:
            file_id: Anthropic file ID
            mime_type: File MIME type
            code_execution_enabled: Whether code execution is active
            citations_enabled: Whether to enable document citations
            filename: Original filename (used as document title for citations)

        Returns:
            Content block dict for Anthropic Messages API
        """
        # Dataset types that benefit from code execution sandbox access
        dataset_types = {
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/json",
            "text/tab-separated-values",
        }

        # Helper to add citations config to document blocks
        def _add_citations(block: dict) -> dict:
            if citations_enabled and block.get("type") == "document":
                block["citations"] = {"enabled": True}
                if filename:
                    block["title"] = filename
            return block

        # For code execution: use container_upload for datasets and text files
        if code_execution_enabled and mime_type in dataset_types:
            return {
                "type": "container_upload",
                "file_id": file_id,
            }

        # For code execution: text files should also go to sandbox
        if code_execution_enabled and mime_type in ("text/plain", "text/markdown"):
            return {
                "type": "container_upload",
                "file_id": file_id,
            }

        # For PDFs - always use document (Claude reads directly)
        if mime_type == "application/pdf":
            return _add_citations({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id,
                },
            })

        # For text files without code execution - use document
        if mime_type in ("text/plain", "text/markdown"):
            return _add_citations({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id,
                },
            })

        # For images
        if mime_type.startswith("image/"):
            return {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": file_id,
                },
            }

        # Default: use container_upload if code execution is enabled
        # otherwise use document block
        if code_execution_enabled:
            return {
                "type": "container_upload",
                "file_id": file_id,
            }
        else:
            return _add_citations({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id,
                },
            })

    async def _process_and_upload_files(
        self,
        session: aiohttp.ClientSession,
        chat_id: str,
        user_id: str,
        api_key: str,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> list[dict]:
        """
        Fetch files from chat history and upload them to Anthropic Files API.

        Returns:
            List of file info dicts with file_id, filename, and mime_type
        """
        log = SessionLogger.get_logger(__name__)
        uploaded_files: list[dict] = []

        try:
            # Fetch chat from database to get file references
            chat = await asyncio.to_thread(
                Chats.get_chat_by_id_and_user_id, chat_id, user_id
            )
            if not chat or not chat.chat:
                log.warning(f"Could not fetch chat: {chat_id}")
                return []

            messages_db: list[dict[str, Any]] = []
            chat_data = chat.chat
            history_messages = chat_data.get("history", {}).get("messages")
            if isinstance(history_messages, dict):
                messages_db = list(history_messages.values())
            elif isinstance(history_messages, list):
                messages_db = history_messages
            else:
                maybe_messages = chat_data.get("messages", [])
                if isinstance(maybe_messages, list):
                    messages_db = maybe_messages

            file_ids_seen: set[str] = set()
            for msg in messages_db:
                for file_ref in msg.get("files", []):
                    fid = file_ref.get("id") or file_ref.get("file", {}).get("id")
                    if not fid:
                        continue
                    file_ids_seen.add(fid)
            if not file_ids_seen:
                log.debug("No files found in chat history")
                return []

            log.info(f"Processing {len(file_ids_seen)} files from chat history")

            # Process each file
            for file_id in file_ids_seen:
                # Fetch content from OpenWebUI storage
                content, filename, mime_type = await self._get_file_content(file_id)
                if content is None:
                    await self._emit_status(
                        event_emitter,
                        f"Failed to read file: {filename or file_id}",
                        done=True,
                    )
                    continue

                await self._emit_status(
                    event_emitter,
                    f"Uploading {filename}...",
                    done=False,
                )

                # Upload to Anthropic
                anthropic_file_id = await self._upload_to_anthropic_files(
                    session, content, filename, mime_type, api_key
                )

                if anthropic_file_id:
                    uploaded_files.append({
                        "file_id": anthropic_file_id,
                        "filename": filename,
                        "mime_type": mime_type,
                    })
                    await self._emit_status(
                        event_emitter,
                        f"Uploaded {filename}",
                        done=True,
                    )
                else:
                    await self._emit_status(
                        event_emitter,
                        f"Failed to upload: {filename}",
                        done=True,
                    )

            return uploaded_files

        except Exception as e:
            log.error(f"Error processing files: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # 4.17 Utilities
    # ─────────────────────────────────────────────────────────────────────────
    def _merge_valves(self, global_valves: "Pipe.Valves", user_valves: "Pipe.UserValves") -> "Pipe.Valves":
        """Merge user valves into global, respecting INHERIT."""
        if not user_valves:
            return global_valves

        update = {
            k: v
            for k, v in user_valves.model_dump().items()
            if v is not None and str(v).upper() != "INHERIT"
        }

        return global_valves.model_copy(update=update)
