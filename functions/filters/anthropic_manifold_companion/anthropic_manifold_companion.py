"""
title: Anthropic Manifold Companion
id: anthropic_manifold_companion
description: Companion filter for Anthropic Manifold - intercepts OpenWebUI toggles for native Claude features.
author: Based on Gemini Manifold Companion by suurt8ll
author_url: https://github.com/jrkropp/open-webui-developer-toolkit
license: MIT
version: 1.1.0
required_open_webui_version: 0.6.10
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field


log = logging.getLogger(__name__)


class Filter:
    """
    Companion filter for the Anthropic Manifold pipe.

    This filter intercepts OpenWebUI's native web_search and code_interpreter
    toggles BEFORE the middleware processes them.

    Features:
    1. Web Search Toggle - Intercepts and routes to Claude's native web search
    2. Code Interpreter Toggle - Intercepts and routes to Claude's native code execution
    3. File Upload Bypass - Routes files to Anthropic Files API instead of OpenWebUI RAG

    This is necessary because OpenWebUI's middleware runs BEFORE pipes,
    so the pipe alone cannot intercept the toggles in time.

    Note: For Extended Thinking toggle, install the separate "Extended Thinking" filter.
    """

    class Valves(BaseModel):
        """Configuration options for the companion filter."""

        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
            default="INFO",
            description="Logging verbosity level",
        )
        MANIFOLD_PREFIX: str = Field(
            default="anthropic.",
            description="Model ID prefix to match for Anthropic manifold models",
        )
        BYPASS_BACKEND_RAG: bool = Field(
            default=True,
            description="Bypass OpenWebUI's RAG and send files directly to Anthropic Files API",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Intercepts OpenWebUI toggles before middleware processes them.

        Args:
            body: The request body containing model, messages, and features
            __metadata__: Request metadata where we store signals for the manifold

        Returns:
            Modified body with OpenWebUI native features disabled
        """
        if __metadata__ is None:
            __metadata__ = {}

        # Check if this is an Anthropic manifold model
        model = body.get("model", "")

        # Also check base_model_id for workspace/custom models
        base_model_id = (
            body.get("metadata", {})
            .get("model", {})
            .get("info", {})
            .get("base_model_id", "")
        )

        effective_model = base_model_id if base_model_id else model
        prefix = self.valves.MANIFOLD_PREFIX

        if prefix not in effective_model:
            # Not an Anthropic manifold model, pass through unchanged
            return body

        log.debug(f"Anthropic Manifold Companion: Processing model {effective_model}")

        # Get features from body (this is where OpenWebUI stores toggle states)
        features = body.get("features", {})
        if not isinstance(features, dict):
            features = {}

        # Ensure metadata features dict exists
        metadata_features = __metadata__.setdefault("features", {})
        if not isinstance(metadata_features, dict):
            metadata_features = {}
            __metadata__["features"] = metadata_features

        # Create anthropic-specific features namespace
        anthropic_features = metadata_features.setdefault("anthropic", {})
        if not isinstance(anthropic_features, dict):
            anthropic_features = {}
            metadata_features["anthropic"] = anthropic_features

        # Intercept web_search toggle
        if features.get("web_search", False):
            log.info(
                "Anthropic Manifold Companion: Intercepting web_search toggle - "
                "disabling OpenWebUI native, enabling Claude native"
            )
            features["web_search"] = False  # Disable OpenWebUI native web search
            anthropic_features["web_search"] = True  # Signal manifold to use Claude's

        # Intercept code_interpreter toggle
        if features.get("code_interpreter", False):
            log.info(
                "Anthropic Manifold Companion: Intercepting code_interpreter toggle - "
                "disabling OpenWebUI native, enabling Claude native"
            )
            features["code_interpreter"] = False  # Disable OpenWebUI native interpreter
            anthropic_features["code_execution"] = True  # Signal manifold to use Claude's

        # Bypass OpenWebUI RAG - clear files and signal pipe to handle them
        if self.valves.BYPASS_BACKEND_RAG:
            chat_id = __metadata__.get("chat_id", "")
            if chat_id == "local":
                log.warning(
                    "Anthropic Manifold Companion: RAG bypass not supported for temporary chats"
                )
                anthropic_features["upload_documents"] = False
            else:
                if files := body.get("files"):
                    log.info(
                        f"Anthropic Manifold Companion: Bypassing OpenWebUI RAG for {len(files)} files"
                    )
                    body["files"] = []  # Clear to prevent OpenWebUI processing
                anthropic_features["upload_documents"] = True
        else:
            anthropic_features["upload_documents"] = False

        return body
