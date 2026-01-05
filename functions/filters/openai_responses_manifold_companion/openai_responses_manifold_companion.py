"""
title: OpenAI Responses Manifold Companion
id: openai_responses_manifold_companion
description: Companion filter for OpenAI Responses Manifold - intercepts OpenWebUI native toggles and optionally bypasses backend RAG for OpenAI file tools.
author: Justin Kropp
author_url: https://github.com/jrkropp
license: MIT
version: 0.1.0
required_open_webui_version: 0.6.10
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field


log = logging.getLogger(__name__)


class Filter:
    """
    Companion filter for the OpenAI Responses Manifold pipe.

    Open WebUI processes native feature toggles (e.g., web_search, code_interpreter)
    after inlet filters run but before pipes execute. This filter intercepts those
    toggles so the OpenAI Responses Manifold can enable OpenAI-native tools instead
    of Open WebUI's built-in handlers.

    Optionally, this filter can bypass Open WebUI's backend RAG pipeline for file
    attachments when OpenAI file tools are enabled, preserving the original file list
    in metadata for the manifold pipe to upload.
    """

    class Valves(BaseModel):
        """Configuration options for the companion filter."""

        priority: int = Field(
            default=0,
            description="Priority level for filter execution order (lower runs earlier).",
        )
        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default="INFO",
            description="Logging verbosity level",
        )
        MANIFOLD_PREFIX: str = Field(
            default="openai_responses.",
            description="Model ID prefix to match for OpenAI Responses manifold models",
        )
        BYPASS_BACKEND_RAG: bool = Field(
            default=True,
            description=(
                "Bypass OpenWebUI's RAG and send files directly to OpenAI when OpenAI file tools "
                "(Code Interpreter / File Search) are enabled for the request."
            ),
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
            body: Request body containing model, messages, files, and feature toggles.
            __metadata__: Request metadata where signals are stored for the manifold.

        Returns:
            Modified request body with OpenWebUI native features disabled.
        """
        log.setLevel(getattr(logging, self.valves.LOG_LEVEL.upper(), logging.INFO))

        if __metadata__ is None:
            __metadata__ = {}

        model = str(body.get("model") or "")
        base_model_id = (
            body.get("metadata", {})
            .get("model", {})
            .get("info", {})
            .get("base_model_id", "")
        )
        metadata_model_id = (__metadata__.get("model") or {}).get("id", "")

        effective_model = str(base_model_id or metadata_model_id or model)
        if self.valves.MANIFOLD_PREFIX not in effective_model:
            return body

        features = body.get("features", {})
        if not isinstance(features, dict):
            features = {}
            body["features"] = features

        metadata_features = __metadata__.setdefault("features", {})
        if not isinstance(metadata_features, dict):
            metadata_features = {}
            __metadata__["features"] = metadata_features

        openai_features = metadata_features.setdefault("openai_responses", {})
        if not isinstance(openai_features, dict):
            openai_features = {}
            metadata_features["openai_responses"] = openai_features

        if features.get("web_search", False):
            log.info(
                "OpenAI Responses Manifold Companion: Intercepting web_search toggle - "
                "disabling OpenWebUI native, enabling OpenAI web search tool"
            )
            features["web_search"] = False
            openai_features["web_search"] = True

        if features.get("code_interpreter", False):
            log.info(
                "OpenAI Responses Manifold Companion: Intercepting code_interpreter toggle - "
                "disabling OpenWebUI native, enabling OpenAI code interpreter tool"
            )
            features["code_interpreter"] = False
            openai_features["code_interpreter"] = True

        file_tools_enabled = bool(
            openai_features.get("code_interpreter", False)
            or openai_features.get("file_search", False)
        )

        if self.valves.BYPASS_BACKEND_RAG and file_tools_enabled:
            files = body.get("files")
            if isinstance(files, list) and files:
                log.info(
                    "OpenAI Responses Manifold Companion: Bypassing OpenWebUI RAG for %s files",
                    len(files),
                )
                openai_features["files"] = list(files)
                body["files"] = []
                openai_features["bypass_backend_rag"] = True
            else:
                openai_features["bypass_backend_rag"] = False

        return body

