"""
title: Extended Thinking
id: extended_thinking_toggle
description: Toggle to enable Claude's extended thinking mode (shows reasoning process)
author: Based on Anthropic Manifold Companion
author_url: https://github.com/jrkropp/open-webui-developer-toolkit
license: MIT
version: 1.0.0
required_open_webui_version: 0.6.10
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field


log = logging.getLogger(__name__)


class Filter:
    """
    Toggle filter to enable Claude's extended thinking mode.

    When enabled, this filter signals the Anthropic Manifold pipe to use
    Claude's extended thinking feature, which displays the model's
    reasoning process in <think> tags.

    The clock icon toggle appears in the chat interface when this filter
    is enabled for the model.
    """

    class Valves(BaseModel):
        """Configuration options for the extended thinking toggle."""

        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
            default="INFO",
            description="Logging verbosity level",
        )
        MANIFOLD_PREFIX: str = Field(
            default="anthropic.",
            description="Model ID prefix to match for Anthropic manifold models",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Toggle - when enabled, signals the manifold to use extended thinking
        self.toggle = True
        # Clock icon for thinking
        self.icon = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0iY3VycmVudENvbG9yIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMC4zODU3IDIuNTA5NzdDMTQuMzQ4NiAyLjcxMDU0IDE3LjUgNS45ODcyNCAxNy41IDEwQzE3LjUgMTQuMTQyMSAxNC4xNDIxIDE3LjUgMTAgMTcuNUM1Ljg1Nzg2IDE3LjUgMi41IDE0LjE0MjEgMi41IDEwQzIuNSA5LjcyMzg2IDIuNzIzODYgOS41IDMgOS41QzMuMjc2MTQgOS41IDMuNSA5LjcyMzg2IDMuNSAxMEMzLjUgMTMuNTg5OSA2LjQxMDE1IDE2LjUgMTAgMTYuNUMxMy41ODk5IDE2LjUgMTYuNSAxMy41ODk5IDE2LjUgMTBDMTYuNSA2LjUyMjUgMTMuNzY5MSAzLjY4MzEyIDEwLjMzNSAzLjUwODc5TDEwIDMuNUw5Ljg5OTQxIDMuNDkwMjNDOS42NzE0NSAzLjQ0MzcxIDkuNSAzLjI0MTcxIDkuNSAzQzkuNSAyLjcyMzg2IDkuNzIzODYgMi41IDEwIDIuNUwxMC4zODU3IDIuNTA5NzdaTTEwIDUuNUMxMC4yNzYxIDUuNSAxMC41IDUuNzIzODYgMTAuNSA2VjkuNjkwNDNMMTMuMjIzNiAxMS4wNTI3QzEzLjQ3MDYgMTEuMTc2MiAxMy41NzA4IDExLjQ3NjYgMTMuNDQ3MyAxMS43MjM2QzEzLjMzOTIgMTEuOTM5NyAxMy4wOTU3IDEyLjA0MzUgMTIuODcxMSAxMS45ODM0TDEyLjc3NjQgMTEuOTQ3M0w5Ljc3NjM3IDEwLjQ0NzNDOS42MDY5OCAxMC4zNjI2IDkuNSAxMC4xODk0IDkuNSAxMFY2QzkuNSA1LjcyMzg2IDkuNzIzODYgNS41IDEwIDUuNVpNMy42NjIxMSA2Ljk0MTQxQzQuMDI3MyA2Ljk0MTU5IDQuMzIzMDMgNy4yMzczNSA0LjMyMzI0IDcuNjAyNTRDNC4zMjMyNCA3Ljk2NzkxIDQuMDI3NDMgOC4yNjQ0NiAzLjY2MjExIDguMjY0NjVDMy4yOTY2MyA4LjI2NDY1IDMgNy45NjgwMiAzIDcuNjAyNTRDMy4wMDAyMSA3LjIzNzIzIDMuMjk2NzYgNi45NDE0MSAzLjY2MjExIDYuOTQxNDFaTTQuOTU2MDUgNC4yOTM5NUM1LjMyMTQ2IDQuMjk0MDQgNS42MTcxOSA0LjU5MDYzIDUuNjE3MTkgNC45NTYwNUM1LjYxNzEgNS4zMjE0IDUuMzIxNCA1LjYxNzA5IDQuOTU2MDUgNS42MTcxOUM0LjU5MDYzIDUuNjE3MTkgNC4yOTQwMyA1LjMyMTQ2IDQuMjkzOTUgNC45NTYwNUM0LjI5Mzk1IDQuNTkwNTcgNC41OTA1NyA0LjI5Mzk1IDQuOTU2MDUgNC4yOTM5NVpNNy42MDI1NCAzQzcuOTY4MDIgMyA4LjI2NDY1IDMuMjk2NjMgOC4yNjQ2NSAzLjY2MjExQzguMjY0NDYgNC4wMjc0MyA3Ljk2NzkxIDQuMzIzMjQgNy42MDI1NCA0LjMyMzI0QzcuMjM3MzYgNC4zMjMwMiA2Ljk0MTU5IDQuMDI3MyA2Ljk0MTQxIDMuNjYyMTFDNi45NDE0MSAzLjI5Njc2IDcuMjM3MjQgMy4wMDAyMiA3LjYwMjU0IDNaIj48L3BhdGg+PC9zdmc+Cg=="

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Signal the manifold to enable extended thinking when toggle is on.

        Args:
            body: The request body containing model, messages, and features
            __metadata__: Request metadata where we store signals for the manifold

        Returns:
            Unmodified body (we only modify metadata)
        """
        log.setLevel(self.valves.LOG_LEVEL)

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

        # Signal thinking toggle state to manifold
        # The toggle_state attribute is set by OpenWebUI based on UI toggle state
        toggle_state = getattr(self, "toggle_state", True)
        if toggle_state:
            anthropic_features["thinking"] = True
            log.debug("Extended Thinking Toggle: Thinking enabled")
        else:
            anthropic_features["thinking"] = False
            log.debug("Extended Thinking Toggle: Thinking disabled")

        return body
