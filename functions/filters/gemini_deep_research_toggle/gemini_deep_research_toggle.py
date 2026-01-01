"""
title: Deep Research
id: gemini_deep_research_toggle
description: Enable Gemini Deep Research for comprehensive autonomous web research. Routes requests to the Deep Research pipe.
author: openwebuidev
author_url: https://github.com/openwebuidev
license: MIT
required_open_webui_version: 0.6.10
version: 0.2.0
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Filter:
    """
    Toggle filter for Gemini Deep Research.

    When enabled via the UI toggle, this filter:
    1. Routes the request to the Deep Research pipe
    2. Disables OpenWebUI's native web search and RAG (Deep Research has its own)
    3. Forces streaming mode for progress updates
    4. Supports follow-up questions via previous_interaction_id
    """

    # Tell Open WebUI to skip its file processing (bypass RAG)
    file_handler = True

    class Valves(BaseModel):
        """Configuration valves for the toggle filter."""

        DEFAULT_MODEL: str = Field(
            default="gemini_deep_research.deep-research-pro-preview-12-2025",
            description="Default Deep Research model/pipe to route requests to.",
        )
        priority: int = Field(
            default=0,
            description="Priority level for filter operations.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

        # Enable toggle UI
        self.toggle = True

        # Gemini Deep Research icon (base64 SVG)
        self.icon = "data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cGF0aCBkPSJNMTEuNTE3MSAxMS45OTI0QzExLjUxNzEgMTEuMTU0NCAxMC44Mzc1IDEwLjQ3NDkgOS45OTk1MSAxMC40NzQ4QzkuMTYxNCAxMC40NzQ4IDguNDgxOTQgMTEuMTU0MyA4LjQ4MTk0IDExLjk5MjRDOC40ODIwNCAxMi44MzA0IDkuMTYxNDYgMTMuNTEgOS45OTk1MSAxMy41MUMxMC44Mzc1IDEzLjUwOTkgMTEuNTE3IDEyLjgzMDQgMTEuNTE3MSAxMS45OTI0Wk0zLjgwNzEzIDguNzEwMThDMy4yODY5OSA4LjkyNjY2IDMuMDE5NzggOS41MDYzNCAzLjE5Mzg1IDEwLjA0MjJMMy41MjY4NiAxMS4wNjQ3TDMuNTY0OTQgMTEuMTY1M0MzLjc3ODYzIDExLjY1MzcgNC4zMjM1MSAxMS45MTg4IDQuODQ3MTcgMTEuNzcxN0w3LjMwMzIyIDExLjA4MDNDNy42ODM2MiA5Ljk1NTI2IDguNzQ2MDYgOS4xNDQ3NSA5Ljk5OTUxIDkuMTQ0NzVDMTAuNjk0NiA5LjE0NDc5IDExLjMzMTMgOS4zOTQwNiAxMS44MjU3IDkuODA3ODNMMTMuMDkzMyA5LjQ1MTM5TDEzLjAyNzggOS4yODA0OUwxMS43NjcxIDUuNDAwNjFMMy44MDcxMyA4LjcxMDE4Wk0xNC41OTYyIDMuMDU3ODNMMTQuNDY4MyAzLjA4NjE1TDEzLjYzODIgMy4zNTU2OUMxMy4wNzA1IDMuNTQwMjcgMTIuNzU5NCA0LjE1MDI1IDEyLjk0MzggNC43MTc5OUwxNC4yOTM1IDguODY5MzZMMTQuMzMyNSA4Ljk3Mjg3QzE0LjU1MzcgOS40NzQwOCAxNS4xMjM1IDkuNzM2NjQgMTUuNjU1OCA5LjU2MzY5TDE2LjQ4NTggOS4yOTMxOUwxNi42MDUgOS4yNDA0NUMxNi44Mjg1IDkuMTEzNjEgMTYuOTU2MiA4Ljg2NDA0IDE2LjkyNzIgOC42MDg2MUwxNi44OTk5IDguNDgxNjZMMTUuMjgwOCAzLjQ5OTI0QzE1LjE4NDQgMy4yMDMyMSAxNC44OTQgMy4wMjQyMiAxNC41OTYyIDMuMDU3ODNaTTEyLjg0NzIgMTEuOTkyNEMxMi44NDcxIDEyLjkyMTMgMTIuMzk5NyAxMy43NDMyIDExLjcxMTQgMTQuMjYyOUwxMy42MTg3IDE3LjMxMzdMMTMuNjc4MiAxNy40MzQ4QzEzLjc4NjMgMTcuNzI0NiAxMy42ODAyIDE4LjA2MDMgMTMuNDA3NyAxOC4yMzA3QzEzLjEzNTIgMTguNDAxIDEyLjc4NjkgMTguMzQ5MyAxMi41NzM3IDE4LjEyNTJMMTIuNDkwNyAxOC4wMTg4TDEwLjQ3NjEgMTQuNzk2MUMxMC4zMjA5IDE0LjgyMjMgMTAuMTYyMSAxNC44NCA5Ljk5OTUxIDE0Ljg0MDFDOS44MzU5NyAxNC44NDAxIDkuNjc2MDMgMTQuODIyNiA5LjUyMDAyIDE0Ljc5NjFMNy41MDYzNSAxOC4wMTg4TDcuNDIzMzQgMTguMTI1MkM3LjIxMDE1IDE4LjM0OTMgNi44NjE4NCAxOC40MDEgNi41ODkzNiAxOC4yMzA3QzYuMjc4MDMgMTguMDM2IDYuMTgzOCAxNy42MjUxIDYuMzc4NDIgMTcuMzEzN0w4LjI4NDY3IDE0LjI2MTlDNy43MjM3OSAxMy44Mzc2IDcuMzI1MzUgMTMuMjEyNCA3LjE5Nzc2IDEyLjQ5MTRMNS4yMDc1MiAxMy4wNTJDNC4wMzk2NyAxMy4zODA0IDIuODI0MDcgMTIuNzg5NiAyLjM0NzE3IDExLjcwMDRMMi4yNjIyMSAxMS40NzU4TDEuOTI5MiAxMC40NTMzQzEuNTQwNzYgOS4yNTc4NSAyLjEzNjY1IDcuOTY0MiAzLjI5NzM3IDcuNDgxNjZMMTEuNTg4NCA0LjAzMzQyQzExLjcxNzkgMy4xNTYyMiAxMi4zMjY2IDIuMzgzNzkgMTMuMjI3MSAyLjA5MTA0TDE0LjA1ODEgMS44MjA1M0wxNC4yNTI0IDEuNzY3NzlDMTUuMjMwNyAxLjU1NjEgMTYuMjI5NSAyLjExNjY4IDE2LjU0NTQgMy4wODkwOEwxOC4xNjQ2IDguMDcwNTNMMTguMjE3MyA4LjI2NTg0QzE4LjQxNDUgOS4xNzg3NCAxNy45NDAxIDEwLjEwOTcgMTcuMDg1NSAxMC40ODY1TDE2Ljg5NyAxMC41NTg4TDE2LjA2NTkgMTAuODI4M0MxNS4zNTQ5IDExLjA1OTIgMTQuNjE0NCAxMC45NDE5IDE0LjAyODggMTAuNTcwNUwxMi42NDk5IDEwLjk1NzJDMTIuNzc1NCAxMS4yNzgzIDEyLjg0NzIgMTEuNjI2OSAxMi44NDcyIDExLjk5MjRaIi8+Cjwvc3ZnPgo="

    async def inlet(
        self,
        body: dict[str, Any],
        __event_emitter__: Any | None = None,
        __metadata__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Inlet: Route request to Deep Research pipe and configure features.

        Args:
            body: Request payload
            __event_emitter__: Event emitter for status messages
            __metadata__: Request metadata for feature flags

        Returns:
            Modified request body
        """
        # Check if files are attached and warn user
        files = body.get("files", [])
        if files and __event_emitter__:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Deep Research does not support file uploads.",
                            "done": True,
                        },
                    }
                )
            except Exception:
                pass  # Don't fail if emit fails

        # Clear files to bypass RAG (file_handler=True also helps)
        body["files"] = []

        if __metadata__:
            # Ensure features dict exists
            features = __metadata__.setdefault("features", {})

            # Disable OpenWebUI's native web search
            # Deep Research has its own built-in google_search and url_context tools
            features["web_search"] = False

            # Signal to the Deep Research pipe that it was activated via toggle
            deep_research = features.setdefault("gemini_deep_research", {})
            deep_research["enabled"] = True
            deep_research["via_toggle"] = True

            # Pass previous interaction ID for follow-up support
            last_interaction_id = deep_research.get("last_interaction_id")
            if last_interaction_id:
                deep_research["previous_interaction_id"] = last_interaction_id

        # Route to Deep Research model/pipe
        body["model"] = self.valves.DEFAULT_MODEL

        # Force streaming for real-time progress updates
        body["stream"] = True

        return body

    async def outlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Outlet: Ensure response has correct model attribution and store interaction ID.

        Args:
            body: Response payload
            __metadata__: Request metadata

        Returns:
            Modified response body
        """
        messages = body.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
                # Set model name for display
                last_msg["model"] = self.valves.DEFAULT_MODEL
                last_msg.setdefault("modelName", "Gemini Deep Research")

        return body
