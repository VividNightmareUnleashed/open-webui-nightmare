"""
title: Verbose
id: verbose_filter
description: Set verbosity to high for more detailed responses.
git_url: https://github.com/jrkropp/open-webui-developer-toolkit.git
required_open_webui_version: 0.6.10
version: 0.1.0
"""

from __future__ import annotations
from typing import Any, Awaitable, Callable, Literal
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        VERBOSITY: Literal["low", "medium", "high"] = "high"
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCAyOCAyOCIgZmlsbD0iY3VycmVudENvbG9yIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik00IDdDNCA2LjQ0NzcxIDQuNDQ3NzIgNiA1IDZIMjRDMjQuNTUyMyA2IDI1IDYuNDQ3NzEgMjUgN0MyNSA3LjU1MjI5IDI0LjU1MjMgOCAyNCA4SDVDNC40NDc3MiA4IDQgNy41NTIyOSA0IDdaIi8+PHBhdGggZD0iTTQgMTMuOTk5OEM0IDEzLjQ0NzUgNC40NDc3MiAxMi45OTk3IDUgMTIuOTk5N0wxNiAxM0MxNi41NTIzIDEzIDE3IDEzLjQ0NzcgMTcgMTRDMTcgMTQuNTUyMyAxNi41NTIzIDE1IDE2IDE1TDUgMTQuOTk5OEM0LjQ0NzcyIDE0Ljk5OTggNCAxNC41NTIgNCAxMy45OTk4WiIvPjxwYXRoIGQ9Ik01IDE5Ljk5OThDNC40NDc3MiAxOS45OTk4IDQgMjAuNDQ3NSA0IDIwLjk5OThDNCAyMS41NTIgNC40NDc3MiAyMS45OTk3IDUgMjEuOTk5N0gyMkMyMi41NTIzIDIxLjk5OTcgMjMgMjEuNTUyIDIzIDIwLjk5OThDMjMgMjAuNDQ3NSAyMi41NTIzIDE5Ljk5OTggMjIgMTkuOTk5OEg1WiIvPjwvc3ZnPgo="

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __metadata__: dict | None = None,
    ) -> dict:
        """
        Inlet: Add verbosity parameter to the request without changing the model.

        Sets the text.verbosity parameter which controls response detail level.
        This parameter is only supported by GPT-5 models (gpt-5, gpt-5-mini, gpt-5-nano).
        """
        verbosity = self.valves.VERBOSITY

        # Create nested text.verbosity parameter (OpenAI pattern)
        text_params = body.get("text", {})
        if not isinstance(text_params, dict):
            text_params = {}
        text_params["verbosity"] = verbosity
        body["text"] = text_params

        # Pass the updated request body downstream
        return body
