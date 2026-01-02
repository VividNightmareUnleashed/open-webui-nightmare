"""
title: Turbo
id: turbo_filter
description: Enable priority service tier for faster responses.
required_open_webui_version: 0.6.10
version: 0.1.0
"""

from __future__ import annotations
from typing import Any, Awaitable, Callable
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Priority level for the filter operations.")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAtOTYwIDk2MCA5NjAiIHdpZHRoPSIyNHB4IiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwYXRoIGQ9Ik0xMDAtMjQwdi00ODBsMzYwIDI0MC0zNjAgMjQwWm00MDAgMHYtNDgwbDM2MCAyNDAtMzYwIDI0MFpNMTgwLTQ4MFptNDAwIDBabS00MDAgOTAgMTM2LTkwLTEzNi05MHYxODBabTQwMCAwIDEzNi05MC0xMzYtOTB2MTgwWiIvPjwvc3ZnPg=="

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __metadata__: dict | None = None,
    ) -> dict:
        """
        Inlet: Set service tier to priority for faster responses.
        """
        body["service_tier"] = "priority"
        return body
