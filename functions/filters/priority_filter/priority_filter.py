"""
title: Priority
id: priority_filter
description: Use priority processing for faster responses.
git_url: https://github.com/jrkropp/open-webui-developer-toolkit.git
required_open_webui_version: 0.6.10
version: 0.1.0
"""

from __future__ import annotations
from typing import Any, Awaitable, Callable, Literal
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        SERVICE_TIER: Literal["default", "flex", "priority"] = "priority"
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0iY3VycmVudENvbG9yIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMS41IDJMNCAxMmg1bC0xIDYgNy41LTEwaC01bDEuNS02eiIvPjwvc3ZnPgo="

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __metadata__: dict | None = None,
    ) -> dict:
        """
        Inlet: Add service_tier parameter to the request for priority processing.

        Sets the service_tier parameter which controls OpenAI's processing speed.
        - "priority": Premium pricing with latency SLAs (faster responses)
        - "default": Standard pricing and performance
        - "flex": 50% cheaper with increased latency (o3, o4-mini, gpt-5 only)
        """
        service_tier = self.valves.SERVICE_TIER

        # Set service_tier parameter (top-level parameter)
        body["service_tier"] = service_tier

        # Pass the updated request body downstream
        return body
