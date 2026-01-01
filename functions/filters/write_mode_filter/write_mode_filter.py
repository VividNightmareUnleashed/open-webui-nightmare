"""
title: Write Mode
id: write_mode_filter
version: 0.1.0
description: Switches to writing model and injects system prompt when enabled.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict


class Filter:
    class Valves(BaseModel):
        MODEL: str = Field(default="claude", description="Model to use in write mode")
        SYSTEM_PROMPT: str = Field(
            default="",
            description="System prompt to inject (optional)",
        )
        USER_PREFIX: str = Field(
            default="",
            description="Prefix to add to user message",
        )
        priority: int = Field(default=0, description="Filter execution order")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTcgM2EyLjg1IDIuODMgMCAxIDEgNCA0TDcuNSAyMC41IDIgMjJsMS41LTUuNVoiLz48cGF0aCBkPSJtMTUgNSA0IDQiLz48L3N2Zz4="

    async def inlet(
        self,
        body: Dict[str, Any],
        __metadata__: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        # Switch model
        body["model"] = self.valves.MODEL

        messages = body.get("messages", [])

        # Inject system prompt (if configured)
        if self.valves.SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": self.valves.SYSTEM_PROMPT})

        # Prepend prefix to last user message (if configured)
        if self.valves.USER_PREFIX:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    msg["content"] = self.valves.USER_PREFIX + msg["content"]
                    break

        body["messages"] = messages
        return body
