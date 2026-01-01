"""
title: Plan Mode
id: plan_mode_filter
version: 0.7.1
description: Switches to planning model while preserving the chat's system prompt.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from open_webui.models.models import Models

MANIFOLD_PREFIX = "gemini_manifold_google_genai."


class Filter:
    class Valves(BaseModel):
        MODEL: str = Field(default="gemini_manifold_google_genai.gemini-3-pro-preview", description="Model to use in plan mode")
        USER_PREFIX: str = Field(
            default="[What do you think?] ",
            description="Prefix to add to user message",
        )
        priority: int = Field(default=0, description="Filter execution order")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWdvbiBwb2ludHM9IjEgNiAxIDIyIDggMTggMTYgMjIgMjMgMTggMjMgMiAxNiA2IDggMiAxIDYiLz48bGluZSB4MT0iOCIgeTE9IjIiIHgyPSI4IiB5Mj0iMTgiLz48bGluZSB4MT0iMTYiIHkxPSI2IiB4Mj0iMTYiIHkyPSIyMiIvPjwvc3ZnPg=="

    async def inlet(
        self,
        body: Dict[str, Any],
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        messages = body.get("messages", [])

        # Check if there's already a system message in messages
        has_system_message = any(msg.get("role") == "system" for msg in messages)

        # If no system message, get it from the original model's database config
        if not has_system_message:
            original_model_id = body.get("model")
            if original_model_id:
                model_info = Models.get_model_by_id(original_model_id)
                if model_info and model_info.params:
                    system_prompt = model_info.params.model_dump().get("system")
                    if system_prompt:
                        messages.insert(0, {"role": "system", "content": system_prompt})

        # Switch model
        body["model"] = self.valves.MODEL

        # Set canonical_model_id for gemini manifold (strip prefix)
        if __metadata__ and self.valves.MODEL.startswith(MANIFOLD_PREFIX):
            __metadata__["canonical_model_id"] = self.valves.MODEL[len(MANIFOLD_PREFIX):]

        # Disable native web search, enable Gemini's google_search_tool
        body.setdefault("features", {})["web_search"] = False
        if __metadata__:
            __metadata__.setdefault("features", {})["google_search_tool"] = True

        # Prepend prefix to last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = self.valves.USER_PREFIX + msg["content"]
                break

        body["messages"] = messages
        return body

