"""
title: Pro
id: pro_filter
description: Switch to Pro model.
required_open_webui_version: 0.6.10
version: 0.1.0
"""

from __future__ import annotations
from typing import Any, Awaitable, Callable
from pydantic import BaseModel, Field
from open_webui.models.models import Models


class Filter:
    class Valves(BaseModel):
        MODEL: str = "openai_responses.gpt-5.2-pro"
        priority: int = Field(default=0, description="Priority level for the filter operations.")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAtOTYwIDk2MCA5NjAiIHdpZHRoPSIyNHB4IiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwYXRoIGQ9Ik0xNjAtMTIwcS0zMyAwLTU2LjUtMjMuNVQ4MC0yMDB2LTU2MHEwLTMzIDIzLjUtNTYuNVQxNjAtODQwaDU2MHEzMyAwIDU2LjUgMjMuNVQ4MDAtNzYwdjgwaDgwdjgwaC04MHY4MGg4MHY4MGgtODB2ODBoODB2ODBoLTgwdjgwcTAgMzMtMjMuNSA1Ni41VDcyMC0xMjBIMTYwWm0wLTgwaDU2MHYtNTYwSDE2MHY1NjBabTgwLTgwaDIwMHYtMTYwSDI0MHYxNjBabTI0MC0yODBoMTYwdi0xMjBINDgwdjEyMFptLTI0MCA4MGgyMDB2LTIwMEgyNDB2MjAwWm0yNDAgMjAwaDE2MHYtMjQwSDQ4MHYyNDBaTTE2MC03NjB2NTYwLTU2MFoiLz48L3N2Zz4="

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __metadata__: dict | None = None,
    ) -> dict:
        """
        Inlet: Switch the model to Pro.
        """
        body["model"] = self.valves.MODEL

        model_info = Models.get_model_by_id(self.valves.MODEL)
        if __metadata__ and model_info:
            __metadata__["model"] = model_info.model_dump()

        return body

    async def outlet(
        self,
        body: dict,
        __metadata__: dict | None = None,
    ) -> dict:
        """
        Outlet: Set model fields for UI display.
        """
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                last_msg["model"] = self.valves.MODEL
                last_msg.setdefault("modelName", "Pro")

        return body
