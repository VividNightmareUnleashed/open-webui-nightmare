"""
title: Regenerate with Pro
id: pro_regenerate_action
description: Regenerate an assistant message with gpt-5.2-pro (OpenAI Responses).
required_open_webui_version: 0.6.10
version: 0.1.0
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import Request
from pydantic import BaseModel, Field

from open_webui.functions import get_function_module_by_id
from open_webui.models.models import Models

_IN_FLIGHT_MESSAGE_IDS: set[str] = set()
_ALLOWED_ROLES = {"user", "assistant", "system", "developer"}
_MAX_COMPUTE_TOOLTIP = (
    "Regenerate the answer using maximum compute. This may take several minutes and be "
    "more costly, however it will produce a more in-depth, thoroughly reasoned and reliable "
    "answer."
)


class Action:
    class Valves(BaseModel):
        MODEL: str = Field(
            default="openai_responses.gpt-5.2-pro",
            description=(
                "Open WebUI model ID to use for regeneration. "
                "This should be a function model (e.g. openai_responses.gpt-5.2-pro)."
            ),
        )
        MODEL_NAME: str = Field(
            default="Pro",
            description="Display name to set on the regenerated message (modelName).",
        )
        PRO_FILTER_ID: str = Field(
            default="pro_filter",
            description=(
                "Optional filter function id to source the Pro model from (reads valves.MODEL). "
                "If the filter isn't installed/active, falls back to MODEL."
            ),
        )
        PREFER_FILTER_MODEL: bool = Field(
            default=True,
            description="Prefer PRO_FILTER_ID's valves.MODEL when available.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.icon = (
            "data:image/svg+xml;base64,"
            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIg"
            "dmlld0JveD0iMCAtOTYwIDk2MCA5NjAiIHdpZHRoPSIyNHB4IiBmaWxsPSJjdXJyZW50Q29s"
            "b3IiPjxwYXRoIGQ9Ik0xNjAtMTIwcS0zMyAwLTU2LjUtMjMuNVQ4MC0yMDB2LTU2MHEwLTMz"
            "IDIzLjUtNTYuNVQxNjAtODQwaDU2MHEzMyAwIDU2LjUgMjMuNVQ4MDAtNzYwdjgwaDgwdjgw"
            "aC04MHY4MGg4MHY4MGgtODB2ODBoODB2ODBoLTgwdjgwcTAgMzMtMjMuNSA1Ni41VDcyMC0x"
            "MjBIMTYwWm0wLTgwaDU2MHYtNTYwSDE2MHY1NjBabTgwLTgwaDIwMHYtMTYwSDI0MHYxNjBa"
            "bTI0MC0yODBoMTYwdi0xMjBINDgwdjEyMFptLTI0MCA4MGgyMDB2LTIwMEgyNDB2MjAwWm0y"
            "NDAgMjAwaDE2MHYtMjQwSDQ4MHYyNDBaTTE2MC03NjB2NTYwLTU2MFoiLz48L3N2Zz4="
        )
        self.actions = [
            {
                "id": "max_compute_regenerate",
                "name": _MAX_COMPUTE_TOOLTIP,
            }
        ]

    async def action(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> dict[str, Any]:
        message_id = body.get("id") or body.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "content": "Pro regeneration failed: missing message id.",
                    },
                }
            )
            return {"messages": []}

        if message_id in _IN_FLIGHT_MESSAGE_IDS:
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {
                        "type": "info",
                        "content": "Already regenerating this message.",
                    },
                }
            )
            return {"messages": [{"id": message_id}]}

        _IN_FLIGHT_MESSAGE_IDS.add(message_id)
        try:
            pro_model_id = self.valves.MODEL
            if self.valves.PREFER_FILTER_MODEL:
                try:
                    pro_filter = get_function_module_by_id(
                        __request__, self.valves.PRO_FILTER_ID
                    )
                    pro_model_id = getattr(
                        getattr(pro_filter, "valves", None),
                        "MODEL",
                        pro_model_id,
                    )
                except Exception:
                    pro_model_id = self.valves.MODEL

            pro_model_id = (pro_model_id or "").strip()
            if "." not in pro_model_id:
                pro_model_id = f"openai_responses.{pro_model_id}"

            pipe_id = pro_model_id.split(".", 1)[0]
            pipe = get_function_module_by_id(__request__, pipe_id)

            messages = body.get("messages") or []
            if not isinstance(messages, list):
                messages = []

            prompt_messages: list[dict[str, Any]] = [
                m
                for m in messages
                if isinstance(m, dict) and m.get("role") in _ALLOWED_ROLES
            ]
            if prompt_messages and prompt_messages[-1].get("role") == "assistant":
                prompt_messages = prompt_messages[:-1]

            await __event_emitter__({"type": "replace", "data": {"content": ""}})
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Regenerating with {self.valves.MODEL_NAME}…",
                        "done": False,
                    },
                }
            )

            metadata: dict[str, Any] = {
                "chat_id": body.get("chat_id"),
                "session_id": body.get("session_id"),
                "message_id": message_id,
                "user_id": __user__.get("id"),
                "features": {},
            }

            model_info = None
            try:
                model_info = Models.get_model_by_id(pro_model_id)
            except Exception:
                model_info = None

            if model_info is not None:
                try:
                    metadata["model"] = model_info.model_dump()
                except Exception:
                    metadata["model"] = {"id": pro_model_id}
            else:
                metadata["model"] = {"id": pro_model_id}

            request_body: dict[str, Any] = {
                "model": pro_model_id,
                "messages": prompt_messages,
                "stream": True,
            }

            result = await pipe.pipe(
                body=request_body,
                __user__=__user__,
                __request__=__request__,
                __event_emitter__=__event_emitter__,
                __metadata__=metadata,
                __tools__=None,
                __files__=None,
                __task__=None,
                __task_body__=None,
                __event_call__=None,
            )

            if isinstance(result, str):
                if result:
                    await __event_emitter__(
                        {"type": "message", "data": {"content": result}}
                    )
            elif result is not None:
                async for chunk in result:
                    if chunk:
                        await __event_emitter__(
                            {"type": "message", "data": {"content": str(chunk)}}
                        )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Regenerated with {self.valves.MODEL_NAME}.",
                        "done": True,
                    },
                }
            )
            await __event_emitter__(
                {"type": "chat:completion", "data": {"content": "", "done": True}}
            )

            return {
                "messages": [
                    {
                        "id": message_id,
                        "model": pro_model_id,
                        "modelName": self.valves.MODEL_NAME,
                        "done": True,
                    }
                ]
            }
        except Exception as exc:
            await __event_emitter__(
                {
                    "type": "chat:message:error",
                    "data": {"error": {"content": f"Pro regeneration failed: {exc}"}},
                }
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Pro regeneration failed.",
                        "done": True,
                        "error": True,
                    },
                }
            )
            return {"messages": [{"id": message_id, "done": True}]}
        finally:
            _IN_FLIGHT_MESSAGE_IDS.discard(message_id)
