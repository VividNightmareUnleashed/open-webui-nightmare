"""
title: Extended Thinking
id: extended_thinking_filter
description: Set reasoning effort to high without changing models.
git_url: https://github.com/jrkropp/open-webui-developer-toolkit.git
required_open_webui_version: 0.6.10
version: 0.1.0
"""

from __future__ import annotations
from typing import Any, Awaitable, Callable, Literal
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        REASONING_EFFORT: Literal["minimal", "low", "medium", "high", "not set"] = "high"
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0iY3VycmVudENvbG9yIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMC4zODU3IDIuNTA5NzdDMTQuMzQ4NiAyLjcxMDU0IDE3LjUgNS45ODcyNCAxNy41IDEwQzE3LjUgMTQuMTQyMSAxNC4xNDIxIDE3LjUgMTAgMTcuNUM1Ljg1Nzg2IDE3LjUgMi41IDE0LjE0MjEgMi41IDEwQzIuNSA5LjcyMzg2IDIuNzIzODYgOS41IDMgOS41QzMuMjc2MTQgOS41IDMuNSA5LjcyMzg2IDMuNSAxMEMzLjUgMTMuNTg5OSA2LjQxMDE1IDE2LjUgMTAgMTYuNUMxMy41ODk5IDE2LjUgMTYuNSAxMy41ODk5IDE2LjUgMTBDMTYuNSA2LjUyMjUgMTMuNzY5MSAzLjY4MzEyIDEwLjMzNSAzLjUwODc5TDEwIDMuNUw5Ljg5OTQxIDMuNDkwMjNDOS42NzE0NSAzLjQ0MzcxIDkuNSAzLjI0MTcxIDkuNSAzQzkuNSAyLjcyMzg2IDkuNzIzODYgMi41IDEwIDIuNUwxMC4zODU3IDIuNTA5NzdaTTEwIDUuNUMxMC4yNzYxIDUuNSAxMC41IDUuNzIzODYgMTAuNSA2VjkuNjkwNDNMMTMuMjIzNiAxMS4wNTI3QzEzLjQ3MDYgMTEuMTc2MiAxMy41NzA4IDExLjQ3NjYgMTMuNDQ3MyAxMS43MjM2QzEzLjMzOTIgMTEuOTM5NyAxMy4wOTU3IDEyLjA0MzUgMTIuODcxMSAxMS45ODM0TDEyLjc3NjQgMTEuOTQ3M0w5Ljc3NjM3IDEwLjQ0NzNDOS42MDY5OCAxMC4zNjI2IDkuNSAxMC4xODk0IDkuNSAxMFY2QzkuNSA1LjcyMzg2IDkuNzIzODYgNS41IDEwIDUuNVpNMy42NjIxMSA2Ljk0MTQxQzQuMDI3MyA2Ljk0MTU5IDQuMzIzMDMgNy4yMzczNSA0LjMyMzI0IDcuNjAyNTRDNC4zMjMyNCA3Ljk2NzkxIDQuMDI3NDMgOC4yNjQ0NiAzLjY2MjExIDguMjY0NjVDMy4yOTY2MyA4LjI2NDY1IDMgNy45NjgwMiAzIDcuNjAyNTRDMy4wMDAyMSA3LjIzNzIzIDMuMjk2NzYgNi45NDE0MSAzLjY2MjExIDYuOTQxNDFaTTQuOTU2MDUgNC4yOTM5NUM1LjMyMTQ2IDQuMjk0MDQgNS42MTcxOSA0LjU5MDYzIDUuNjE3MTkgNC45NTYwNUM1LjYxNzEgNS4zMjE0IDUuMzIxNCA1LjYxNzA5IDQuOTU2MDUgNS42MTcxOUM0LjU5MDYzIDUuNjE3MTkgNC4yOTQwMyA1LjMyMTQ2IDQuMjkzOTUgNC45NTYwNUM0LjI5Mzk1IDQuNTkwNTcgNC41OTA1NyA0LjI5Mzk1IDQuOTU2MDUgNC4yOTM5NVpNNy42MDI1NCAzQzcuOTY4MDIgMyA4LjI2NDY1IDMuMjk2NjMgOC4yNjQ2NSAzLjY2MjExQzguMjY0NDYgNC4wMjc0MyA3Ljk2NzkxIDQuMzIzMjQgNy42MDI1NCA0LjMyMzI0QzcuMjM3MzYgNC4zMjMwMiA2Ljk0MTU5IDQuMDI3MyA2Ljk0MTQxIDMuNjYyMTFDNi45NDE0MSAzLjI5Njc2IDcuMjM3MjQgMy4wMDAyMiA3LjYwMjU0IDNaIj48L3BhdGg+PC9zdmc+Cg=="

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __metadata__: dict | None = None,
    ) -> dict:
        """
        Inlet: Add reasoning effort to the request without changing the model.
        """
        effort = self.valves.REASONING_EFFORT
        if effort != "not set":
            body["reasoning_effort"] = effort

        # Pass the updated request body downstream
        return body
