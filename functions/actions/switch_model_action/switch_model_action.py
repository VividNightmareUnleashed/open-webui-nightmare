"""
title: Pro Mode
id: switch_model_action
author: Open WebUI Developer Toolkit
type: action
description: Regenerate the response using a pro model.
git_url: https://github.com/jrkropp/open-webui-developer-toolkit.git
required_open_webui_version: 0.6.10
version: 0.2.2
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiBjbGFzcz0iaC01IHctNSBzaHJpbmstMCIgY29sb3I9InByaW1hcnkiPjxwYXRoIGQ9Ik0zLjU3NzQ2IDkuMTQwMDZMNC4wNDM4NyA5LjYxNDA2QzQuMTcwOTEgOS40ODkwNiA0LjI0MjQ2IDkuMzE4MjkgNC4yNDI0NiA5LjE0MDA2QzQuMjQyNDYgOC45NjE4MyA0LjE3MDkxIDguNzkxMDYgNC4wNDM4NyA4LjY2NjA1TDMuNTc3NDYgOS4xNDAwNlpNNS4xNTI2NSA1LjEwNjdMNS4yNDcxNiA1Ljc2NDk1QzUuNTc2IDUuNzE3NzQgNS44MTk1NSA1LjQzNTA4IDUuODE3NjQgNS4xMDI4OEw1LjE1MjY1IDUuMTA2N1pNNS4xNTI0OSA1LjA3Nzk3SDQuNDg3NDdMNC40ODc1IDUuMDgxNzlMNS4xNTI0OSA1LjA3Nzk3Wk0xNC44NDc1IDUuMDc3OTdMMTUuNTEyNSA1LjA4MTc5VjUuMDc3OTdIMTQuODQ3NVpNMTQuODQ3MyA1LjEwNjdMMTQuMTgyNCA1LjEwMjg4QzE0LjE4MDUgNS40MzUwOSAxNC40MjQgNS43MTc3NCAxNC43NTI4IDUuNzY0OTVMMTQuODQ3MyA1LjEwNjdaTTE2LjQyMjUgOS4xNDAwNkwxNS45NTYxIDguNjY2MDVDMTUuODI5MSA4Ljc5MTA2IDE1Ljc1NzUgOC45NjE4MyAxNS43NTc1IDkuMTQwMDZDMTUuNzU3NSA5LjMxODI5IDE1LjgyOTEgOS40ODkwNiAxNS45NTYxIDkuNjE0MDZMMTYuNDIyNSA5LjE0MDA2Wk0xNS40NDQ5IDE0LjYzMDFMMTUuMjA4NSAxNC4wMDg1QzE0Ljk5NDIgMTQuMDkgMTQuODM3IDE0LjI3NjIgMTQuNzkyNSAxNC41MDExTDE1LjQ0NDkgMTQuNjMwMVpNNC41NTUxIDE0LjYzMDFMNS4yMDc0OCAxNC41MDExQzUuMTYzMDIgMTQuMjc2MiA1LjAwNTgxIDE0LjA5IDQuNzkxNDkgMTQuMDA4NUw0LjU1NTEgMTQuNjMwMVpNMy4wODc0MyA4LjcxOTQ3QzIuODI1MTEgOC45NzY1MyAyLjgyMDg0IDkuMzk3NTYgMy4wNzc4OSA5LjY1OTg4QzMuMzM0OTQgOS45MjIyIDMuNzU1OTcgOS45MjY0NyA0LjAxODI5IDkuNjY5NDFMMy4wODc0MyA4LjcxOTQ3Wk01LjUyNjA2IDkuMDUzODlDNS44OTMzMyA5LjA1Mzg5IDYuMTkxMDYgOC43NTYxNiA2LjE5MTA2IDguMzg4ODlDNi4xOTEwNiA4LjAyMTYyIDUuODkzMzMgNy43MjM4OSA1LjUyNjA2IDcuNzIzODlWOS4wNTM4OVpNOS41MzQ1NyAxMC4zMzA2QzkuMjcyMjUgMTAuNTg3NiA5LjI2Nzk4IDExLjAwODcgOS41MjUwMyAxMS4yNzFDOS43ODIwOCAxMS41MzMzIDEwLjIwMzEgMTEuNTM3NiAxMC40NjU0IDExLjI4MDVMOS41MzQ1NyAxMC4zMzA2Wk0xMS45NzMyIDEwLjY2NUMxMi4zNDA1IDEwLjY2NSAxMi42MzgyIDEwLjM2NzMgMTIuNjM4MiAxMEMxMi42MzgyIDkuNjMyNzMgMTIuMzQwNSA5LjMzNSAxMS45NzMyIDkuMzM1VjEwLjY2NVpNMTcuOTEyMyAxMi4wMTQxQzE3LjkxMjMgMTEuNjQ2OCAxNy42MTQ2IDExLjM0OTEgMTcuMjQ3MyAxMS4zNDkxQzE2Ljg4IDExLjM0OTEgMTYuNTgyMyAxMS42NDY4IDE2LjU4MjMgMTIuMDE0MUgxNy45MTIzWk0xMy41MDQ5IDEzLjk2MTZDMTMuMTczMSAxMy44MDQgMTIuNzc2NSAxMy45NDUxIDEyLjYxODkgMTQuMjc2OUMxMi40NjEzIDE0LjYwODYgMTIuNjAyNCAxNS4wMDUzIDEyLjkzNDIgMTUuMTYyOUwxMy41MDQ5IDEzLjk2MTZaTTcuODY4NjggMTEuNTQ0NUM3LjUzNjk0IDExLjM4NjkgNy4xNDAyNiAxMS41MjgxIDYuOTgyNjcgMTEuODU5OEM2LjgyNTA3IDEyLjE5MTUgNi45NjYyNSAxMi41ODgyIDcuMjk3OTkgMTIuNzQ1OEw3Ljg2ODY4IDExLjU0NDVaTTEwLjI4NTMgMTIuNzQ1OEMxMC42MTcxIDEyLjU4ODIgMTAuNzU4MyAxMi4xOTE1IDEwLjYwMDcgMTEuODU5OEMxMC40NDMxIDExLjUyODEgMTAuMDQ2NCAxMS4zODY5IDkuNzE0NjUgMTEuNTQ0NUwxMC4yODUzIDEyLjc0NThaTTguMTAwODUgNi4xNzY4OEM3Ljc2OTExIDYuMzM0NDggNy42Mjc5MyA2LjczMTE2IDcuNzg1NTIgNy4wNjI5QzcuOTQzMTIgNy4zOTQ2NCA4LjMzOTggNy41MzU4MSA4LjY3MTU0IDcuMzc4MjJMOC4xMDA4NSA2LjE3Njg4Wk05LjU5NDUzIDcuMTcxMjNDOS45NjE3OSA3LjE3MTIzIDEwLjI1OTUgNi44NzM1IDEwLjI1OTUgNi41MDYyM0MxMC4yNTk1IDYuMTM4OTYgOS45NjE3OSA1Ljg0MTIzIDkuNTk0NTMgNS44NDEyM1Y3LjE3MTIzWk0xNS40OTk0IDUuMTY2NjdDMTUuNDk5NCA0Ljc5OTQgMTUuMjAxNyA0LjUwMTY3IDE0LjgzNDQgNC41MDE2N0MxNC40NjcxIDQuNTAxNjcgMTQuMTY5NCA0Ljc5OTQgMTQuMTY5NCA1LjE2NjY3SDE1LjQ5OTRaTTEzLjE4ODggNi43NDAxQzEyLjg1OTIgNi45MDIwNiAxMi43MjMyIDcuMzAwNTcgMTIuODg1MiA3LjYzMDJDMTMuMDQ3MiA3Ljk1OTgzIDEzLjQ0NTcgOC4wOTU3NiAxMy43NzUzIDcuOTMzOEwxMy4xODg4IDYuNzQwMVpNNS4xNTI0OSA1LjA3Nzk3TDQuNDg3NSA1LjA4MTc5TDQuNDg3NjYgNS4xMTA1Mkw1LjE1MjY1IDUuMTA2N0w1LjgxNzY0IDUuMTAyODhMNS44MTc0NyA1LjA3NDE0TDUuMTUyNDkgNS4wNzc5N1pNMTQuODQ3MyA1LjEwNjdMMTUuNTEyMyA1LjExMDUyTDE1LjUxMjUgNS4wODE3OUwxNC44NDc1IDUuMDc3OTdMMTQuMTgyNSA1LjA3NDE1TDE0LjE4MjQgNS4xMDI4OEwxNC44NDczIDUuMTA2N1pNMTAgNS4xNDI5NUg5LjMzNVYxNC44NDQxSDEwSDEwLjY2NVY1LjE0Mjk1SDEwWk00LjU1NTEgMTQuNjMwMUwzLjkwMjcyIDE0Ljc1OUM0LjI1NTQyIDE2LjU0MzYgNS42NDA2MSAxNy43MzI0IDcuMTI2NzkgMTcuODk1OEM3Ljg3NjQgMTcuOTc4MyA4LjY1MDcgMTcuNzk3MiA5LjI5OTM4IDE3LjI5NzJDOS45NDc1NSAxNi43OTc1IDEwLjQyMTEgMTYuMDE4NCAxMC42NDkzIDE0Ljk4NzlMMTAgMTQuODQ0MUw5LjM1MDczIDE0LjcwMDNDOS4xNzYxMyAxNS40ODg4IDguODQyMzQgMTUuOTcwMiA4LjQ4NzQxIDE2LjI0MzhDOC4xMzMgMTYuNTE3IDcuNzA4NDQgMTYuNjIxOCA3LjI3MjE5IDE2LjU3MzhDNi4zODY2NSAxNi40NzY0IDUuNDUyMTcgMTUuNzM5MiA1LjIwNzQ4IDE0LjUwMTFMNC41NTUxIDE0LjYzMDFaTTEwIDE0Ljg0NDFMOS4zNTA3MyAxNC45ODc5QzkuNTc4OTEgMTYuMDE4NCAxMC4wNTI0IDE2Ljc5NzUgMTAuNzAwNiAxNy4yOTcyQzExLjM0OTMgMTcuNzk3MiAxMi4xMjM2IDE3Ljk3ODMgMTIuODczMiAxNy44OTU4QzE0LjM1OTQgMTcuNzMyNCAxNS43NDQ2IDE2LjU0MzYgMTYuMDk3MyAxNC43NTlMMTUuNDQ0OSAxNC42MzAxTDE0Ljc5MjUgMTQuNTAxMUMxNC41NDc4IDE1LjczOTIgMTMuNjEzNCAxNi40NzY0IDEyLjcyNzggMTYuNTczOEMxMi4yOTE2IDE2LjYyMTggMTEuODY3IDE2LjUxNyAxMS41MTI2IDE2LjI0MzhDMTEuMTU3NyAxNS45NzAyIDEwLjgyMzkgMTUuNDg4OCAxMC42NDkzIDE0LjcwMDNMMTAgMTQuODQ0MVpNMy41Nzc0NiA5LjE0MDA2TDMuMTExMDQgOC42NjYwNUMyLjE4MzUyIDkuNTc4NzIgMS45NDc1OCAxMS4wMjEzIDIuMTU1NzYgMTIuMjU1NUMyLjM2MzU0IDEzLjQ4NzQgMy4wNjEwNiAxNC43NzMzIDQuMzE4NzEgMTUuMjUxNkw0LjU1NTEgMTQuNjMwMUw0Ljc5MTQ5IDE0LjAwODVDNC4xNDM3OCAxMy43NjIyIDMuNjMyMTMgMTMuMDExOSAzLjQ2NzI0IDEyLjAzNDNDMy4zMDI3NiAxMS4wNTkyIDMuNTMwNjggMTAuMTE5IDQuMDQzODcgOS42MTQwNkwzLjU3NzQ2IDkuMTQwMDZaTTUuMTUyNjUgNS4xMDY3TDUuMDU4MTQgNC40NDg0NUMzLjczMjczIDQuNjM4NzQgMi42OTk2OCA1LjM2NTc4IDIuMjgyMDcgNi4zOTkzNkMxLjg1NTIzIDcuNDU1NzggMi4xNDEwNiA4LjY1OTYxIDMuMTExMDQgOS42MTQwNkwzLjU3NzQ2IDkuMTQwMDZMNC4wNDM4NyA4LjY2NjA1QzMuMzkyODIgOC4wMjU0MiAzLjMyMTU1IDcuMzc2OTIgMy41MTUyMSA2Ljg5NzYxQzMuNzE4MTEgNi4zOTU0NSA0LjI4MzE3IDUuOTAzMzUgNS4yNDcxNiA1Ljc2NDk1TDUuMTUyNjUgNS4xMDY3Wk0xNi40MjI1IDkuMTQwMDZMMTYuODg5IDkuNjE0MDZDMTcuODU4OSA4LjY1OTYxIDE4LjE0NDggNy40NTU3OCAxNy43MTc5IDYuMzk5MzZDMTcuMzAwMyA1LjM2NTc4IDE2LjI2NzMgNC42Mzg3NCAxNC45NDE5IDQuNDQ4NDVMMTQuODQ3MyA1LjEwNjdMMTQuNzUyOCA1Ljc2NDk1QzE1LjcxNjggNS45MDMzNSAxNi4yODE5IDYuMzk1NDUgMTYuNDg0OCA2Ljg5NzYxQzE2LjY3ODUgNy4zNzY5MiAxNi42MDcyIDguMDI1NDIgMTUuOTU2MSA4LjY2NjA1TDE2LjQyMjUgOS4xNDAwNlpNMTUuNDQ0OSAxNC42MzAxTDE1LjY4MTMgMTUuMjUxNkMxNi45Mzg5IDE0Ljc3MzMgMTcuNjM2NSAxMy40ODc0IDE3Ljg0NDIgMTIuMjU1NkMxOC4wNTI0IDExLjAyMTMgMTcuODE2NSA5LjU3ODcyIDE2Ljg4OSA4LjY2NjA1TDE2LjQyMjUgOS4xNDAwNkwxNS45NTYxIDkuNjE0MDZDMTYuNDY5MyAxMC4xMTkgMTYuNjk3MiAxMS4wNTkyIDE2LjUzMjggMTIuMDM0M0MxNi4zNjc5IDEzLjAxMTkgMTUuODU2MiAxMy43NjIyIDE1LjIwODUgMTQuMDA4NUwxNS40NDQ5IDE0LjYzMDFaTTE0Ljg0NzUgNS4wNzc5N0gxNS41MTI1QzE1LjUxMjUgNC4xMDY4NCAxNS4xMjk0IDMuMzM0MzMgMTQuNTIxNyAyLjgxNTg3QzEzLjkyODEgMi4zMDk0MSAxMy4xNTk1IDIuMDc3MzUgMTIuNDE2NyAyLjA4NTE5QzExLjY3MzcgMi4wOTMwNCAxMC45MDgxIDIuMzQxMzkgMTAuMzE4NiAyLjg1ODk5QzkuNzE2MTUgMy4zODgwMyA5LjMzNSA0LjE2NzExIDkuMzM1IDUuMTQyOTVIMTBIMTAuNjY1QzEwLjY2NSA0LjUzODM2IDEwLjg4OTggNC4xMjc0NCAxMS4xOTYyIDMuODU4MzlDMTEuNTE1NiAzLjU3NzkxIDExLjk2MiAzLjQyMDA3IDEyLjQzMDggMy40MTUxMkMxMi44OTk5IDMuNDEwMTcgMTMuMzQzMiAzLjU1ODY5IDEzLjY1ODUgMy44Mjc2N0MxMy45NTk3IDQuMDg0NjYgMTQuMTgyNSA0LjQ4MjQ1IDE0LjE4MjUgNS4wNzc5N0gxNC44NDc1Wk0xMCA1LjE0Mjk1SDEwLjY2NUMxMC42NjUgNC4xNjcxMSAxMC4yODM5IDMuMzg4MDMgOS42ODEzNSAyLjg1ODk5QzkuMDkxODcgMi4zNDEzOCA4LjMyNjMzIDIuMDkzMDQgNy41ODMyNyAyLjA4NTE5QzYuODQwNTEgMi4wNzczNSA2LjA3MTkzIDIuMzA5NDEgNS40NzgzIDIuODE1ODdDNC44NzA2MSAzLjMzNDMzIDQuNDg3NDkgNC4xMDY4NCA0LjQ4NzQ5IDUuMDc3OTdINS4xNTI0OUg1LjgxNzQ5QzUuODE3NDkgNC40ODI0NSA2LjA0MDMgNC4wODQ2NiA2LjM0MTUyIDMuODI3NjdDNi42NTY4IDMuNTU4NjkgNy4xMDAxIDMuNDEwMTYgNy41NjkyMiAzLjQxNTEyQzguMDM4MDMgMy40MjAwNyA4LjQ4NDM3IDMuNTc3OTEgOC44MDM4IDMuODU4MzlDOS4xMTAyMSA0LjEyNzQ0IDkuMzM1IDQuNTM4MzYgOS4zMzUgNS4xNDI5NUgxMFpNMy41NTI4NiA5LjE5NDQ0TDQuMDE4MjkgOS42Njk0MUM0LjQwNzU1IDkuMjg3OTcgNC45Mzg4MSA5LjA1Mzg5IDUuNTI2MDYgOS4wNTM4OVY4LjM4ODg5VjcuNzIzODlDNC41NzY4NyA3LjcyMzg5IDMuNzE1MjEgOC4xMDQzIDMuMDg3NDMgOC43MTk0N0wzLjU1Mjg2IDkuMTk0NDRaTTEwIDEwLjgwNTZMMTAuNDY1NCAxMS4yODA1QzEwLjg1NDcgMTAuODk5MSAxMS4zODU5IDEwLjY2NSAxMS45NzMyIDEwLjY2NVYxMFY5LjMzNUMxMS4wMjQgOS4zMzUgMTAuMTYyNCA5LjcxNTQxIDkuNTM0NTcgMTAuMzMwNkwxMCAxMC44MDU2Wk0xNy4yNDczIDEyLjAxNDFIMTYuNTgyM0MxNi41ODIzIDEzLjIwNCAxNS42MTc3IDE0LjE2ODYgMTQuNDI3OSAxNC4xNjg2VjE0LjgzMzZWMTUuNDk4NkMxNi4zNTIzIDE1LjQ5ODYgMTcuOTEyMyAxMy45Mzg1IDE3LjkxMjMgMTIuMDE0MUgxNy4yNDczWk0xNC40Mjc5IDE0LjgzMzZWMTQuMTY4NkMxNC4wOTYyIDE0LjE2ODYgMTMuNzgzOCAxNC4wOTQxIDEzLjUwNDkgMTMuOTYxNkwxMy4yMTk1IDE0LjU2MjJMMTIuOTM0MiAxNS4xNjI5QzEzLjM4NzcgMTUuMzc4MyAxMy44OTQ3IDE1LjQ5ODYgMTQuNDI3OSAxNS40OTg2VjE0LjgzMzZaTTguNzkxNjcgMTIuNDE2NVYxMS43NTE1QzguNDYwMDIgMTEuNzUxNSA4LjE0NzYzIDExLjY3NyA3Ljg2ODY4IDExLjU0NDVMNy41ODMzMyAxMi4xNDUyTDcuMjk3OTkgMTIuNzQ1OEM3Ljc1MTQ5IDEyLjk2MTMgOC4yNTg0NyAxMy4wODE1IDguNzkxNjcgMTMuMDgxNVYxMi40MTY1Wk0xMCAxMi4xNDUyTDkuNzE0NjUgMTEuNTQ0NUM5LjQzNTcgMTEuNjc3IDkuMTIzMzEgMTEuNzUxNSA4Ljc5MTY3IDExLjc1MTVWMTIuNDE2NVYxMy4wODE1QzkuMzI0ODcgMTMuMDgxNSA5LjgzMTg0IDEyLjk2MTMgMTAuMjg1MyAxMi43NDU4TDEwIDEyLjE0NTJaTTguMzg2MTkgNi43Nzc1NUw4LjY3MTU0IDcuMzc4MjJDOC45NTA0OSA3LjI0NTcxIDkuMjYyODggNy4xNzEyMyA5LjU5NDUzIDcuMTcxMjNWNi41MDYyM1Y1Ljg0MTIzQzkuMDYxMzMgNS44NDEyMyA4LjU1NDM1IDUuOTYxNDUgOC4xMDA4NSA2LjE3Njg4TDguMzg2MTkgNi43Nzc1NVpNMTQuODM0NCA1LjE2NjY3SDE0LjE2OTRDMTQuMTY5NCA1Ljg1NjI2IDEzLjc3MSA2LjQ1NDA1IDEzLjE4ODggNi43NDAxTDEzLjQ4MiA3LjMzNjk1TDEzLjc3NTMgNy45MzM4QzE0Ljc5NTEgNy40MzI3MiAxNS40OTk0IDYuMzgyNTYgMTUuNDk5NCA1LjE2NjY3SDE0LjgzNDRaIiBmaWxsPSJjdXJyZW50Q29sb3IiPjwvcGF0aD48L3N2Zz4K
"""

from __future__ import annotations
import inspect
import logging
from typing import Any, Awaitable, Callable, Optional
from fastapi import Request
from pydantic import BaseModel, Field

# Import Open WebUI utilities for loading function modules
from open_webui.utils.plugin import get_function_module_from_cache
from open_webui.models.functions import Functions


class Action:
    class Valves(BaseModel):
        TARGET_MODEL: str = Field(
            default="openai_responses.gpt-5-thinking-high",
            description="The model to switch to (use full manifold model ID, e.g., 'openai_responses.gpt-5-thinking-high')"
        )
        MANIFOLD_ID: str = Field(
            default="openai_responses",
            description="The ID of the manifold pipe to invoke (default: openai_responses)"
        )
        DEBUG: bool = Field(
            default=False,
            description="Show debug information in notifications (useful when troubleshooting)"
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.logger = logging.getLogger(__name__)

    async def action(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __event_call__: Optional[Callable[[dict[str, Any]], Awaitable[Any]]] = None,
        __metadata__: Optional[dict] = None,
        __tools__: Optional[list | dict] = None,
        __task__: Optional[dict] = None,
        __task_body__: Optional[dict] = None,
    ) -> dict:
        """
        Action: Switch to pro model and regenerate the response.

        This action invokes the OpenAI Responses Manifold directly, ensuring
        all manifold logic (reasoning_effort, text.verbosity, service_tier)
        and active filters (Extended Thinking, Verbose, Priority) are respected.
        """
        manifold_id = self.valves.MANIFOLD_ID
        target_model = self.valves.TARGET_MODEL

        # 1. Load the manifold module from cache
        try:
            manifold_module, _, _ = get_function_module_from_cache(__request__, manifold_id)
        except Exception as e:
            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": f"Failed to load manifold '{manifold_id}': {e}"
                }
            })
            return {"content": f"Error: Could not load manifold '{manifold_id}'"}

        # 2. Load and set manifold valves from database
        if hasattr(manifold_module, "valves") and hasattr(manifold_module, "Valves"):
            try:
                valves = Functions.get_function_valves_by_id(manifold_id)
                manifold_module.valves = manifold_module.Valves(**(valves if valves else {}))
            except Exception as e:
                await __event_emitter__({
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "content": f"Failed to load manifold valves: {e}"
                    }
                })
                return {"content": f"Error: Could not load manifold valves"}

        # 3. Change model to target pro model
        body["model"] = target_model

        # 3a. Ensure stream is set (required by manifold)
        if "stream" not in body:
            body["stream"] = True

        # 3b. CRITICAL FIX: Update metadata to match new model
        # The manifold uses __metadata__["model"]["id"] to fetch conversation history
        # and apply model-specific logic. It MUST match body["model"].
        if __metadata__ is None:
            __metadata__ = {}
        if "model" not in __metadata__:
            __metadata__["model"] = {}
        __metadata__["model"]["id"] = target_model

        # 3b. Verify body has required fields
        if "messages" not in body or not body.get("messages"):
            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": "Invalid request: no messages found"
                }
            })
            return {"content": "Error: No messages in request body"}

        # 3c. Debug logging and notification
        self.logger.info("Pro Mode Action Debug:")
        self.logger.info(f"  Target Model: {target_model}")
        self.logger.info(f"  Body Model: {body.get('model')}")
        self.logger.info(f"  Metadata Model: {__metadata__.get('model', {}).get('id')}")
        self.logger.info(f"  Message Count: {len(body.get('messages', []))}")
        self.logger.info(f"  Stream: {body.get('stream', 'not set')}")
        self.logger.info(f"  Chat ID: {__metadata__.get('chat_id', 'not set')}")

        # Show debug info in notification if DEBUG valve is enabled
        if self.valves.DEBUG:
            import json
            debug_info = {
                "target_model": target_model,
                "body_model": body.get("model"),
                "metadata_model": __metadata__.get("model", {}).get("id"),
                "message_count": len(body.get("messages", [])),
                "stream": body.get("stream"),
                "chat_id": __metadata__.get("chat_id"),
                "body_keys": list(body.keys()),
            }
            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "info",
                    "content": f"Debug Info:\n{json.dumps(debug_info, indent=2)}"
                }
            })

        # 4. Show minimal status feedback
        await __event_emitter__({
            "type": "status",
            "data": {
                "description": "Pro Mode",
                "done": True
            }
        })

        # 5. Call manifold's pipe method with all parameters
        try:
            result = await manifold_module.pipe(
                body=body,
                __user__=__user__,
                __request__=__request__,
                __event_emitter__=__event_emitter__,
                __metadata__=__metadata__ or {},
                __tools__=__tools__,
                __task__=__task__,
                __task_body__=__task_body__,
                __event_call__=__event_call__,
            )
        except Exception as e:
            # Log detailed error information
            self.logger.error(f"Pro Mode Action Error: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")

            # Try to extract more details from aiohttp errors
            error_detail = str(e)
            if hasattr(e, 'status'):
                self.logger.error(f"HTTP Status: {e.status}")
            if hasattr(e, 'message'):
                self.logger.error(f"HTTP Message: {e.message}")
            if hasattr(e, 'headers'):
                self.logger.error(f"Response Headers: {e.headers}")

            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": f"Failed to generate response: {error_detail}"
                }
            })
            return {"content": f"Error: {error_detail}"}

        # 6. Handle streaming or non-streaming response
        if inspect.isasyncgen(result):
            # Streaming: collect all chunks
            full_response = ""
            try:
                async for chunk in result:
                    full_response += str(chunk)
            except Exception as e:
                await __event_emitter__({
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "content": f"Streaming error: {e}"
                    }
                })
                return {"content": full_response or f"Error during streaming: {e}"}

            return {"content": full_response}
        elif isinstance(result, str):
            # Non-streaming: return directly
            return {"content": result}
        else:
            # Unexpected result type
            return {"content": ""}
