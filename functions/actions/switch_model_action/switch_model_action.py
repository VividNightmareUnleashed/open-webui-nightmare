"""
title: Pro Mode
id: switch_model_action
author: Open WebUI Developer Toolkit
type: action
description: Regenerate the response using a pro model with full conversation context.
git_url: https://github.com/jrkropp/open-webui-developer-toolkit.git
required_open_webui_version: 0.6.10
version: 0.4.2
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiBjbGFzcz0iaC01IHctNSBzaHJpbmstMCIgY29sb3I9InByaW1hcnkiPjxwYXRoIGQ9Ik0zLjU3NzQ2IDkuMTQwMDZMNC4wNDM4NyA5LjYxNDA2QzQuMTcwOTEgOS40ODkwNiA0LjI0MjQ2IDkuMzE4MjkgNC4yNDI0NiA5LjE0MDA2QzQuMjQyNDYgOC45NjE4MyA0LjE3MDkxIDguNzkxMDYgNC4wNDM4NyA4LjY2NjA1TDMuNTc3NDYgOS4xNDAwNlpNNS4xNTI2NSA1LjEwNjdMNS4yNDcxNiA1Ljc2NDk1QzUuNTc2IDUuNzE3NzQgNS44MTk1NSA1LjQzNTA4IDUuODE3NjQgNS4xMDI4OEw1LjE1MjY1IDUuMTA2N1pNNS4xNTI0OSA1LjA3Nzk3SDQuNDg3NDdMNC40ODc1IDUuMDgxNzlMNS4xNTI0OSA1LjA3Nzk3Wk0xNC44NDc1IDUuMDc3OTdMMTUuNTEyNSA1LjA4MTc5VjUuMDc3OTdIMTQuODQ3NVpNMTQuODQ3MyA1LjEwNjdMMTQuMTgyNCA1LjEwMjg4QzE0LjE4MDUgNS40MzUwOSAxNC40MjQgNS43MTc3NCAxNC43NTI4IDUuNzY0OTVMMTQuODQ3MyA1LjEwNjdaTTE2LjQyMjUgOS4xNDAwNkwxNS45NTYxIDguNjY2MDVDMTUuODI5MSA4Ljc5MTA2IDE1Ljc1NzUgOC45NjE4MyAxNS43NTc1IDkuMTQwMDZDMTUuNzU3NSA5LjMxODI5IDE1LjgyOTEgOS40ODkwNiAxNS45NTYxIDkuNjE0MDZMMTYuNDIyNSA5LjE0MDA2Wk0xNS40NDQ5IDE0LjYzMDFMMTUuMjA4NSAxNC4wMDg1QzE0Ljk5NDIgMTQuMDkgMTQuODM3IDE0LjI3NjIgMTQuNzkyNSAxNC41MDExTDE1LjQ0NDkgMTQuNjMwMVpNNC41NTUxIDE0LjYzMDFMNS4yMDc0OCAxNC41MDExQzUuMTYzMDIgMTQuMjc2MiA1LjAwNTgxIDE0LjA5IDQuNzkxNDkgMTQuMDA4NUw0LjU1NTEgMTQuNjMwMVpNMy4wODc0MyA4LjcxOTQ3QzIuODI1MTEgOC45NzY1MyAyLjgyMDg0IDkuMzk3NTYgMy4wNzc4OSA5LjY1OTg4QzMuMzM0OTQgOS45MjIyIDMuNzU1OTcgOS45MjY0NyA0LjAxODI5IDkuNjY5NDFMMy4wODc0MyA4LjcxOTQ3Wk01LjUyNjA2IDkuMDUzODlDNS44OTMzMyA5LjA1Mzg5IDYuMTkxMDYgOC43NTYxNiA2LjE5MTA2IDguMzg4ODlDNi4xOTEwNiA4LjAyMTYyIDUuODkzMzMgNy43MjM4OSA1LjUyNjA2IDcuNzIzODlWOS4wNTM4OVpNOS41MzQ1NyAxMC4zMzA2QzkuMjcyMjUgMTAuNTg3NiA5LjI2Nzk4IDExLjAwODcgOS41MjUwMyAxMS4yNzFDOS43ODIwOCAxMS41MzMzIDEwLjIwMzEgMTEuNTM3NiAxMC40NjU0IDExLjI4MDVMOS41MzQ1NyAxMC4zMzA2Wk0xMS45NzMyIDEwLjY2NUMxMi4zNDA1IDEwLjY2NSAxMi42MzgyIDEwLjM2NzMgMTIuNjM4MiAxMEMxMi42MzgyIDkuNjMyNzMgMTIuMzQwNSA5LjMzNSAxMS45NzMyIDkuMzM1VjEwLjY2NVpNMTcuOTEyMyAxMi4wMTQxQzE3LjkxMjMgMTEuNjQ2OCAxNy42MTQ2IDExLjM0OTEgMTcuMjQ3MyAxMS4zNDkxQzE2Ljg4IDExLjM0OTEgMTYuNTgyMyAxMS42NDY4IDE2LjU4MjMgMTIuMDE0MUgxNy45MTIzWk0xMy41MDQ5IDEzLjk2MTZDMTMuMTczMSAxMy44MDQgMTIuNzc2NSAxMy45NDUxIDEyLjYxODkgMTQuMjc2OUMxMi40NjEzIDE0LjYwODYgMTIuNjAyNCAxNS4wMDUzIDEyLjkzNDIgMTUuMTYyOUwxMy41MDQ5IDEzLjk2MTZaTTcuODY4NjggMTEuNTQ0NUM3LjUzNjk0IDExLjM4NjkgNy4xNDAyNiAxMS41MjgxIDYuOTgyNjcgMTEuODU5OEM2LjgyNTA3IDEyLjE5MTUgNi45NjYyNSAxMi41ODgyIDcuMjk3OTkgMTIuNzQ1OEw3Ljg2ODY4IDExLjU0NDVaTTEwLjI4NTMgMTIuNzQ1OEMxMC42MTcxIDEyLjU4ODIgMTAuNzU4MyAxMi4xOTE1IDEwLjYwMDcgMTEuODU5OEMxMC40NDMxIDExLjUyODEgMTAuMDQ2NCAxMS4zODY5IDkuNzE0NjUgMTEuNTQ0NUwxMC4yODUzIDEyLjc0NThaTTguMTAwODUgNi4xNzY4OEM3Ljc2OTExIDYuMzM0NDggNy42Mjc5MyA2LjczMTE2IDcuNzg1NTIgNy4wNjI5QzcuOTQzMTIgNy4zOTQ2NCA4LjMzOTggNy41MzU4MSA4LjY3MTU0IDcuMzc4MjJMOC4xMDA4NSA2LjE3Njg4Wk05LjU5NDUzIDcuMTcxMjNDOS45NjE3OSA3LjE3MTIzIDEwLjI1OTUgNi44NzM1IDEwLjI1OTUgNi41MDYyM0MxMC4yNTk1IDYuMTM4OTYgOS45NjE3OSA1Ljg0MTIzIDkuNTk0NTMgNS44NDEyM1Y3LjE3MTIzWk0xNS40OTk0IDUuMTY2NjdDMTUuNDk5NCA0Ljc5OTQgMTUuMjAxNyA0LjUwMTY3IDE0LjgzNDQgNC41MDE2N0MxNC40NjcxIDQuNTAxNjcgMTQuMTY5NCA0Ljc5OTQgMTQuMTY5NCA1LjE2NjY3SDE1LjQ5OTRaTTEzLjE4ODggNi43NDAxQzEyLjg1OTIgNi45MDIwNiAxMi43MjMyIDcuMzAwNTcgMTIuODg1MiA3LjYzMDJDMTMuMDQ3MiA3Ljk1OTgzIDEzLjQ0NTcgOC4wOTU3NiAxMy43NzUzIDcuOTMzOEwxMy4xODg4IDYuNzQwMVpNNS4xNTI0OSA1LjA3Nzk3TDQuNDg3NSA1LjA4MTc5TDQuNDg3NjYgNS4xMTA1Mkw1LjE1MjY1IDUuMTA2N0w1LjgxNzY0IDUuMTAyODhMNS44MTc0NyA1LjA3NDE0TDUuMTUyNDkgNS4wNzc5N1pNMTQuODQ3MyA1LjEwNjdMMTUuNTEyMyA1LjExMDUyTDE1LjUxMjUgNS4wODE3OUwxNC44NDc1IDUuMDc3OTdMMTQuMTgyNSA1LjA3NDE1TDE0LjE4MjQgNS4xMDI4OEwxNC44NDczIDUuMTA2N1pNMTAgNS4xNDI5NUg5LjMzNVYxNC44NDQxSDEwSDEwLjY2NVY1LjE0Mjk1SDEwWk00LjU1NTEgMTQuNjMwMUwzLjkwMjcyIDE0Ljc1OUM0LjI1NTQyIDE2LjU0MzYgNS42NDA2MSAxNy43MzI0IDcuMTI2NzkgMTcuODk1OEM3Ljg3NjQgMTcuOTc4MyA4LjY1MDcgMTcuNzk3MiA5LjI5OTM4IDE3LjI5NzJDOS45NDc1NSAxNi43OTc1IDEwLjQyMTEgMTYuMDE4NCAxMC42NDkzIDE0Ljk4NzlMMTAgMTQuODQ0MUw5LjM1MDczIDE0LjcwMDNDOS4xNzYxMyAxNS40ODg4IDguODQyMzQgMTUuOTcwMiA4LjQ4NzQxIDE2LjI0MzhDOC4xMzMgMTYuNTE3IDcuNzA4NDQgMTYuNjIxOCA3LjI3MjE5IDE2LjU3MzhDNi4zODY2NSAxNi40NzY0IDUuNDUyMTcgMTUuNzM5MiA1LjIwNzQ4IDE0LjUwMTFMNC41NTUxIDE0LjYzMDFaTTEwIDE0Ljg0NDFMOS4zNTA3MyAxNC45ODc5QzkuNTc4OTEgMTYuMDE4NCAxMC4wNTI0IDE2Ljc5NzUgMTAuNzAwNiAxNy4yOTcyQzExLjM0OTMgMTcuNzk3MiAxMi4xMjM2IDE3Ljk3ODMgMTIuODczMiAxNy44OTU4QzE0LjM1OTQgMTcuNzMyNCAxNS43NDQ2IDE2LjU0MzYgMTYuMDk3MyAxNC43NTlMMTUuNDQ0OSAxNC42MzAxTDE0Ljc5MjUgMTQuNTAxMUMxNC41NDc4IDE1LjczOTIgMTMuNjEzNCAxNi40NzY0IDEyLjcyNzggMTYuNTczOEMxMi4yOTE2IDE2LjYyMTggMTEuODY3IDE2LjUxNyAxMS41MTI2IDE2LjI0MzhDMTEuMTU3NyAxNS45NzAyIDEwLjgyMzkgMTUuNDg4OCAxMC42NDkzIDE0LjcwMDNMMTAgMTQuODQ0MVpNMy41Nzc0NiA5LjE0MDA2TDMuMTExMDQgOC42NjYwNUMyLjE4MzUyIDkuNTc4NzIgMS45NDc1OCAxMS4wMjEzIDIuMTU1NzYgMTIuMjU1NUMyLjM2MzU0IDEzLjQ4NzQgMy4wNjEwNiAxNC43NzMzIDQuMzE4NzEgMTUuMjUxNkw0LjU1NTEgMTQuNjMwMUw0Ljc5MTQ5IDE0LjAwODVDNC4xNDM3OCAxMy43NjIyIDMuNjMyMTMgMTMuMDExOSAzLjQ2NzI0IDEyLjAzNDNDMy4zMDI3NiAxMS4wNTkyIDMuNTMwNjggMTAuMTE5IDQuMDQzODcgOS42MTQwNkwzLjU3NzQ2IDkuMTQwMDZaTTUuMTUyNjUgNS4xMDY3TDUuMDU4MTQgNC40NDg0NUMzLjczMjczIDQuNjM4NzQgMi42OTk2OCA1LjM2NTc4IDIuMjgyMDcgNi4zOTkzNkMxLjg1NTIzIDcuNDU1NzggMi4xNDEwNiA4LjY1OTYxIDMuMTExMDQgOS42MTQwNkwzLjU3NzQ2IDkuMTQwMDZMNC4wNDM4NyA4LjY2NjA1QzMuMzkyODIgOC4wMjU0MiAzLjMyMTU1IDcuMzc2OTIgMy41MTUyMSA2Ljg5NzYxQzMuNzE4MTEgNi4zOTU0NSA0LjI4MzE3IDUuOTAzMzUgNS4yNDcxNiA1Ljc2NDk1TDUuMTUyNjUgNS4xMDY3Wk0xNi40MjI1IDkuMTQwMDZMMTYuODg5IDkuNjE0MDZDMTcuODU4OSA4LjY1OTYxIDE4LjE0NDggNy40NTU3OCAxNy43MTc5IDYuMzk5MzZDMTcuMzAwMyA1LjM2NTc4IDE2LjI2NzMgNC42Mzg3NCAxNC45NDE5IDQuNDQ4NDVMMTQuODQ3MyA1LjEwNjdMMTQuNzUyOCA1Ljc2NDk1QzE1LjcxNjggNS45MDMzNSAxNi4yODE5IDYuMzk1NDUgMTYuNDg0OCA2Ljg5NzYxQzE2LjY3ODUgNy4zNzY5MiAxNi42MDcyIDguMDI1NDIgMTUuOTU2MSA4LjY2NjA1TDE2LjQyMjUgOS4xNDAwNlpNMTUuNDQ0OSAxNC42MzAxTDE1LjY4MTMgMTUuMjUxNkMxNi45Mzg5IDE0Ljc3MzMgMTcuNjM2NSAxMy40ODc0IDE3Ljg0NDIgMTIuMjU1NkMxOC4wNTI0IDExLjAyMTMgMTcuODE2NSA5LjU3ODcyIDE2Ljg4OSA4LjY2NjA1TDE2LjQyMjUgOS4xNDAwNkwxNS45NTYxIDkuNjE0MDZDMTYuNDY5MyAxMC4xMTkgMTYuNjk3MiAxMS4wNTkyIDE2LjUzMjggMTIuMDM0M0MxNi4zNjc5IDEzLjAxMTkgMTUuODU2MiAxMy43NjIyIDE1LjIwODUgMTQuMDA4NUwxNS40NDQ5IDE0LjYzMDFaTTE0Ljg0NzUgNS4wNzc5N0gxNS41MTI1QzE1LjUxMjUgNC4xMDY4NCAxNS4xMjk0IDMuMzM0MzMgMTQuNTIxNyAyLjgxNTg3QzEzLjkyODEgMi4zMDk0MSAxMy4xNTk1IDIuMDc3MzUgMTIuNDE2NyAyLjA4NTE5QzExLjY3MzcgMi4wOTMwNCAxMC45MDgxIDIuMzQxMzkgMTAuMzE4NiAyLjg1ODk5QzkuNzE2MTUgMy4zODgwMyA5LjMzNSA0LjE2NzExIDkuMzM1IDUuMTQyOTVIMTBIMTAuNjY1QzEwLjY2NSA0LjUzODM2IDEwLjg4OTggNC4xMjc0NCAxMS4xOTYyIDMuODU4MzlDMTEuNTE1NiAzLjU3NzkxIDExLjk2MiAzLjQyMDA3IDEyLjQzMDggMy40MTUxMkMxMi44OTk5IDMuNDEwMTcgMTMuMzQzMiAzLjU1ODY5IDEzLjY1ODUgMy44Mjc2N0MxMy45NTk3IDQuMDg0NjYgMTQuMTgyNSA0LjQ4MjQ1IDE0LjE4MjUgNS4wNzc5N0gxNC44NDc1Wk0xMCA1LjE0Mjk1SDEwLjY2NUMxMC42NjUgNC4xNjcxMSAxMC4yODM5IDMuMzg4MDMgOS42ODEzNSAyLjg1ODk5QzkuMDkxODcgMi4zNDEzOCA4LjMyNjMzIDIuMDkzMDQgNy41ODMyNyAyLjA4NTE5QzYuODQwNTEgMi4wNzczNSA2LjA3MTkzIDIuMzA5NDEgNS40NzgzIDIuODE1ODdDNC44NzA2MSAzLjMzNDMzIDQuNDg3NDkgNC4xMDY4NCA0LjQ4NzQ5IDUuMDc3OTdINS4xNTI0OUg1LjgxNzQ5QzUuODE3NDkgNC40ODI0NSA2LjA0MDMgNC4wODQ2NiA2LjM0MTUyIDMuODI3NjdDNi42NTY4IDMuNTU4NjkgNy4xMDAxIDMuNDEwMTYgNy41NjkyMiAzLjQxNTEyQzguMDM4MDMgMy40MjAwNyA4LjQ4NDM3IDMuNTc3OTEgOC44MDM4IDMuODU4MzlDOS4xMTAyMSA0LjEyNzQ0IDkuMzM1IDQuNTM4MzYgOS4zMzUgNS4xNDI5NUgxMFpNMy41NTI4NiA5LjE5NDQ0TDQuMDE4MjkgOS42Njk0MUM0LjQwNzU1IDkuMjg3OTcgNC45Mzg4MSA5LjA1Mzg5IDUuNTI2MDYgOS4wNTM4OVY4LjM4ODg5VjcuNzIzODlDNC41NzY4NyA3LjcyMzg5IDMuNzE1MjEgOC4xMDQzIDMuMDg3NDMgOC43MTk0N0wzLjU1Mjg2IDkuMTk0NDRaTTEwIDEwLjgwNTZMMTAuNDY1NCAxMS4yODA1QzEwLjg1NDcgMTAuODk5MSAxMS4zODU5IDEwLjY2NSAxMS45NzMyIDEwLjY2NVYxMFY5LjMzNUMxMS4wMjQgOS4zMzUgMTAuMTYyNCA5LjcxNTQxIDkuNTM0NTcgMTAuMzMwNkwxMCAxMC44MDU2Wk0xNy4yNDczIDEyLjAxNDFIMTYuNTgyM0MxNi41ODIzIDEzLjIwNCAxNS42MTc3IDE0LjE2ODYgMTQuNDI3OSAxNC4xNjg2VjE0LjgzMzZWMTUuNDk4NkMxNi4zNTIzIDE1LjQ5ODYgMTcuOTEyMyAxMy45Mzg1IDE3LjkxMjMgMTIuMDE0MUgxNy4yNDczWk0xNC40Mjc5IDE0LjgzMzZWMTQuMTY4NkMxNC4wOTYyIDE0LjE2ODYgMTMuNzgzOCAxNC4wOTQxIDEzLjUwNDkgMTMuOTYxNkwxMy4yMTk1IDE0LjU2MjJMMTIuOTM0MiAxNS4xNjI5QzEzLjM4NzcgMTUuMzc4MyAxMy44OTQ3IDE1LjQ5ODYgMTQuNDI3OSAxNS40OTg2VjE0LjgzMzZaTTguNzkxNjcgMTIuNDE2NVYxMS43NTE1QzguNDYwMDIgMTEuNzUxNSA4LjE0NzYzIDExLjY3NyA3Ljg2ODY4IDExLjU0NDVMNy41ODMzMyAxMi4xNDUyTDcuMjk3OTkgMTIuNzQ1OEM3Ljc1MTQ5IDEyLjk2MTMgOC4yNTg0NyAxMy4wODE1IDguNzkxNjcgMTMuMDgxNVYxMi40MTY1Wk0xMCAxMi4xNDUyTDkuNzE0NjUgMTEuNTQ0NUM5LjQzNTcgMTEuNjc3IDkuMTIzMzEgMTEuNzUxNSA4Ljc5MTY3IDExLjc1MTVWMTIuNDE2NVYxMy4wODE1QzkuMzI0ODcgMTMuMDgxNSA5LjgzMTg0IDEyLjk2MTMgMTAuMjg1MyAxMi43NDU4TDEwIDEyLjE0NTJaTTguMzg2MTkgNi43Nzc1NUw4LjY3MTU0IDcuMzc4MjJDOC45NTA0OSA3LjI0NTcxIDkuMjYyODggNy4xNzEyMyA5LjU5NDUzIDcuMTcxMjNWNi41MDYyM1Y1Ljg0MTIzQzkuMDYxMzMgNS44NDEyMyA4LjU1NDM1IDUuOTYxNDUgOC4xMDA4NSA2LjE3Njg4TDguMzg2MTkgNi43Nzc1NVpNMTQuODM0NCA1LjE2NjY3SDE0LjE2OTRDMTQuMTY5NCA1Ljg1NjI2IDEzLjc3MSA2LjQ1NDA1IDEzLjE4ODggNi43NDAxTDEzLjQ4MiA3LjMzNjk1TDEzLjc3NTMgNy45MzM4QzE0Ljc5NTEgNy40MzI3MiAxNS40OTk0IDYuMzgyNTYgMTUuNDk5NCA1LjE2NjY3SDE0LjgzNDRaIiBmaWxsPSJjdXJyZW50Q29sb3IiPjwvcGF0aD48L3N2Zz4K
"""

from __future__ import annotations
import aiohttp
import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, List, Optional
from pydantic import BaseModel, Field


class Action:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""),
            description="OpenAI API key (or set OPENAI_API_KEY environment variable)"
        )
        OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API base URL"
        )
        TARGET_MODEL: str = Field(
            default="gpt-5",
            description="The model to switch to (e.g., 'gpt-5', 'gpt-4o', 'o3-mini')"
        )
        REASONING_EFFORT: str = Field(
            default="high",
            description="Reasoning effort level ('minimal', 'high', or leave empty for default). Only applies to reasoning models."
        )
        DEBUG: bool = Field(
            default=False,
            description="Show debug information in logs (useful when troubleshooting)"
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_or_create_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session for API calls."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def build_conversation_input(self, messages: List[dict]) -> List[dict]:
        """
        Convert Open WebUI messages to OpenAI Responses API input format.

        Excludes the last assistant message (which is being regenerated).
        Strips system messages (they should go in 'instructions' parameter).
        Strips status blocks from assistant messages.
        """
        if self.valves.DEBUG:
            self.logger.debug(f"=== RAW MESSAGES RECEIVED (total: {len(messages)}) ===")
            for i, msg in enumerate(messages):
                self.logger.debug(f"Message {i}: role={msg.get('role')}, content_type={type(msg.get('content'))}")
                if isinstance(msg.get('content'), list):
                    for j, block in enumerate(msg.get('content', [])):
                        self.logger.debug(f"  Block {j}: {block.get('type')} - {list(block.keys())}")

        openai_input = []

        # Process all messages except the last one (which should be the assistant message being regenerated)
        for msg in messages[:-1]:
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (they belong in 'instructions', not 'input')
            if role == "system":
                continue

            # User message
            elif role == "user":
                # Convert string content to a block list
                content_blocks = content or []
                if isinstance(content_blocks, str):
                    content_blocks = [{"type": "text", "text": content_blocks}]

                # Transform each block to Responses API format
                block_transform = {
                    "text":       lambda b: {"type": "input_text",  "text": b.get("text", "")},
                    "image_url":  lambda b: {"type": "input_image", "image_url": b.get("image_url", {}).get("url")},
                    "input_file": lambda b: {"type": "input_file",  "file_id": b.get("file_id")},
                }

                openai_input.append({
                    "role": "user",
                    "content": [
                        block_transform.get(block.get("type"), lambda b: b)(block)
                        for block in content_blocks if block
                    ],
                })

            # Assistant message
            elif role == "assistant":
                # Strip <details> blocks and clean content
                clean_content = re.sub(r'<details[^>]*>.*?</details>', '', content, flags=re.DOTALL).strip()
                if clean_content:
                    openai_input.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": clean_content}]
                    })

        if self.valves.DEBUG:
            self.logger.debug(f"=== TRANSFORMED CONVERSATION INPUT (total: {len(openai_input)} messages) ===")
            for i, msg in enumerate(openai_input):
                self.logger.debug(f"Transformed {i}: role={msg.get('role')}")
                content_items = msg.get('content', [])
                if isinstance(content_items, list):
                    for j, item in enumerate(content_items):
                        item_type = item.get('type', 'unknown')
                        if item_type == 'input_image':
                            self.logger.debug(f"  Content {j}: {item_type} - URL length: {len(item.get('image_url', ''))}")
                        else:
                            self.logger.debug(f"  Content {j}: {item_type}")

        return openai_input

    async def action(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __event_call__: Optional[Callable[[dict[str, Any]], Awaitable[Any]]] = None,
    ) -> dict:
        """
        Action: Switch to pro model and regenerate the response.

        Makes a direct HTTP request to OpenAI Responses API with full conversation context.
        """
        # Validate API key
        if not self.valves.OPENAI_API_KEY:
            error_msg = "OPENAI_API_KEY is not set. Please configure it in the action valves or environment variables."
            self.logger.error(error_msg)
            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": error_msg
                }
            })
            # Modify body to show error
            for i in range(len(body.get("messages", [])) - 1, -1, -1):
                if body["messages"][i].get("role") == "assistant":
                    body["messages"][i]["content"] = f"Error: {error_msg}"
                    break
            return body

        # Get configuration
        model_name = self.valves.TARGET_MODEL
        reasoning_effort = self.valves.REASONING_EFFORT.strip()

        if self.valves.DEBUG:
            self.logger.info("=== PRO MODE ACTION ===")
            self.logger.info(f"Target model: {model_name}")
            self.logger.info(f"Reasoning effort: {reasoning_effort}")

        # Extract messages and convert to Responses API format
        messages = body.get("messages", [])
        if not messages:
            error_msg = "No messages found in request"
            self.logger.error(error_msg)
            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": error_msg
                }
            })
            return body  # Return body as-is if no messages

        # Convert conversation to Responses API format (includes full context)
        conversation_input = self.build_conversation_input(messages)

        if not conversation_input:
            error_msg = "No conversation context found"
            self.logger.error(error_msg)
            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": error_msg
                }
            })
            # Modify body to show error
            for i in range(len(body.get("messages", [])) - 1, -1, -1):
                if body["messages"][i].get("role") == "assistant":
                    body["messages"][i]["content"] = f"Error: {error_msg}"
                    break
            return body

        # Build Responses API request body with full conversation context
        request_body = {
            "model": model_name,
            "input": conversation_input,  # Full conversation history!
            "stream": False
        }

        # Add reasoning effort if specified
        if reasoning_effort:
            request_body["reasoning"] = {"effort": reasoning_effort}

        if self.valves.DEBUG:
            # Log summary of what's being sent
            self.logger.debug(f"=== FINAL REQUEST TO OPENAI ===")
            self.logger.debug(f"Model: {request_body.get('model')}")
            self.logger.debug(f"Input messages: {len(request_body.get('input', []))}")

            # Log details about images in the input
            for i, msg in enumerate(request_body.get('input', [])):
                content_list = msg.get('content', [])
                if isinstance(content_list, list):
                    image_count = sum(1 for item in content_list if item.get('type') == 'input_image')
                    text_count = sum(1 for item in content_list if item.get('type') == 'input_text')
                    if image_count > 0:
                        self.logger.debug(f"  Message {i} ({msg.get('role')}): {text_count} text blocks, {image_count} image(s)")

            # Full request body for detailed inspection
            self.logger.debug(f"Full request body: {json.dumps(request_body, indent=2)}")

        # Show status to user
        await __event_emitter__({
            "type": "status",
            "data": {
                "description": "Request for gpt-5 pro mode...",
                "done": False
            }
        })

        # Make API call to OpenAI Responses API
        try:
            session = await self._get_or_create_session()
            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            url = f"{self.valves.OPENAI_BASE_URL.rstrip('/')}/responses"

            async with session.post(url, json=request_body, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"OpenAI API error (status {response.status}): {error_text}"
                    self.logger.error(error_msg)

                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "Generation failed",
                            "done": True
                        }
                    })

                    await __event_emitter__({
                        "type": "notification",
                        "data": {
                            "type": "error",
                            "content": f"Failed to generate response: {error_text[:200]}"
                        }
                    })

                    # Modify body to show error
                    for i in range(len(body.get("messages", [])) - 1, -1, -1):
                        if body["messages"][i].get("role") == "assistant":
                            body["messages"][i]["content"] = f"Error: {error_text}"
                            break
                    return body

                response_data = await response.json()

                if self.valves.DEBUG:
                    self.logger.debug(f"Response data: {json.dumps(response_data, indent=2)}")

                # Extract content from Responses API response
                # Response structure: {"output": [{"type": "message", "content": [{"type": "output_text", "text": "..."}]}]}
                output = response_data.get("output", [])

                # Collect all text from message items
                content_parts = []
                for item in output:
                    if item.get("type") != "message":
                        continue
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            content_parts.append(content.get("text", ""))

                content_result = "\n".join(content_parts).strip()

                # Show completion status
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Generation complete",
                        "done": True
                    }
                })

                if not content_result:
                    error_msg = "No content in response from OpenAI"
                    self.logger.error(error_msg)
                    await __event_emitter__({
                        "type": "notification",
                        "data": {
                            "type": "error",
                            "content": error_msg
                        }
                    })
                    # Return error by modifying body
                    for i in range(len(body.get("messages", [])) - 1, -1, -1):
                        if body["messages"][i].get("role") == "assistant":
                            body["messages"][i]["content"] = f"Error: {error_msg}"
                            break
                    return body

                # Find the last assistant message and replace its content
                message_replaced = False
                for i in range(len(body.get("messages", [])) - 1, -1, -1):
                    if body["messages"][i].get("role") == "assistant":
                        body["messages"][i]["content"] = content_result
                        message_replaced = True
                        break

                if not message_replaced:
                    # No assistant message found, append new one
                    body["messages"].append({
                        "role": "assistant",
                        "content": content_result
                    })

                # Return the modified body (not just {"content": "..."})
                return body

        except Exception as e:
            error_msg = f"Failed to call OpenAI API: {str(e)}"
            self.logger.error(error_msg)
            self.logger.exception("Full error:")

            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "Generation failed",
                    "done": True
                }
            })

            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": error_msg
                }
            })

            # Modify body to show error
            for i in range(len(body.get("messages", [])) - 1, -1, -1):
                if body["messages"][i].get("role") == "assistant":
                    body["messages"][i]["content"] = f"Error: {error_msg}"
                    break
            return body
