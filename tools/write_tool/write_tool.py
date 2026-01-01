"""
title: Write Tool
id: write_tool
version: 0.1.0
description: A writing tool that calls Claude via OpenRouter to generate prose, documents, and emails.
license: MIT
"""

import aiohttp
from pydantic import BaseModel, Field


class Tools:
    """Write tool that calls Claude via OpenRouter to generate content."""

    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="Your OpenRouter API key",
        )
        MODEL: str = Field(
            default="anthropic/claude-sonnet-4",
            description="Model to use for writing (e.g., anthropic/claude-sonnet-4)",
        )
        SYSTEM_PROMPT: str = Field(
            default="You are an expert writer. Write the requested content clearly and professionally. Output only the written content, no preamble.",
            description="System prompt for the writing model",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def write(self, instructions: str) -> str:
        """
        Write content based on the provided instructions.

        The orchestrator provides all necessary context and instructions,
        and this tool generates the written content using Claude.

        :param instructions: Complete instructions for what to write, including content, tone, format, etc.
        :return: The generated written content.
        """
        if not self.valves.OPENROUTER_API_KEY:
            return "Error: OpenRouter API key not configured. Please set it in the tool's Valves settings."

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.valves.MODEL,
                    "messages": [
                        {"role": "system", "content": self.valves.SYSTEM_PROMPT},
                        {"role": "user", "content": instructions},
                    ],
                },
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    return f"Error generating content: {error}"

                data = await response.json()
                return data["choices"][0]["message"]["content"]
