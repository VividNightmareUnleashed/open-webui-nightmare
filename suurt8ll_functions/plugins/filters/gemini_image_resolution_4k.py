"""
title: 4K Images
id: gemini_image_resolution_4k
description: Forces 4K image generation resolution for Gemini 3 Pro Image (Gemini Manifold)
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.0.0
"""

from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

MANIFOLD_PREFIX = "gemini_manifold_google_genai."

if TYPE_CHECKING:
    from utils.manifold_types import Body, Features


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Makes the filter toggleable in the front-end.
        self.toggle = True
        # Icon from https://icon-sets.iconify.design/material-symbols/4k/
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0yMSAzSDNjLTEuMSAwLTIgLjktMiAydjE0YzAgMS4xLjkgMiAyIDJoMThjMS4xIDAgMi0uOSAyLTJWNUMyMyAzLjkgMjIuMSAzIDIxIDNNOSAxNkg3djJoLTJ2LTJoLTV2LTJsNS02aDJ2NmgyVjE2bS00IDBWNTAuMTdsLTIuMzMgMi44M0g1LjY3VjE2em03IDJoLTJWN0gxMnY0LjQ3TDE0LjMzIDdIMTd2NC42N2wtMi4zMyAyLjgzTDE3IDE4aC0yLjY3bC0yLjMzLTIuODNWMThabTAgLTMuNTNsMS4zMy0xLjU3SDEyVjE0LjQ3bDEuMzMgMS41N0gxMloiLz48L3N2Zz4="

    async def inlet(self, body: "Body") -> "Body":
        """
        Signals downstream Gemini Manifold pipe that 4K image resolution is desired.

        The pipe uses the toggle state to enforce 4K; this metadata flag is just an
        additional hint for debugging/forward-compatibility.
        """
        model_id = body.get("model", "")
        if not isinstance(model_id, str) or not model_id.startswith(MANIFOLD_PREFIX):
            return body

        metadata = body.get("metadata") or {}
        metadata_features = metadata.get("features")
        if metadata_features is None:
            metadata_features = cast("Features", {})
            metadata["features"] = metadata_features

        metadata_features["gemini_image_resolution_4k"] = True
        body["metadata"] = metadata
        return body

