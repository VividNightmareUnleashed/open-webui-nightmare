import asyncio

import json

from tools.form_prompt import form_prompt as mod


def test_prompt_form_requires_event_call():
    tool = mod.Tools()
    out = asyncio.run(
        tool.AskUserQuestion(
            schema={
                "title": "Test",
                "fields": [{"name": "x", "label": "X", "type": "text"}],
            }
        )
    )
    assert out["error"].startswith("Missing __event_call__")


def test_prompt_form_validates_schema():
    async def fake_call(event: dict):
        return event

    tool = mod.Tools()
    out = asyncio.run(tool.AskUserQuestion(schema={"title": "Bad", "fields": []}, __event_call__=fake_call))
    assert "Invalid schema" in out["error"]


def test_prompt_form_calls_execute_and_returns_result():
    seen_events: list[dict] = []
    seen_emits: list[dict] = []

    async def fake_call(event: dict):
        seen_events.append(event)
        code = (event.get("data") or {}).get("code") or ""
        if "JSON.parse(atob(" in code:
            return {"status": "opened"}
        return {"cancelled": False, "values": {"destination": "Tokyo"}}

    async def fake_emit(event: dict) -> None:
        seen_emits.append(event)

    tool = mod.Tools()
    out = asyncio.run(
        tool.AskUserQuestion(
            schema={
                "title": "Trip Planner",
                "fields": [
                    {
                        "name": "destination",
                        "label": "Destination",
                        "type": "text",
                        "required": True,
                        "placeholder": "e.g. Tokyo",
                    }
                ],
            },
            __event_call__=fake_call,
            __event_emitter__=fake_emit,
        )
    )

    assert out == {"cancelled": False, "values": {"destination": "Tokyo"}}

    assert seen_events
    assert len(seen_events) >= 2
    assert seen_events[0]["type"] == "execute"
    assert "code" in seen_events[0]["data"]
    assert "JSON.parse(atob(" in seen_events[0]["data"]["code"]
    assert "Explain model what to do instead" in seen_events[0]["data"]["code"]
    assert "Additional info, if the user provided:" in seen_events[0]["data"]["code"]

    assert [e["type"] for e in seen_emits] == ["notification", "status", "status"]
    assert "would like you to answer some questions" in seen_emits[0]["data"]["content"]
    assert seen_emits[1]["data"]["done"] is False
    assert seen_emits[2]["data"]["done"] is True


def test_prompt_form_accepts_common_llm_schema_variants():
    seen_events: list[dict] = []

    async def fake_call(event: dict):
        seen_events.append(event)
        code = (event.get("data") or {}).get("code") or ""
        if "JSON.parse(atob(" in code:
            return {"status": "opened"}
        return {"cancelled": True}

    tool = mod.Tools()
    out = asyncio.run(
        tool.AskUserQuestion(
            schema={
                "title": "Test",
                "fields": [
                    {
                        "id": "level",
                        "type": "radio",
                        "options": [{"label": "Expert", "value": "Expert"}],
                    }
                ],
            },
            __event_call__=fake_call,
        )
    )

    assert out.get("cancelled") is True
    assert seen_events
    assert "JSON.parse(atob(" in (seen_events[0].get("data") or {}).get("code", "")


def test_prompt_form_accepts_json_string_schema():
    async def fake_call(event: dict):
        code = (event.get("data") or {}).get("code") or ""
        if "JSON.parse(atob(" in code:
            return {"status": "opened"}
        return {"cancelled": True}

    tool = mod.Tools()
    schema = json.dumps(
        {
            "title": "Test",
            "fields": [{"name": "x", "label": "X", "type": "text"}],
        }
    )

    out = asyncio.run(tool.AskUserQuestion(schema=schema, __event_call__=fake_call))
    assert out.get("cancelled") is True


def test_prompt_form_timeout_does_not_return_refusal():
    async def fake_call(event: dict):
        code = (event.get("data") or {}).get("code") or ""
        # First call opens the UI.
        if "JSON.parse(atob(" in code:
            return {"status": "opened"}
        # Poll/cleanup calls: no result yet.
        return None

    tool = mod.Tools()
    out = asyncio.run(
        tool.AskUserQuestion(
            schema={
                "title": "Test",
                "fields": [{"name": "x", "label": "X", "type": "text"}],
            },
            __event_call__=fake_call,
            timeout_seconds=0.2,
            poll_interval_ms=10,
        )
    )

    assert out.get("timeout") is True
    assert "Timed out" in (out.get("error") or "")
    assert "refusal" not in out
    assert "cancelled" not in out


def test_prompt_form_prefers_selected_model_name_from_metadata_for_notifications():
    seen_emits: list[dict] = []

    async def fake_call(event: dict):
        code = (event.get("data") or {}).get("code") or ""
        if "JSON.parse(atob(" in code:
            return {"status": "opened"}
        return {"cancelled": True}

    async def fake_emit(event: dict) -> None:
        seen_emits.append(event)

    tool = mod.Tools()
    out = asyncio.run(
        tool.AskUserQuestion(
            schema={
                "title": "Test",
                "fields": [{"name": "x", "label": "X", "type": "text"}],
            },
            __event_call__=fake_call,
            __event_emitter__=fake_emit,
            __model__={"name": "Task Model"},
            __metadata__={"model": {"name": "Selected Model"}},
        )
    )

    assert out.get("cancelled") is True
    assert seen_emits
    assert (
        seen_emits[0]["data"]["content"]
        == "Selected Model would like you to answer some questions."
    )
