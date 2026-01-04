import asyncio


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
    assert isinstance(out, str)
    assert "WebSocket event calls" in out


def test_prompt_form_validates_schema():
    async def fake_call(event: dict):
        return event

    tool = mod.Tools()
    out = asyncio.run(tool.AskUserQuestion(schema={"title": "Bad", "fields": []}, __event_call__=fake_call))
    assert "Invalid schema" in out["error"]


def test_prompt_form_calls_execute_and_returns_result():
    seen_events: list[dict] = []
    seen_status: list[dict] = []

    async def fake_call(event: dict):
        seen_events.append(event)
        return {"cancelled": False, "values": {"destination": "Tokyo"}}

    async def fake_emit(event: dict) -> None:
        seen_status.append(event)

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
    assert seen_events[0]["type"] == "execute"
    assert "code" in seen_events[0]["data"]
    assert "JSON.parse(atob(" in seen_events[0]["data"]["code"]

    assert [e["type"] for e in seen_status] == ["status", "status"]
    assert seen_status[0]["data"]["done"] is False
    assert seen_status[1]["data"]["done"] is True


def test_prompt_form_accepts_common_llm_schema_variants():
    async def fake_call(event: dict):
        return event

    tool = mod.Tools()
    out = asyncio.run(
        tool.AskUserQuestion(
            schema={
                "title": "Test",
                "fields": [
                    {
                        "key": "level",
                        "type": "radio",
                        "options": [{"label": "Expert", "value": "Expert"}],
                    }
                ],
            },
            __event_call__=fake_call,
        )
    )

    assert out["type"] == "execute"
