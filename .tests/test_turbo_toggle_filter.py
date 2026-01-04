import asyncio


from functions.filters.turbo_toggle_filter import turbo_toggle_filter as mod


def test_sets_service_tier_for_openai_responses_models():
    filt = mod.Filter()
    body = {"model": "openai_responses.gpt-5"}
    metadata: dict = {}
    out = asyncio.run(filt.inlet(body, __metadata__=metadata))
    assert out["service_tier"] == "priority"
    assert metadata["features"]["openai_responses"]["service_tier"] == "priority"


def test_skips_non_openai_models_and_emits_status():
    events: list[dict] = []

    async def emitter(event: dict) -> None:
        events.append(event)

    filt = mod.Filter()
    body = {"model": "anthropic.claude-3-5-sonnet"}
    out = asyncio.run(filt.inlet(body, __event_emitter__=emitter))

    assert "service_tier" not in out
    assert events
    assert events[0]["type"] == "status"
