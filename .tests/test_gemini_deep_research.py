import asyncio

import aiohttp

from functions.pipes.gemini_deep_research import gemini_deep_research as mod


async def _collect(gen):
    out = []
    async for chunk in gen:
        out.append(chunk)
    return out


def test_stream_drop_dedupes_final_output(monkeypatch):
    pipe = mod.Pipe()

    async def fake_create(_session, *, api_key, query, previous_interaction_id):
        return "i1"

    async def fake_stream(_session, *, api_key, interaction_id, last_event_id):
        assert interaction_id == "i1"
        assert last_event_id is None
        yield {
            "event_type": "content.delta",
            "event_id": "1",
            "delta": {"type": "text", "text": "Hello "},
        }

    async def fake_get(_session, *, api_key, interaction_id):
        assert interaction_id == "i1"
        return {"status": "completed", "outputs": [{"text": "Hello world"}]}

    monkeypatch.setattr(pipe, "_create_interaction", fake_create)
    monkeypatch.setattr(pipe, "_stream_interaction_sse", fake_stream)
    monkeypatch.setattr(pipe, "_get_interaction", fake_get)

    chunks = asyncio.run(
        _collect(
            pipe._run_research_streaming(
                query="q",
                api_key="k",
                event_emitter=None,
                previous_interaction_id=None,
                metadata=None,
            )
        )
    )
    assert chunks == ["__INTERACTION_ID__:i1", "Hello ", "world"]


def test_stream_reconnect_uses_last_event_id(monkeypatch):
    pipe = mod.Pipe()

    async def fast_sleep(_seconds):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", fast_sleep)

    async def fake_create(_session, *, api_key, query, previous_interaction_id):
        return "i1"

    calls: list[str | None] = []

    async def fake_stream(_session, *, api_key, interaction_id, last_event_id):
        calls.append(last_event_id)
        if len(calls) == 1:
            assert last_event_id is None
            yield {
                "event_type": "content.delta",
                "event_id": "10",
                "delta": {"type": "text", "text": "Part1"},
            }
            raise aiohttp.ClientError("deadline expired")

        assert last_event_id == "10"
        yield {
            "event_type": "content.delta",
            "event_id": "11",
            "delta": {"type": "text", "text": "Part2"},
        }
        yield {"event_type": "interaction.complete", "event_id": "12"}

    async def fake_get(_session, *, api_key, interaction_id):
        return {"status": "completed", "outputs": [{"text": "Part1Part2"}]}

    monkeypatch.setattr(pipe, "_create_interaction", fake_create)
    monkeypatch.setattr(pipe, "_stream_interaction_sse", fake_stream)
    monkeypatch.setattr(pipe, "_get_interaction", fake_get)

    chunks = asyncio.run(
        _collect(
            pipe._run_research_streaming(
                query="q",
                api_key="k",
                event_emitter=None,
                previous_interaction_id=None,
                metadata=None,
            )
        )
    )

    assert calls == [None, "10"]
    assert chunks == ["__INTERACTION_ID__:i1", "Part1", "Part2"]
