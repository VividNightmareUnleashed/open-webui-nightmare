import asyncio

from functions.actions.pro_regenerate_action import pro_regenerate_action as mod


def test_regenerates_via_openai_responses_pipe(monkeypatch):
    events: list[dict] = []

    async def emitter(event: dict) -> None:
        events.append(event)

    class DummyModelInfo:
        def model_dump(self) -> dict:
            return {"id": "openai_responses.gpt-5.2-pro"}

    class DummyPipe:
        def __init__(self) -> None:
            self.seen_body: dict | None = None

        async def pipe(self, body, **kwargs):  # noqa: ANN001, ARG002
            self.seen_body = body

            async def gen():
                yield "hello"
                yield " world"

            return gen()

    class DummyProFilter:
        def __init__(self) -> None:
            self.valves = type("Valves", (), {"MODEL": "openai_responses.gpt-5.2-pro"})()

    pipe = DummyPipe()

    def fake_get_function_module_by_id(request, function_id: str):  # noqa: ARG001
        if function_id == "pro_filter":
            return DummyProFilter()
        if function_id == "openai_responses":
            return pipe
        raise AssertionError(f"Unexpected function id: {function_id}")

    monkeypatch.setattr(mod, "get_function_module_by_id", fake_get_function_module_by_id)
    monkeypatch.setattr(mod.Models, "get_model_by_id", lambda _model_id: DummyModelInfo())

    action = mod.Action()

    body = {
        "id": "m1",
        "chat_id": "c1",
        "session_id": "s1",
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A (old)"},
        ],
    }
    user = {"id": "u1"}

    out = asyncio.run(action.action(body, __user__=user, __request__=object(), __event_emitter__=emitter))

    assert out["messages"][0]["model"] == "openai_responses.gpt-5.2-pro"
    assert pipe.seen_body is not None
    assert pipe.seen_body["model"] == "openai_responses.gpt-5.2-pro"
    # Regeneration drops the existing assistant message from the prompt.
    assert pipe.seen_body["messages"] == [{"role": "user", "content": "Q"}]

    types = [e["type"] for e in events]
    assert "replace" in types
    assert "message" in types
    assert "chat:completion" in types

