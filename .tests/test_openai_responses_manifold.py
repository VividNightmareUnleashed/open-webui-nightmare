import asyncio
import json
from dataclasses import dataclass

import pytest

from functions.pipes.openai_responses_manifold import openai_responses_manifold as mod


@pytest.fixture()
def dummy_chats(monkeypatch):
    """Simple in-memory Chats stub."""
    storage: dict[str, dict] = {}

    @dataclass
    class DummyChatModel:
        chat: dict

    class DummyChats:
        @staticmethod
        def get_chat_by_id(cid):
            chat = storage.get(cid)
            if chat is None:
                return None
            return DummyChatModel(chat)

        @staticmethod
        def update_chat_by_id(cid, chat):
            storage[cid] = chat
            return DummyChatModel(chat)

    monkeypatch.setattr(mod, "Chats", DummyChats)
    return storage


def test_marker_roundtrip():
    marker = mod.create_marker("function_call", ulid="01HX4Y2VW5VR2Z2H", model_id="gpt-4o")
    wrapped = mod.wrap_marker(marker)
    assert mod.contains_marker(wrapped)

    parsed = mod.parse_marker(marker)
    assert parsed["metadata"]["model"] == "gpt-4o"

    text = f"pre {wrapped} post"
    assert mod.extract_markers(text) == [marker]

    segments = mod.split_text_by_markers(text)
    assert segments[1] == {"type": "marker", "marker": marker}
    assert segments[0]["text"].startswith("pre")
    assert segments[-1]["text"].strip().endswith("post")


def test_persistence_fetch_and_input(dummy_chats):
    dummy_chats["c1"] = {"history": {"messages": {}}}
    marker1 = mod.persist_openai_response_items(
        "c1",
        "m1",
        [{"type": "function_call", "name": "calc", "arguments": "{}"}],
        "openai_responses.gpt-4o",
    )
    marker2 = mod.persist_openai_response_items(
        "c1",
        "m2",
        [{"type": "function_call", "name": "other", "arguments": "{}"}],
        "openai_responses.gpt-3.5",
    )

    uid1 = mod.extract_markers(marker1, parsed=True)[0]["ulid"]
    uid2 = mod.extract_markers(marker2, parsed=True)[0]["ulid"]

    fetched = mod.fetch_openai_response_items(
        "c1", [uid1, uid2], openwebui_model_id="openai_responses.gpt-4o"
    )
    assert list(fetched) == [uid1]

    messages = [{"role": "assistant", "content": marker1 + "ok"}]
    output = mod.ResponsesBody.transform_messages_to_input(
        messages,
        chat_id="c1",
        openwebui_model_id="openai_responses.gpt-4o",
    )
    assert output[0]["type"] == "function_call"
    assert output[1]["content"][0]["text"] == "ok"


def test_tool_transforms_and_mcp():
    tools = [
        {"spec": {"name": "add", "description": "", "parameters": {}}},
        {"type": "function", "function": {"name": "add", "parameters": {}}},
        {"type": "web_search"},
    ]
    out = mod.ResponsesBody.transform_tools(tools, strict=True)
    names = {t.get("name", t.get("type")) for t in out}
    assert names == {"add", "web_search"}
    for t in out:
        if t.get("type") == "function":
            assert t["strict"] is True
            assert t["parameters"]["additionalProperties"] is False

    mcp_json = json.dumps({"server_label": "main", "server_url": "https://x.y"})
    assert mod.ResponsesBody._build_mcp_tools(mcp_json) == [
        {"type": "mcp", "server_label": "main", "server_url": "https://x.y"}
    ]


def test_normalize_model_family_aliases():
    assert mod.normalize_model_family("o3-2025-04-16") == "o3"
    assert mod.normalize_model_family("gpt-5-chat-latest") == "gpt-5"
    assert mod.normalize_model_family("chatgpt-4o-latest") == "gpt-4o"


def test_native_tool_wrappers_converted(monkeypatch):
    pipe = mod.Pipe()

    captured: dict[str, object] = {}

    async def fake_run_streaming_loop(
        body, valves, event_emitter, metadata, tools, *, openai_file_citations=None
    ):
        captured["body"] = body
        return "ok"

    monkeypatch.setattr(pipe, "_run_streaming_loop", fake_run_streaming_loop)

    async def emitter(_event):
        return None

    body = {
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "AskUserQuestion",
                    "description": "x",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "AskUserQuestion"}},
    }

    result = asyncio.run(
        pipe.pipe(
            body,
            __user__={"id": "u1"},
            __request__=None,  # unused by this code path
            __event_emitter__=emitter,
            __metadata__={"model": {"id": "openai_responses.gpt-5.2"}, "chat_id": None},
            __tools__=None,
        )
    )
    assert result == "ok"

    sent = captured["body"]
    assert isinstance(sent, mod.ResponsesBody)
    tool = next(t for t in (sent.tools or []) if t.get("type") == "function")
    assert tool["name"] == "AskUserQuestion"
    assert tool["strict"] is True
    assert tool["parameters"]["additionalProperties"] is False
    assert sent.tool_choice == {"type": "function", "name": "AskUserQuestion"}


@pytest.mark.parametrize("item_type", ["", "a", "bad!", "x" * 31])
def test_create_marker_rejects_bad_types(item_type):
    """Ensure invalid item_type values raise."""
    with pytest.raises(ValueError):
        mod.create_marker(item_type)


def test_marker_no_markers():
    text = "no markers"
    assert not mod.contains_marker(text)
    assert mod.extract_markers(text) == []
    assert mod.split_text_by_markers(text) == [{"type": "text", "text": text}]


def test_multiple_markers_and_parsing():
    m1 = mod.create_marker("fc", ulid="A" * 16)
    m2 = mod.create_marker("tool", ulid="B" * 16)
    txt = f"pre {mod.wrap_marker(m1)} mid {mod.wrap_marker(m2)} end"
    assert mod.extract_markers(txt) == [m1, m2]
    segs = mod.split_text_by_markers(txt)
    assert [s["type"] for s in segs] == ["text", "marker", "text", "marker", "text"]


def test_parse_marker_invalid_version():
    with pytest.raises(ValueError):
        mod.parse_marker("openai_responses:v1:bad")


def test_persist_missing_and_empty(dummy_chats):
    assert (
        mod.persist_openai_response_items(
            "x", "m", [{"type": "t"}], "model"
        )
        == ""
    )
    dummy_chats["c1"] = {"history": {"messages": {}}}
    assert mod.persist_openai_response_items("c1", "m", [], "model") == ""


def test_fetch_nonexistent(dummy_chats):
    dummy_chats["c1"] = {"history": {"messages": {}}}
    assert mod.fetch_openai_response_items("c1", ["bad"]) == {}


def test_duplicate_persistence(dummy_chats, monkeypatch):
    dummy_chats["c1"] = {"history": {"messages": {}}}
    monkeypatch.setattr(mod, "generate_item_id", lambda: "A" * 16)
    mod.persist_openai_response_items("c1", "m1", [{"type": "ab"}], "model")
    mod.persist_openai_response_items("c1", "m1", [{"type": "bb"}], "model")
    store = dummy_chats["c1"]["openai_responses_pipe"]["items"]
    assert list(store) == ["A" * 16]
    assert store["A" * 16]["payload"]["type"] == "bb"
    ids = dummy_chats["c1"]["openai_responses_pipe"]["messages_index"]["m1"][
        "item_ids"
    ]
    assert ids == ["A" * 16, "A" * 16]


def test_transform_messages_various(monkeypatch):
    monkeypatch.setattr(mod, "fetch_openai_response_items", lambda *a, **k: {})
    msgs = [
        {"role": "system", "content": "skip"},
        {"role": "user", "content": "hi"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "t"},
                {"type": "image_url", "image_url": {"url": "u"}},
                {"type": "unknown", "value": 1},
            ],
        },
        {"role": "developer", "content": "dev"},
        {"role": "assistant", "content": "ok"},
        {"content": "ignored"},
    ]
    out = mod.ResponsesBody.transform_messages_to_input(msgs)
    assert [o["role"] for o in out] == [
        "user",
        "user",
        "developer",
        "assistant",
        "assistant",
    ]
    assert out[1]["content"][1]["image_url"] == "u"
    assert out[1]["content"][2] == {"type": "unknown", "value": 1}
    # chat_id without model_id should still transform without raising
    out2 = mod.ResponsesBody.transform_messages_to_input(msgs, chat_id="c1")
    assert len(out2) == len(out)
    # model_id without chat_id should also transform without raising
    out3 = mod.ResponsesBody.transform_messages_to_input(
        msgs, openwebui_model_id="model"
    )
    assert len(out3) == len(out)


def test_transform_messages_missing_item(monkeypatch, dummy_chats):
    dummy_chats["c1"] = {"history": {"messages": {}}}
    marker = mod.wrap_marker(mod.create_marker("fc", ulid="B" * 16))
    monkeypatch.setattr(mod, "fetch_openai_response_items", lambda *a, **k: {})
    out = mod.ResponsesBody.transform_messages_to_input(
        [{"role": "assistant", "content": marker}],
        chat_id="c1",
        openwebui_model_id="model",
    )
    assert out == []


@pytest.mark.parametrize(
    "tools,expected",
    [
        (None, []),
        ([1, "x"], []),
        ({"bad": 1}, []),
    ],
)
def test_transform_tools_invalid(tools, expected):
    assert mod.ResponsesBody.transform_tools(tools) == expected


def test_transform_tools_dedup_and_unknown():
    tools = [
        {"spec": {"name": "add", "parameters": {"a": {"type": "number"}}}},
        {"type": "function", "function": {"name": "add", "parameters": {"b": 1}}},
        {"type": "foo"},
    ]
    out = mod.ResponsesBody.transform_tools(tools)
    names = {t.get("name", t.get("type")) for t in out}
    assert names == {"add", "foo"}
    func = next(t for t in out if t.get("type") == "function")
    assert func["parameters"].get("b") == 1


def test_collect_openwebui_file_attachments(monkeypatch, tmp_path):
    pipe = mod.Pipe()

    local_file = tmp_path / "doc.txt"
    local_file.write_text("hi", encoding="utf-8")

    dummy = mod.Files.DummyFile(  # type: ignore[attr-defined]
        id="f1",
        user_id="u1",
        filename="doc.txt",
        path=str(local_file),
        meta={"content_type": "text/plain", "size": local_file.stat().st_size},
    )

    monkeypatch.setattr(mod.Files, "get_file_by_id_and_user_id", lambda fid, uid: dummy)
    monkeypatch.setattr(mod.Storage, "get_file", lambda p: p)

    out = pipe._collect_openwebui_file_attachments([{"id": "f1", "name": "doc.txt"}], user_id="u1")
    assert len(out) == 1
    assert out[0].id == "f1"
    assert out[0].filename == "doc.txt"
    assert out[0].local_path == str(local_file)
    assert out[0].content_type == "text/plain"


def test_builtin_file_tools_injected(monkeypatch):
    pipe = mod.Pipe()
    pipe.valves = pipe.Valves(
        ALLOW_OPENAI_FILE_UPLOADS=True,
        ENABLE_CODE_INTERPRETER_TOOL=True,
        ENABLE_FILE_SEARCH_TOOL=True,
    )

    attachment = mod._OpenWebUIFileAttachment(
        id="f1",
        filename="doc.txt",
        local_path="/tmp/doc.txt",
        size_bytes=2,
        content_type="text/plain",
    )
    monkeypatch.setattr(pipe, "_collect_openwebui_file_attachments", lambda raw, user_id: [attachment])

    async def fake_uploads(*args, **kwargs):
        return (
            ["file-abc"],
            {"file-abc": {"openwebui_file_id": "f1", "filename": "doc.txt"}},
        )

    async def fake_vs(*args, **kwargs):
        return "vs-123"

    monkeypatch.setattr(pipe, "_ensure_openai_file_uploads", fake_uploads)
    monkeypatch.setattr(pipe, "_ensure_vector_store_indexed", fake_vs)

    captured: dict[str, object] = {}

    async def fake_run_streaming_loop(
        body, valves, event_emitter, metadata, tools, *, openai_file_citations=None
    ):
        captured["body"] = body
        captured["openai_file_citations"] = openai_file_citations
        return "ok"

    monkeypatch.setattr(pipe, "_run_streaming_loop", fake_run_streaming_loop)

    async def emitter(_event):
        return None

    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "f1"},
                    {"type": "text", "text": "analyze"},
                ],
            }
        ],
        "stream": True,
    }

    result = asyncio.run(
        pipe.pipe(
            body,
            __user__={"id": "u1"},
            __request__=None,  # unused by this code path
            __event_emitter__=emitter,
            __metadata__={"model": {"id": "openai_responses.gpt-4o"}, "chat_id": None},
            __tools__=None,
            __files__=[{"id": "f1", "name": "doc.txt"}],
        )
    )
    assert result == "ok"

    sent = captured["body"]
    assert isinstance(sent, mod.ResponsesBody)
    code_interpreter_tool = next(t for t in (sent.tools or []) if t.get("type") == "code_interpreter")
    assert code_interpreter_tool["container"]["type"] == "auto"
    assert code_interpreter_tool["container"]["file_ids"] == ["file-abc"]

    file_search_tool = next(t for t in (sent.tools or []) if t.get("type") == "file_search")
    assert file_search_tool["vector_store_ids"] == ["vs-123"]

    # input_file rewritten from Open WebUI id -> OpenAI file id
    assert sent.input[0]["content"][0]["file_id"] == "file-abc"

    # input_file rewritten from Open WebUI id -> OpenAI file id
    assert sent.input[0]["content"][0]["file_id"] == "file-abc"


def test_builtin_file_tools_use_stashed_files(monkeypatch):
    pipe = mod.Pipe()
    pipe.valves = pipe.Valves(
        ALLOW_OPENAI_FILE_UPLOADS=True,
        ENABLE_CODE_INTERPRETER_TOOL=True,
        ENABLE_FILE_SEARCH_TOOL=True,
    )

    attachment = mod._OpenWebUIFileAttachment(
        id="f1",
        filename="doc.txt",
        local_path="/tmp/doc.txt",
        size_bytes=2,
        content_type="text/plain",
    )

    def fake_collect(raw, user_id):
        assert raw == [{"id": "f1", "name": "doc.txt"}]
        assert user_id == "u1"
        return [attachment]

    monkeypatch.setattr(pipe, "_collect_openwebui_file_attachments", fake_collect)

    async def fake_uploads(*args, **kwargs):
        return (
            ["file-abc"],
            {"file-abc": {"openwebui_file_id": "f1", "filename": "doc.txt"}},
        )

    async def fake_vs(*args, **kwargs):
        return "vs-123"

    monkeypatch.setattr(pipe, "_ensure_openai_file_uploads", fake_uploads)
    monkeypatch.setattr(pipe, "_ensure_vector_store_indexed", fake_vs)

    captured: dict[str, object] = {}

    async def fake_run_streaming_loop(
        body, valves, event_emitter, metadata, tools, *, openai_file_citations=None
    ):
        captured["body"] = body
        return "ok"

    monkeypatch.setattr(pipe, "_run_streaming_loop", fake_run_streaming_loop)

    async def emitter(_event):
        return None

    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "f1"},
                    {"type": "text", "text": "analyze"},
                ],
            }
        ],
        "stream": True,
    }

    result = asyncio.run(
        pipe.pipe(
            body,
            __user__={"id": "u1"},
            __request__=None,  # unused by this code path
            __event_emitter__=emitter,
            __metadata__={
                "model": {"id": "openai_responses.gpt-4o"},
                "chat_id": None,
                "features": {"openai_responses": {"files": [{"id": "f1", "name": "doc.txt"}]}},
            },
            __tools__=None,
            __files__=None,
        )
    )
    assert result == "ok"

    sent = captured["body"]
    assert isinstance(sent, mod.ResponsesBody)
    code_interpreter_tool = next(t for t in (sent.tools or []) if t.get("type") == "code_interpreter")
    assert code_interpreter_tool["container"]["type"] == "auto"
    assert code_interpreter_tool["container"]["file_ids"] == ["file-abc"]

    file_search_tool = next(t for t in (sent.tools or []) if t.get("type") == "file_search")
    assert file_search_tool["vector_store_ids"] == ["vs-123"]


def test_reasoning_effort_user_valve_applied(monkeypatch):
    pipe = mod.Pipe()

    captured: dict[str, object] = {}

    async def fake_run_streaming_loop(
        body, valves, event_emitter, metadata, tools, *, openai_file_citations=None
    ):
        captured["body"] = body
        return "ok"

    monkeypatch.setattr(pipe, "_run_streaming_loop", fake_run_streaming_loop)

    async def emitter(_event):
        return None

    body = {"model": "o3", "messages": [{"role": "user", "content": "hi"}], "stream": True}

    result = asyncio.run(
        pipe.pipe(
            body,
            __user__={"id": "u1", "valves": {"REASONING_EFFORT": "xhigh"}},
            __request__=None,  # unused by this code path
            __event_emitter__=emitter,
            __metadata__={"model": {"id": "openai_responses.o3"}, "chat_id": None},
            __tools__=None,
        )
    )
    assert result == "ok"

    sent = captured["body"]
    assert isinstance(sent, mod.ResponsesBody)
    assert sent.reasoning["effort"] == "xhigh"


def test_reasoning_effort_defaults_to_medium(monkeypatch):
    pipe = mod.Pipe()

    captured: dict[str, object] = {}

    async def fake_run_streaming_loop(
        body, valves, event_emitter, metadata, tools, *, openai_file_citations=None
    ):
        captured["body"] = body
        return "ok"

    monkeypatch.setattr(pipe, "_run_streaming_loop", fake_run_streaming_loop)

    async def emitter(_event):
        return None

    body = {"model": "o3", "messages": [{"role": "user", "content": "hi"}], "stream": True}

    result = asyncio.run(
        pipe.pipe(
            body,
            __user__={"id": "u1"},
            __request__=None,  # unused by this code path
            __event_emitter__=emitter,
            __metadata__={"model": {"id": "openai_responses.o3"}, "chat_id": None},
            __tools__=None,
        )
    )
    assert result == "ok"

    sent = captured["body"]
    assert isinstance(sent, mod.ResponsesBody)
    assert sent.reasoning["effort"] == "medium"


def test_reasoning_effort_user_valve_legacy_inherit_coerces_to_medium():
    user_valves = mod.Pipe.UserValves.model_validate({"REASONING_EFFORT": "INHERIT"})
    assert user_valves.REASONING_EFFORT == "medium"


def test_user_valves_ignores_legacy_log_level_key():
    user_valves = mod.Pipe.UserValves.model_validate({"LOG_LEVEL": "DEBUG"})
    assert user_valves.REASONING_EFFORT == "medium"


@pytest.mark.parametrize(
    "payload",
    ["", "{", json.dumps([1, {}]), json.dumps({"server_label": "x"})],
)
def test_build_mcp_tools_invalid(payload):
    assert mod.ResponsesBody._build_mcp_tools(payload) == []


def test_responses_request_retries_without_tool_resources(monkeypatch):
    pipe = mod.Pipe()

    class DummyResponse:
        def __init__(self, status: int, payload: object):
            self.status = status
            self._payload = payload

        async def json(self):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class DummySession:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = []

        def post(self, url, json=None, headers=None):
            self.calls.append({"url": url, "json": json, "headers": headers})
            return self._responses.pop(0)

    session = DummySession(
        [
            DummyResponse(
                400,
                {
                    "error": {
                        "message": "Unknown parameter: 'tool_resources'.",
                        "type": "invalid_request_error",
                        "param": "tool_resources",
                        "code": "unknown_parameter",
                    }
                },
            ),
            DummyResponse(200, {"id": "resp-ok"}),
        ]
    )

    async def fake_get_or_init_http_session():
        return session

    monkeypatch.setattr(pipe, "_get_or_init_http_session", fake_get_or_init_http_session)

    request = {
        "model": "gpt-4o",
        "input": "hi",
        "tools": [{"type": "code_interpreter"}],
        "tool_resources": {"code_interpreter": {"file_ids": ["file-1"]}},
    }

    out = asyncio.run(
        pipe.send_openai_responses_nonstreaming_request(
            request,
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
    )
    assert out == {"id": "resp-ok"}
    assert len(session.calls) == 2
    assert session.calls[0]["json"]["tool_resources"]["code_interpreter"]["file_ids"] == ["file-1"]
    assert session.calls[0]["headers"]["OpenAI-Beta"] == "assistants=v2"

    # Retry removes top-level tool_resources and folds known resources into the tool object.
    assert "tool_resources" not in session.calls[1]["json"]
    assert session.calls[1]["json"]["tools"][0]["container"]["file_ids"] == ["file-1"]
    assert session.calls[1]["headers"]["OpenAI-Beta"] == "assistants=v2"


def test_responses_request_retries_with_missing_code_interpreter_container(monkeypatch):
    pipe = mod.Pipe()

    class DummyResponse:
        def __init__(self, status: int, payload: object):
            self.status = status
            self._payload = payload

        async def json(self):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class DummySession:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = []

        def post(self, url, json=None, headers=None):
            self.calls.append({"url": url, "json": json, "headers": headers})
            return self._responses.pop(0)

    session = DummySession(
        [
            DummyResponse(
                400,
                {
                    "error": {
                        "message": "Missing required parameter: 'tools[0].container'.",
                        "type": "invalid_request_error",
                        "param": "tools[0].container",
                        "code": "missing_required_parameter",
                    }
                },
            ),
            DummyResponse(200, {"id": "resp-ok"}),
        ]
    )

    async def fake_get_or_init_http_session():
        return session

    monkeypatch.setattr(pipe, "_get_or_init_http_session", fake_get_or_init_http_session)

    request = {
        "model": "gpt-4o",
        "input": "hi",
        "tools": [{"type": "code_interpreter"}],
    }

    out = asyncio.run(
        pipe.send_openai_responses_nonstreaming_request(
            request,
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
    )
    assert out == {"id": "resp-ok"}
    assert len(session.calls) == 2
    assert "container" not in session.calls[0]["json"]["tools"][0]
    assert session.calls[0]["headers"]["OpenAI-Beta"] == "assistants=v2"

    # Retry injects a default container value into the code interpreter tool.
    assert session.calls[1]["json"]["tools"][0]["container"] == {"type": "auto"}
    assert session.calls[1]["headers"]["OpenAI-Beta"] == "assistants=v2"
