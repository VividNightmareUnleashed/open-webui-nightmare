from functions.filters.openai_responses_manifold_companion import (
    openai_responses_manifold_companion as mod,
)


def test_companion_passthrough_for_non_matching_model():
    flt = mod.Filter()
    body = {
        "model": "gpt-4o",
        "features": {"web_search": True, "code_interpreter": True},
        "files": [{"id": "f1", "name": "doc.txt"}],
    }
    metadata: dict[str, object] = {}

    out = flt.inlet(body, __metadata__=metadata)
    assert out["features"]["web_search"] is True
    assert out["features"]["code_interpreter"] is True
    assert out["files"] == [{"id": "f1", "name": "doc.txt"}]
    assert metadata == {}


def test_companion_intercepts_toggles_and_stashes_files():
    flt = mod.Filter()
    body = {
        "model": "openai_responses.gpt-4o",
        "features": {"web_search": True, "code_interpreter": True},
        "files": [{"id": "f1", "name": "doc.txt"}],
    }
    metadata: dict[str, object] = {}

    out = flt.inlet(body, __metadata__=metadata)
    assert out["features"]["web_search"] is False
    assert out["features"]["code_interpreter"] is False
    assert out["files"] == []

    openai_features = metadata["features"]["openai_responses"]
    assert openai_features["web_search"] is True
    assert openai_features["code_interpreter"] is True
    assert openai_features["files"] == [{"id": "f1", "name": "doc.txt"}]
    assert openai_features["bypass_backend_rag"] is True

