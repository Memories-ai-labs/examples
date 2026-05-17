"""Unit tests for the MemoriesClient HTTP wrapper."""

from __future__ import annotations

import pytest

from common import MemoriesClient
from common.client import MemoriesError
from tests.conftest import envelope, fake_response, make_session


def _client(routes):
    session = make_session(routes)
    return MemoriesClient(api_key="sk-mavi-test", session=session), session


def test_search_unwraps_data_list():
    payload = envelope([
        {"videoNo": "VI1", "startTime": "10", "endTime": "15", "score": 0.51, "audio_ts": ""}
    ])
    client, session = _client({("POST", "/search"): payload})

    hits = client.search("a person walking", unique_id="u", top_k=5)
    assert hits[0]["videoNo"] == "VI1"
    # Verify the wire body
    body = session.post.call_args.kwargs["json"]
    assert body["search_param"] == "a person walking"
    assert body["search_type"] == "BY_CLIP"
    assert body["unique_id"] == "u"
    assert body["top_k"] == 5
    assert body["filtering_level"] == "medium"


def test_search_audio_search_type():
    payload = envelope([])
    client, session = _client({("POST", "/search"): payload})
    client.search("hello", search_type="BY_AUDIO")
    assert session.post.call_args.kwargs["json"]["search_type"] == "BY_AUDIO"


def test_get_metadata_returns_status():
    payload = envelope({"videoNo": "VI1", "status": "PARSE"})
    client, _ = _client({("GET", "/get_metadata"): payload})
    meta = client.get_metadata("VI1")
    assert meta["status"] == "PARSE"


def test_business_error_raises():
    client, _ = _client({("POST", "/search"): {"code": "1024", "msg": "bad query", "data": None}})
    with pytest.raises(MemoriesError) as excinfo:
        client.search("x")
    assert excinfo.value.code == "1024"


def test_http_error_raises():
    routes = {("POST", "/search"): lambda m, u, **k: fake_response({"error": "boom"}, status_code=500)}
    client, _ = _client(routes)
    with pytest.raises(MemoriesError):
        client.search("x")


def test_vlm_complete_parses_gemini_text_envelope():
    body = {
        "status": "completed",
        "choices": [{"text": '{"ok": true}', "index": 0}],
    }
    routes = {("POST", "/vu/chat/completions"): body}
    client, session = _client(routes)
    out = client.vlm_complete("describe", video_url="https://cdn/x.mp4")
    assert out == '{"ok": true}'
    sent = session.post.call_args.kwargs["json"]
    assert sent["model"].startswith("gemini:")
    assert sent["messages"][-1]["content"][0]["type"] == "text"
    assert sent["messages"][-1]["content"][1]["type"] == "input_file"


def test_vlm_complete_strips_markdown_code_fence():
    body = {"status": "completed", "choices": [{"text": '```json\n{"ok": true}\n```', "index": 0}]}
    client, _ = _client({("POST", "/vu/chat/completions"): body})
    out = client.vlm_complete("x")
    assert out == '{"ok": true}'


def test_vlm_complete_falls_back_to_message_content_shape():
    body = {"status": "completed", "choices": [{"message": {"content": "hello"}}]}
    client, _ = _client({("POST", "/vu/chat/completions"): body})
    assert client.vlm_complete("x") == "hello"


def test_vlm_complete_raises_on_status_errored():
    body = {
        "status": "errored",
        "choices": [],
        "error": {"code": "INVALID_ARGUMENT", "message": "boom"},
    }
    client, _ = _client({("POST", "/vu/chat/completions"): body})
    with pytest.raises(MemoriesError) as excinfo:
        client.vlm_complete("x")
    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_vlm_complete_with_system_message():
    body = {"status": "completed", "choices": [{"text": "ok"}]}
    client, session = _client({("POST", "/vu/chat/completions"): body})
    client.vlm_complete("x", system="you are strict")
    msgs = session.post.call_args.kwargs["json"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "you are strict"
    # User content must be an array (bare string is rejected by the API)
    assert isinstance(msgs[1]["content"], list)


def test_search_video_nos_filter_included():
    client, session = _client({("POST", "/search"): envelope([])})
    client.search("x", video_nos=["VI1", "VI2"])
    assert session.post.call_args.kwargs["json"]["video_nos"] == ["VI1", "VI2"]


def test_search_retries_on_transient_0001_then_succeeds(monkeypatch):
    """code=0001 ('network abnormal') is transient — the client retries it."""
    monkeypatch.setattr("common.client.time.sleep", lambda _: None)
    state = {"calls": 0}

    def search_responder(method, url, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return fake_response({"code": "0001", "msg": "network abnormal", "data": None})
        return fake_response(envelope([{"videoNo": "VI1", "startTime": "0", "endTime": "1", "score": 0.5}]))

    client, _ = _client({("POST", "/search"): search_responder})
    client.max_retries = 3
    hits = client.search("hello")
    assert state["calls"] == 2
    assert hits[0]["videoNo"] == "VI1"


def test_search_retry_eventually_gives_up(monkeypatch):
    monkeypatch.setattr("common.client.time.sleep", lambda _: None)
    state = {"calls": 0}

    def search_responder(method, url, **kwargs):
        state["calls"] += 1
        return fake_response({"code": "0001", "msg": "network abnormal", "data": None})

    client, _ = _client({("POST", "/search"): search_responder})
    client.max_retries = 2
    with pytest.raises(MemoriesError) as excinfo:
        client.search("hello")
    assert excinfo.value.code == "0001"
    assert state["calls"] == 2


def test_search_audio_transcripts_get_query_params():
    payload = envelope({"current_page": 1, "page_size": 100, "total_count": 0, "videos": []})
    client, session = _client({("GET", "/search_audio_transcripts"): payload})
    client.search_audio_transcripts("hello", page_size=10, unique_id="u")
    params = session.get.call_args.kwargs["params"]
    assert params["query"] == "hello"
    assert params["page_size"] == 10
    assert params["unique_id"] == "u"
