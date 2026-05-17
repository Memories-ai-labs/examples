"""Tests for the four PRD-aligned agents added in v2:

  Agent 3 — Security & Threat Detection
  Agent 4 — Video Searching Agent (SSE)
  Agent 5 — Video Editing Agent (VEA)
  Agent 8 — Creator Intelligence

Plus the underlying client surface they depend on: queries_stream(),
video_clip/video_split/video_edit, and the SSE parser.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from common import MemoriesClient
from common.client import MemoriesError, _iter_sse
from tests.conftest import envelope, fake_response, make_session


def _vlm_body(payload: dict) -> dict:
    return {"status": "completed", "choices": [{"text": json.dumps(payload), "index": 0}]}


def _client(routes, *, with_media_bridge: bool = True):
    return MemoriesClient(
        api_key="sk-mavi-test",
        session=make_session(routes),
        media_url_map=({"VI1": "https://cdn/x.mp4"} if with_media_bridge else None),
    )


# --------------------- Security & Threat Detection ------------------------

def test_security_threat_runs_against_real_search_shape():
    from agents.security_threat import run, DEFAULT_SCENARIOS

    # Two hits per scenario search (we'll feed it 2 scenarios)
    routes = {
        ("POST", "/search"): envelope([
            {"videoNo": "VI1", "startTime": "10", "endTime": "15", "score": 0.5},
            {"videoNo": "VI1", "startTime": "60", "endTime": "65", "score": 0.4},
        ]),
        ("POST", "/vu/chat/completions"): _vlm_body({
            "violation": True,
            "confidence": "high",
            "notes": "clearly concealed",
        }),
    }
    client = _client(routes)
    summary = run(
        client,
        video_nos=["VI1"],
        unique_id="u",
        scenarios=DEFAULT_SCENARIOS[:2],
        top_k_per_scenario=2,
        filtering_level="low",
        verbose=False,
    )
    # 2 scenarios * 2 hits = 4 candidates, all verified
    assert summary["candidates_total"] == 4
    assert summary["confirmed_total"] == 4
    # Severity counts add up to confirmed_total
    assert sum(summary["severity_counts"].values()) == 4


def test_security_threat_retrieval_only_no_bridge():
    from agents.security_threat import run, DEFAULT_SCENARIOS

    routes = {
        ("POST", "/search"): envelope([
            {"videoNo": "VI1", "startTime": "10", "endTime": "15", "score": 0.5},
        ]),
    }
    client = _client(routes, with_media_bridge=False)
    summary = run(
        client,
        video_nos=["VI1"],
        unique_id="u",
        scenarios=DEFAULT_SCENARIOS[:1],
        top_k_per_scenario=2,
        filtering_level="low",
        verbose=False,
    )
    assert summary["summary"] == "retrieval_only"
    assert summary["skipped_no_bridge"] >= 1


def test_security_threat_no_candidates():
    from agents.security_threat import run, DEFAULT_SCENARIOS

    routes = {("POST", "/search"): envelope([])}
    client = _client(routes)
    summary = run(
        client,
        video_nos=["VI1"],
        unique_id="u",
        scenarios=DEFAULT_SCENARIOS[:1],
        top_k_per_scenario=2,
        filtering_level="low",
        verbose=False,
    )
    assert summary["summary"] == "no_candidates"


# --------------------- SSE parser ------------------------------------------

def test_iter_sse_parses_named_events_and_json():
    resp = MagicMock()
    resp.iter_lines.return_value = iter([
        "event: started",
        'data: {"session_id":"abc"}',
        "",
        "event: progress",
        'data: {"step":1,"message":"go"}',
        "",
        "event: complete",
        'data: {"video_references":[]}',
        "",
    ])
    events = list(_iter_sse(resp))
    assert [e[0] for e in events] == ["started", "progress", "complete"]
    assert events[0][1] == {"session_id": "abc"}
    assert events[2][1]["video_references"] == []


def test_iter_sse_skips_comments_and_handles_no_trailing_blank():
    resp = MagicMock()
    resp.iter_lines.return_value = iter([
        ": comment line",
        "event: complete",
        'data: {"ok":true}',
    ])
    events = list(_iter_sse(resp))
    assert events == [("complete", {"ok": True})]


# --------------------- Video Searching Agent (SSE) -------------------------

class _SSEResponse:
    """Context-managed fake response that yields SSE lines from iter_lines()."""
    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self.url = "https://fake/queries/stream"
        self.text = ""
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def json(self):
        return {}


def _sse_session(lines, *, status_code: int = 200) -> MagicMock:
    session = MagicMock()
    session.post.side_effect = lambda url, **kw: _SSEResponse(lines, status_code=status_code)
    return session


def test_video_searching_agent_runs_sse_and_returns_complete():
    from agents.video_searching import run

    lines = [
        "event: started", 'data: {"session_id":"S1","query":"x"}', "",
        "event: progress", 'data: {"step":1,"max_steps":3,"message":"searching tiktok"}', "",
        "event: tool_call", 'data: {"tool":"tiktok_search","arguments":{"q":"x"}}', "",
        "event: tool_result", 'data: {"tool":"tiktok_search","success":true,"summary":"5 hits"}', "",
        "event: complete",
        'data: {"session_id":"S1","query":"x","answer":"OK","video_references":[{"video_id":"v1","platform":"tiktok"}],"confidence_score":0.9}',
        "",
    ]
    client = MemoriesClient(api_key="sk-mavi-test", session=_sse_session(lines))
    final = run(
        client,
        query="x",
        platforms=["tiktok"],
        max_results=5,
        time_frame="past_week",
        verbose=False,
    )
    assert final["video_references"][0]["video_id"] == "v1"
    assert final["confidence_score"] == 0.9


def test_queries_stream_raises_when_no_complete_event():
    lines = [
        "event: started", 'data: {"session_id":"S"}', "",
        "event: error", 'data: {"message":"upstream fail"}', "",
    ]
    client = MemoriesClient(api_key="sk-mavi-test", session=_sse_session(lines))
    with pytest.raises(MemoriesError) as excinfo:
        client.queries_stream("x")
    assert "upstream fail" in str(excinfo.value)


# --------------------- Video Editing Agent (VEA) ---------------------------

def test_video_editing_agent_kicks_off_async_tasks():
    from agents.video_editing import run

    # The /video/edit and /video/clip envelopes use integer code: 200.
    routes = {
        ("POST", "/video/clip"): {"code": 200, "msg": "ok", "data": {"task_id": "t_clip_1"}},
        ("POST", "/video/edit"): {"code": 200, "msg": "ok", "data": {"task_id": "t_edit_1"}},
    }
    client = _client(routes)
    summary = run(
        client,
        asset_ids=["re_1", "re_2"],
        user_prompt="highlight reel",
        orientation="landscape",
        do_scene_detection=True,
        verbose=False,
    )
    assert summary["edit_task"]["task_id"] == "t_edit_1"
    # Two source assets -> two scene-detection tasks
    assert len(summary["scene_tasks"]) == 2
    assert all(t["task_id"] == "t_clip_1" for t in summary["scene_tasks"])


def test_video_edit_raises_on_business_error_with_int_code():
    routes = {("POST", "/video/edit"): {"code": 400, "msg": "bad asset", "data": None}}
    client = _client(routes)
    with pytest.raises(MemoriesError) as excinfo:
        client.video_edit(["re_x"], "edit me")
    assert excinfo.value.code == "400"


def test_video_split_passes_segment_seconds():
    routes = {("POST", "/video/split"): {"code": 200, "msg": "ok", "data": {"task_id": "t_split_1"}}}
    client = _client(routes)
    r = client.video_split("re_x", segment_seconds=30)
    assert r["task_id"] == "t_split_1"


# --------------------- Creator Intelligence --------------------------------

def test_creator_intelligence_aggregates_dimensions():
    from agents.creator_intelligence import run

    # Two videos with different scores so the aggregator has something to mean over
    scores_per_call = iter([
        _vlm_body({"production_quality": 80, "audio_quality": 70, "delivery": 75,
                   "hook_strength": 60, "brand_safety": 95, "content_style": "tutorial",
                   "red_flags": [], "notes": "clean"}),
        _vlm_body({"production_quality": 60, "audio_quality": 50, "delivery": 55,
                   "hook_strength": 70, "brand_safety": 90, "content_style": "vlog",
                   "red_flags": ["one expletive"], "notes": "ok"}),
    ])
    routes = {
        ("POST", "/vu/chat/completions"): lambda m, u, **k: fake_response(next(scores_per_call)),
    }
    client = _client(routes)
    summary = run(
        client,
        creator_name="@alex",
        video_urls=["https://cdn/v1.mp4", "https://cdn/v2.mp4"],
        verbose=False,
    )
    assert summary["video_count"] == 2
    assert summary["per_dimension"]["production_quality"] == 70.0
    assert summary["per_dimension"]["brand_safety"] == 92.5
    assert "one expletive" in summary["red_flags"]
    assert summary["recommendation"] in (
        "Strong Fit", "Moderate Fit", "Not Recommended", "Not Recommended (brand-safety risk)",
    )


def test_creator_intelligence_flags_brand_safety_risk():
    from agents.creator_intelligence import _recommendation

    assert _recommendation({"overall_score": 90, "per_dimension": {"brand_safety": 40}}).startswith(
        "Not Recommended"
    )
    assert _recommendation({"overall_score": 80, "per_dimension": {"brand_safety": 95}}) == "Strong Fit"
    assert _recommendation({"overall_score": 65, "per_dimension": {"brand_safety": 95}}) == "Moderate Fit"
    assert _recommendation({"overall_score": 40, "per_dimension": {"brand_safety": 95}}) == "Not Recommended"
