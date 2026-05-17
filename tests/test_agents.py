"""End-to-end-with-mocks tests for each agent.

These exercise the full ReAct control flow but route HTTP through a fake session
so they're hermetic and fast.
"""

from __future__ import annotations

import json

from common import MemoriesClient
from tests.conftest import envelope, fake_response, make_session


def _vlm_body(payload: dict) -> dict:
    return {"status": "completed", "choices": [{"text": json.dumps(payload), "index": 0}]}


def _client_with_routes(routes, *, with_media_bridge: bool = True):
    """Tests that exercise the VLM step need a media-URL bridge configured —
    pass with_media_bridge=False to assert the retrieval-only fallback path.
    """
    media_url_map = (
        {"VI1": "https://cdn/x.mp4", "VF": "https://cdn/x.mp4", "VL": "https://cdn/x.mp4",
         "VA": "https://cdn/x.mp4", "VR": "https://cdn/x.mp4"}
        if with_media_bridge
        else None
    )
    return MemoriesClient(
        api_key="sk-mavi-test",
        session=make_session(routes),
        media_url_map=media_url_map,
    )


def test_qsr_drivethru_runs_end_to_end():
    from agents.qsr_drivethru_sop import run

    routes = {
        ("POST", "/search"): envelope([
            {"videoNo": "VI1", "startTime": "10", "endTime": "20", "score": 0.61, "audio_ts": ""},
            {"videoNo": "VI1", "startTime": "55", "endTime": "65", "score": 0.55, "audio_ts": ""},
        ]),
        ("POST", "/download"): envelope({"download_url": "https://cdn/x.mp4"}),  # unused now
        ("POST", "/vu/chat/completions"): _vlm_body({
            "physical_interactions": 2,
            "bag_handed": True,
            "drinks_included": True,
            "greeting_observed": True,
            "notes": "ok",
        }),
    }
    client = _client_with_routes(routes)

    summary = run(
        client,
        video_no="VI1",
        upload_path=None,
        unique_id="u",
        top_k=5,
        filtering_level="medium",
        query="handoff",
        verbose=False,
    )
    assert summary["candidates"] == 2
    assert summary["fully_compliant"] == 2
    assert summary["events"][0]["result"]["bag_handed"] is True


def test_restaurant_quality_builds_timeline():
    from agents.restaurant_service_quality import run, DEFAULT_QUERIES

    counter = {"i": 0}

    def search_responder(method, url, **kwargs):
        counter["i"] += 1
        # one fake hit per query at a unique timestamp
        t = 100 * counter["i"]
        return fake_response(envelope([
            {"videoNo": "VF", "startTime": str(t), "endTime": str(t + 5), "score": 0.5},
            {"videoNo": "VF", "startTime": str(t + 1), "endTime": str(t + 6), "score": 0.49},
        ]))

    routes = {("POST", "/search"): search_responder}
    client = _client_with_routes(routes)

    summary = run(
        client,
        video_no="VF",
        upload_path=None,
        unique_id="u",
        top_k=10,
        filtering_level="medium",
        queries=DEFAULT_QUERIES,
        verbose=False,
    )
    # Each query contributes one event after dedupe-within-5s.
    assert summary["event_count"] == len(DEFAULT_QUERIES)
    assert summary["metrics"]["food_delivery_count"] == 1
    assert summary["metrics"]["table_touches"] == 1


def test_luci_personal_memory_composes_answer():
    from agents.luci_personal_memory import run

    routes = {
        ("POST", "/search"): envelope([
            {"videoNo": "VL", "startTime": "30", "endTime": "60", "score": 0.71, "audio_ts": "I'll have salmon"},
        ]),
        ("GET", "/search_audio_transcripts"): envelope({
            "current_page": 1, "page_size": 20, "total_count": 1,
            "videos": [{"videoNo": "VL", "startTime": "30", "audio_ts": "I'll have the salmon"}],
        }),
        ("POST", "/download"): envelope({"download_url": "https://cdn/x.mp4"}),  # unused now
        ("POST", "/vu/chat/completions"): _vlm_body({
            "venue": "Joe's Crab Shack",
            "activity": "ordering lunch",
            "order": "grilled salmon and fries",
        }),
    }
    client = _client_with_routes(routes)
    summary = run(
        client,
        question="What did I order for lunch last Tuesday?",
        date_after="2026-05-05 11:00:00",
        unique_id="u",
        top_k=5,
        filtering_level="medium",
        verbose=False,
    )
    assert "Joe's Crab Shack" in summary["answer"]
    assert "salmon" in summary["answer"]
    assert summary["transcript_hint"] == "I'll have the salmon"


def test_automotive_sop_audits_each_arrival():
    from agents.automotive_sop import run

    routes = {
        ("POST", "/search"): envelope([
            {"videoNo": "VA", "startTime": "0",    "endTime": "5",   "score": 0.6},
            {"videoNo": "VA", "startTime": "900",  "endTime": "905", "score": 0.55},
        ]),
        ("POST", "/download"): envelope({"download_url": "https://cdn/x.mp4"}),  # unused now
        ("POST", "/vu/chat/completions"): _vlm_body({
            "technician_greeted": True,
            "time_to_greet_sec": 30,
            "air_filter_checked": False,
            "tire_pressure_checked": True,
            "notes": "ok",
        }),
    }
    client = _client_with_routes(routes)
    summary = run(
        client,
        video_no="VA",
        upload_path=None,
        unique_id="u",
        top_k=10,
        filtering_level="medium",
        window_sec=60.0,
        greet_target_sec=60.0,
        verbose=False,
    )
    assert summary["arrival_count"] == 2
    assert summary["greeting_within_target"] == 2


def test_visual_rag_two_channel_then_verify():
    from agents.visual_rag import run

    visual_payload = envelope([
        {"videoNo": "VR", "startTime": "10", "endTime": "20", "score": 0.55},
        {"videoNo": "VR", "startTime": "60", "endTime": "70", "score": 0.45},
    ])
    audio_payload = envelope([
        {"videoNo": "VR", "startTime": "11", "endTime": "21", "score": 0.4, "audio_ts": "equation"},
    ])

    calls = {"search": 0, "vlm": 0}

    def search_responder(method, url, **kwargs):
        calls["search"] += 1
        body = kwargs.get("json", {})
        if body.get("search_type") == "BY_AUDIO":
            return fake_response(audio_payload)
        return fake_response(visual_payload)

    def vlm_responder(method, url, **kwargs):
        calls["vlm"] += 1
        body = _vlm_body({"has_match": True, "extract": "E=mc^2", "confidence": "high"})
        return fake_response(body)

    routes = {
        ("POST", "/search"): search_responder,
        ("POST", "/download"): envelope({"download_url": "https://cdn/x.mp4"}),  # unused now
        ("POST", "/vu/chat/completions"): vlm_responder,
    }
    client = _client_with_routes(routes)
    summary = run(
        client,
        query="professor writes equation",
        visual_query=None,
        audio_query=None,
        extract_hint=None,
        unique_id="u",
        video_nos=["VR"],
        top_k=10,
        filtering_level="low",
        max_verify=5,
        verbose=False,
    )
    assert calls["search"] == 2  # one visual, one audio
    # The first visual hit (10-20) and audio hit (11-21) should have merged.
    assert summary["candidates_total"] == 2
    assert summary["candidates_verified"] == 2
    assert len(summary["matches"]) == 2
    assert summary["matches"][0]["verdict"]["extract"] == "E=mc^2"


def test_qsr_drivethru_retrieval_only_when_no_media_bridge():
    """When MEMORIES_MEDIA_URL_TEMPLATE is unset and no map is given, the
    agent should retrieve candidates but skip VLM verification gracefully."""
    from agents.qsr_drivethru_sop import run

    routes = {
        ("POST", "/search"): envelope([
            {"videoNo": "VI1", "startTime": "10", "endTime": "20", "score": 0.61},
        ]),
    }
    client = _client_with_routes(routes, with_media_bridge=False)
    summary = run(
        client,
        video_no="VI1",
        upload_path=None,
        unique_id="u",
        top_k=5,
        filtering_level="medium",
        query="handoff",
        verbose=False,
    )
    assert summary["summary"] == "retrieval_only"
    assert len(summary["candidates"]) == 1


def test_visual_rag_retrieval_only_when_no_media_bridge():
    from agents.visual_rag import run

    def search_responder(method, url, **kwargs):
        body = kwargs.get("json", {})
        if body.get("search_type") == "BY_AUDIO":
            return fake_response(envelope([
                {"videoNo": "VR", "startTime": "11", "endTime": "21", "score": 0.4},
            ]))
        return fake_response(envelope([
            {"videoNo": "VR", "startTime": "10", "endTime": "20", "score": 0.55},
        ]))

    routes = {("POST", "/search"): search_responder}
    client = _client_with_routes(routes, with_media_bridge=False)
    summary = run(
        client,
        query="test",
        visual_query=None,
        audio_query=None,
        extract_hint=None,
        unique_id="u",
        video_nos=["VR"],
        top_k=10,
        filtering_level="low",
        max_verify=5,
        verbose=False,
    )
    assert summary.get("summary") == "retrieval_only"
    assert summary["matches"] == []


def test_client_resolve_media_url_uses_map():
    from common import MemoriesClient
    c = MemoriesClient(api_key="sk-mavi-test", media_url_map={"VI1": "https://m/x.mp4"})
    assert c.resolve_media_url("VI1") == "https://m/x.mp4"


def test_client_resolve_media_url_uses_env_template(monkeypatch):
    from common import MemoriesClient
    monkeypatch.setenv("MEMORIES_MEDIA_URL_TEMPLATE", "https://cdn/{video_no}.mp4")
    c = MemoriesClient(api_key="sk-mavi-test")
    assert c.resolve_media_url("VI42") == "https://cdn/VI42.mp4"


def test_client_resolve_media_url_raises_without_bridge(monkeypatch):
    from common import MemoriesClient
    from common.client import MemoriesError
    import pytest

    monkeypatch.delenv("MEMORIES_MEDIA_URL_TEMPLATE", raising=False)
    c = MemoriesClient(api_key="sk-mavi-test")
    with pytest.raises(MemoriesError):
        c.resolve_media_url("VI1")
