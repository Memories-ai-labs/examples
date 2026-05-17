"""Pure-logic tests for helpers (no HTTP)."""

from __future__ import annotations

import pytest

from common import ReActAgent
from common.poll import wait_for_parse
from common.client import MemoriesError


def test_react_trace_records_each_step():
    a = ReActAgent("X", verbose=False)
    a.thought("plan")
    a.action("call")
    a.observe("got 3", payload=[1, 2, 3])
    a.answer("done")
    assert [s.kind for s in a.trace] == ["THOUGHT", "ACTION", "OBSERVE", "ANSWER"]


def test_react_markdown_dump_includes_steps():
    a = ReActAgent("X", verbose=False)
    a.thought("plan")
    a.action("call")
    md = a.to_markdown()
    assert "# X" in md
    assert "THOUGHT" in md and "ACTION" in md


def test_wait_for_parse_completes(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = 0
        def get_metadata(self, video_no):
            self.calls += 1
            return {"status": "UNPARSE"} if self.calls < 3 else {"status": "PARSE"}

    times = iter(range(100))
    monkeypatch.setattr("common.poll.time.sleep", lambda _: None)
    monkeypatch.setattr("common.poll.time.monotonic", lambda: next(times))

    client = FakeClient()
    meta = wait_for_parse(client, "VI1", initial_delay=0.0, max_delay=0.0)
    assert meta["status"] == "PARSE"
    assert client.calls == 3


def test_wait_for_parse_tolerates_null_then_completes(monkeypatch):
    """The real /get_metadata returns data=null until the video is visible."""
    class FakeClient:
        def __init__(self):
            self.calls = 0
        def get_metadata(self, video_no):
            self.calls += 1
            if self.calls == 1:
                return None
            if self.calls == 2:
                return {"status": "UNPARSE"}
            return {"status": "PARSE"}

    times = iter(range(100))
    monkeypatch.setattr("common.poll.time.sleep", lambda _: None)
    monkeypatch.setattr("common.poll.time.monotonic", lambda: next(times))

    ticks: list[str] = []
    client = FakeClient()
    meta = wait_for_parse(
        client, "VI1", initial_delay=0.0, max_delay=0.0, on_tick=ticks.append
    )
    assert meta["status"] == "PARSE"
    assert ticks == ["NOT_FOUND", "UNPARSE", "PARSE"]


def test_wait_for_parse_fail_raises(monkeypatch):
    class FakeClient:
        def get_metadata(self, video_no):
            return {"status": "FAIL"}
    monkeypatch.setattr("common.poll.time.sleep", lambda _: None)
    monkeypatch.setattr("common.poll.time.monotonic", lambda: 0)

    with pytest.raises(MemoriesError):
        wait_for_parse(FakeClient(), "VI1", initial_delay=0.0)


def test_visual_rag_merge_overlapping_collapses():
    from agents.visual_rag import _merge_overlapping
    events = [
        {"videoNo": "V", "start": 10, "end": 20, "score": 0.4, "channels": {"visual"}},
        {"videoNo": "V", "start": 19, "end": 25, "score": 0.6, "channels": {"audio"}},
        {"videoNo": "V", "start": 60, "end": 70, "score": 0.5, "channels": {"visual"}},
        {"videoNo": "W", "start": 1,  "end": 5,  "score": 0.3, "channels": {"visual"}},
    ]
    merged = _merge_overlapping(events)
    assert len(merged) == 3
    overlap = next(m for m in merged if m["videoNo"] == "V" and m["start"] == 10)
    assert overlap["end"] == 25
    assert set(overlap["channels"]) == {"visual", "audio"}
    assert overlap["score"] == 0.6


def test_restaurant_compute_metrics():
    from agents.restaurant_service_quality import _compute_metrics
    events = [
        {"type": "server_approaches", "t_sec": 60.0, "score": 0.5, "end_sec": 65.0},
        {"type": "food_delivery", "t_sec": 300.0, "score": 0.5, "end_sec": 305.0},
        {"type": "food_delivery", "t_sec": 1200.0, "score": 0.5, "end_sec": 1205.0},
        {"type": "walkin_no_greet", "t_sec": 30.0, "score": 0.5, "end_sec": 35.0},
    ]
    metrics = _compute_metrics(events)
    assert metrics["food_delivery_count"] == 2
    assert metrics["avg_inter_course_sec"] == 900.0
    assert metrics["table_touches"] == 1
    assert metrics["bounce_count"] == 1


def test_restaurant_dedupe_same_type_within_5s():
    from agents.restaurant_service_quality import _merge_dedupe
    events = [
        {"type": "food_delivery", "t_sec": 100.0, "score": 0.5, "end_sec": 105.0},
        {"type": "food_delivery", "t_sec": 102.0, "score": 0.9, "end_sec": 107.0},  # within 5s, higher score
        {"type": "food_delivery", "t_sec": 200.0, "score": 0.5, "end_sec": 205.0},
    ]
    merged = _merge_dedupe(events)
    assert len(merged) == 2
    assert merged[0]["score"] == 0.9
