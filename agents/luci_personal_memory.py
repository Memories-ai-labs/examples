"""Agent 3 — LUCI-Style Personal Video Memory.

Answers free-text questions over a user's indexed personal-video library —
"What did I order for lunch last Tuesday?" — using time-windowed semantic
search, transcript lookup, then VLM identification of the scene. Mirrors
cookbook use case 3.

ReAct loop:
  1. Pick the time window from the question (--date or natural arg).
  2. Semantic /search restricted to videos captured >= date_after.
  3. Exact-phrase /search_audio_transcripts inside the top hit for the
     spoken-words signal.
  4. VLM call on the top hit's video URL to identify the visual scene.
  5. Compose a one-line answer.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent
from common.client import MemoriesError


SYSTEM_PROMPT = (
    "You are LUCI, a personal memory assistant. Combine the visual evidence and "
    "transcript fragment to identify location/context and order/intent. "
    "Reply strict JSON only."
)


def _build_prompt(start: float, end: float, transcript_hint: str | None) -> str:
    extra = f' Transcript hint near this moment: "{transcript_hint}".' if transcript_hint else ""
    return (
        f"Between {start:.0f}s and {end:.0f}s in this recording, identify: "
        '(1) location/venue (read signage if visible), (2) what the user is doing, '
        '(3) what is being ordered or consumed. Reply JSON: '
        '{"venue": str|null, "activity": str, "order": str|null}.' + extra
    )


def run(
    client: MemoriesClient,
    *,
    question: str,
    date_after: str | None,
    unique_id: str,
    top_k: int,
    filtering_level: str,
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name="LUCI Personal Memory", verbose=verbose)
    agent.thought(f"Question: {question!r}. Date filter (>=): {date_after}.")

    hits = agent.timed(
        f"search('{question}', BY_CLIP)",
        client.search,
        question,
        unique_id=unique_id,
        datetime_taken=date_after,
        top_k=top_k,
        filtering_level=filtering_level,
    )
    if not hits:
        agent.answer("No relevant clip found in that window.")
        return {"question": question, "answer": None, "hits": []}

    top = hits[0]
    agent.thought(
        f"Top hit: video={top.get('videoNo')} window={top.get('startTime')}-{top.get('endTime')}s "
        f"score={top.get('score'):.3f}. Pull a transcript clue for the same period."
    )

    transcript_hint: str | None = None
    try:
        tx = agent.timed(
            "search_audio_transcripts(query='I')",  # generic — refine in real life
            client.search_audio_transcripts,
            "I",
            unique_id=unique_id,
            page_size=20,
        )
        for v in (tx or {}).get("videos", []):
            if v.get("videoNo") == top.get("videoNo"):
                transcript_hint = v.get("audio_ts")
                break
    except Exception as e:
        agent.observe(f"transcript lookup skipped: {e}")

    try:
        url = client.resolve_media_url(top["videoNo"])
    except MemoriesError as e:
        agent.observe(f"no media-URL bridge configured: {e}")
        answer_text = (
            f"Found a candidate moment in video {top['videoNo']} "
            f"({top.get('startTime')}-{top.get('endTime')}s) but cannot identify the "
            "scene visually without a public URL. Set MEMORIES_MEDIA_URL_TEMPLATE."
        )
        agent.answer(answer_text, payload={"transcript_hint": transcript_hint, "top": top})
        return {
            "question": question,
            "answer": answer_text,
            "top_hit": top,
            "transcript_hint": transcript_hint,
            "summary": "retrieval_only",
        }

    raw = agent.timed(
        "vlm_complete(question + scene)",
        client.vlm_complete,
        _build_prompt(float(top.get("startTime", 0)), float(top.get("endTime", 0)), transcript_hint),
        video_url=url,
        system=SYSTEM_PROMPT,
    )
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        parsed = {"raw": raw, "parse_error": True}

    answer_text = _compose_answer(question, parsed)
    agent.answer(answer_text, payload=parsed)
    return {
        "question": question,
        "answer": answer_text,
        "evidence": parsed,
        "top_hit": top,
        "transcript_hint": transcript_hint,
    }


def _compose_answer(question: str, evidence: dict[str, Any]) -> str:
    if not isinstance(evidence, dict) or evidence.get("parse_error"):
        return "I couldn't parse a structured answer from the scene. Try rephrasing."
    venue = evidence.get("venue")
    order = evidence.get("order")
    activity = evidence.get("activity")
    pieces = []
    if venue:
        pieces.append(f"at {venue}")
    if order:
        pieces.append(f"you had {order}")
    if not pieces and activity:
        pieces.append(activity)
    return f"Based on the recording, {', '.join(pieces) or 'I found a related moment'}."


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LUCI-style personal-memory agent")
    p.add_argument("question", help="Free-text question to ask of the library")
    p.add_argument("--date-after", help="yyyy-MM-dd HH:mm:ss — restrict to videos captured at-or-after this time")
    p.add_argument("--unique-id", default="luci-default")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--filtering-level", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.date_after:
        # validate format early to fail fast
        datetime.strptime(args.date_after, "%Y-%m-%d %H:%M:%S")

    client = MemoriesClient()
    summary = run(
        client,
        question=args.question,
        date_after=args.date_after,
        unique_id=args.unique_id,
        top_k=args.top_k,
        filtering_level=args.filtering_level,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
