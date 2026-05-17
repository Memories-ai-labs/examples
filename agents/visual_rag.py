"""Agent 5 — Visual RAG over a Lecture Library.

Answers an arbitrary natural-language query over an indexed video library by
combining cheap retrieval (semantic visual + semantic audio) with expensive
VLM verification per merged candidate. Mirrors cookbook use case 5.

ReAct loop:
  1. Two-channel /search: BY_CLIP for visual and BY_AUDIO for spoken cues.
  2. Merge overlapping time-ranges across channels.
  3. VLM-verify each candidate with a strict-JSON prompt.
  4. Return the verified hits as the agent's answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent
from common.client import MemoriesError


SYSTEM_PROMPT = (
    "You verify whether a candidate clip actually matches the user's query. "
    "Be strict: only mark has_match=true if visual or audio evidence is clear. "
    "Reply JSON only."
)


def _prompt(query: str, start: float, end: float, extract_hint: str | None) -> str:
    extract_clause = (
        f' If has_match is true, also fill "extract": {extract_hint}.'
        if extract_hint
        else ' If has_match is true, also fill "extract" with a short label.'
    )
    return (
        f'Between {start:.0f}s and {end:.0f}s, does this clip match the query: '
        f'"{query}"? Reply JSON: {{"has_match": bool, "extract": str|null, '
        '"confidence": "low"|"medium"|"high"}}.' + extract_clause
    )


def _merge_overlapping(events: list[dict[str, Any]], pad: float = 2.0) -> list[dict[str, Any]]:
    """Merge time-ranges that overlap (with `pad` seconds of slack) within the same video."""
    events.sort(key=lambda e: (e["videoNo"], e["start"]))
    merged: list[dict[str, Any]] = []
    for ev in events:
        if merged and merged[-1]["videoNo"] == ev["videoNo"] and ev["start"] - pad <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], ev["end"])
            merged[-1]["channels"].update(ev["channels"])
            merged[-1]["score"] = max(merged[-1]["score"], ev["score"])
        else:
            merged.append({**ev, "channels": set(ev["channels"])})
    for m in merged:
        m["channels"] = sorted(m["channels"])
    return merged


def _to_event(hit: dict[str, Any], channel: str) -> dict[str, Any]:
    return {
        "videoNo": hit.get("videoNo"),
        "start": float(hit.get("startTime", 0)),
        "end": float(hit.get("endTime", 0)),
        "score": float(hit.get("score", 0.0)),
        "channels": {channel},
        "audio_ts": hit.get("audio_ts"),
    }


def run(
    client: MemoriesClient,
    *,
    query: str,
    visual_query: str | None,
    audio_query: str | None,
    extract_hint: str | None,
    unique_id: str,
    video_nos: list[str] | None,
    top_k: int,
    filtering_level: str,
    max_verify: int,
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name="Visual RAG", verbose=verbose)
    vq = visual_query or query
    aq = audio_query or query
    agent.thought(
        f"Two-channel retrieve for {query!r}. visual={vq!r}, audio={aq!r}. "
        f"Wide net (filtering_level={filtering_level}) — VLM filters false positives later."
    )

    visual = agent.timed(
        "search(visual, BY_CLIP)",
        client.search,
        vq,
        unique_id=unique_id,
        video_nos=video_nos,
        top_k=top_k,
        filtering_level=filtering_level,
    )
    audio = agent.timed(
        "search(audio, BY_AUDIO)",
        client.search,
        aq,
        search_type="BY_AUDIO",
        unique_id=unique_id,
        video_nos=video_nos,
        top_k=top_k,
        filtering_level=filtering_level,
    )

    events = [_to_event(h, "visual") for h in visual] + [_to_event(h, "audio") for h in audio]
    agent.thought(f"Merging {len(events)} raw hits across channels.")
    merged = _merge_overlapping(events)
    agent.observe(f"merged -> {len(merged)} candidates", payload=merged[:3])

    candidates = sorted(merged, key=lambda e: e["score"], reverse=True)[:max_verify]
    verified: list[dict[str, Any]] = []
    url_cache: dict[str, str] = {}
    skipped_no_bridge = False
    for i, c in enumerate(candidates):
        vno = c["videoNo"]
        if vno not in url_cache:
            try:
                url_cache[vno] = client.resolve_media_url(vno)
            except MemoriesError as e:
                agent.observe(f"no media-URL bridge for {vno}: {e}")
                skipped_no_bridge = True
                continue
        agent.thought(
            f"Verify {i+1}/{len(candidates)} @ video={vno} {c['start']:.0f}-{c['end']:.0f}s "
            f"(channels={c['channels']}, score={c['score']:.3f})."
        )
        raw = agent.timed(
            f"vlm_complete({vno} {c['start']:.0f}-{c['end']:.0f}s)",
            client.vlm_complete,
            _prompt(query, c["start"], c["end"], extract_hint),
            video_url=url_cache[vno],
            system=SYSTEM_PROMPT,
        )
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            parsed = {"raw": raw, "parse_error": True}
        if isinstance(parsed, dict) and parsed.get("has_match"):
            verified.append({"candidate": c, "verdict": parsed})

    summary = {
        "query": query,
        "candidates_total": len(merged),
        "candidates_verified": len(candidates),
        "matches": verified,
    }
    if skipped_no_bridge and not verified:
        agent.answer(
            f"{len(candidates)} candidates retrieved across both channels. "
            "VLM verification skipped (no MEMORIES_MEDIA_URL_TEMPLATE).",
            payload={"candidates": candidates[:3]},
        )
        summary["summary"] = "retrieval_only"
    else:
        agent.answer(
            f"{len(verified)}/{len(candidates)} candidates verified as matches.",
            payload={"matches": verified[:3]},
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Visual RAG agent (two-channel retrieve + VLM verify)")
    p.add_argument("query", help="Free-text query over the library")
    p.add_argument("--visual-query", help="Override the visual-channel query string")
    p.add_argument("--audio-query", help="Override the audio-channel query string")
    p.add_argument("--extract-hint", help="If a match is found, what to extract (e.g. 'LaTeX of the equation')")
    p.add_argument("--unique-id", default="default")
    p.add_argument("--video-no", action="append", help="Restrict to specific video(s); repeat flag")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--filtering-level", default="low", choices=["low", "medium", "high"])
    p.add_argument("--max-verify", type=int, default=10, help="Max candidates sent to the VLM")
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    client = MemoriesClient()
    summary = run(
        client,
        query=args.query,
        visual_query=args.visual_query,
        audio_query=args.audio_query,
        extract_hint=args.extract_hint,
        unique_id=args.unique_id,
        video_nos=args.video_no,
        top_k=args.top_k,
        filtering_level=args.filtering_level,
        max_verify=args.max_verify,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
