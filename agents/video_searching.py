"""Agent 4 — Video Searching Agent (public-platform discovery).

Wraps the managed `POST /queries/stream` SSE endpoint, which runs Memories.ai's
open-source Video Searching Agent against YouTube / TikTok / Instagram / X.
Mirrors the PRD's Agent 4 — see Notion PRD §"Agent 4".

This is the simplest of the eight agents because the heavy lifting happens
server-side: this script just opens the SSE stream, prints the typed events
as they arrive, and returns the final structured `complete` payload.

CLI:
    python -m agents.video_searching "Find trending sous vide tutorials on TikTok" \
        --platforms tiktok --time-frame past_week --max-results 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent


def run(
    client: MemoriesClient,
    *,
    query: str,
    platforms: list[str] | None,
    max_results: int,
    time_frame: str | None,
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name="Video Searching Agent (public platforms)", verbose=verbose)
    agent.thought(
        f"Dispatching to /queries/stream: query={query!r}, "
        f"platforms={platforms}, max_results={max_results}, time_frame={time_frame}."
    )

    def on_event(name: str, data: Any) -> None:
        # Each SSE event becomes one observation line. The /queries/stream
        # endpoint emits: started, progress, tool_call, tool_result, error, complete.
        if name == "started":
            agent.observe(f"session_id={data.get('session_id')}", payload=data)
        elif name == "progress":
            agent.observe(f"step {data.get('step')}/{data.get('max_steps')}: {data.get('message')}")
        elif name == "tool_call":
            agent.action(f"tool_call: {data.get('tool')}", payload=data.get("arguments"))
        elif name == "tool_result":
            ok = "ok" if data.get("success") else "FAIL"
            agent.observe(f"tool_result {data.get('tool')} [{ok}]: {data.get('summary')}")
        elif name == "error":
            agent.observe(f"ERROR: {data.get('message')}", payload=data)
        elif name == "complete":
            refs = (data or {}).get("video_references", []) or []
            agent.observe(f"complete: {len(refs)} video(s) returned", payload={"n": len(refs)})
        else:
            agent.observe(f"event[{name}]", payload=data)

    final = agent.timed(
        f"queries_stream({query!r})",
        client.queries_stream,
        query,
        platforms=platforms,
        max_results=max_results,
        time_frame=time_frame,
        on_event=on_event,
    )

    refs = final.get("video_references") or []
    agent.answer(
        f"{len(refs)} video(s) discovered; confidence={final.get('confidence_score')}.",
        payload={"first_3": refs[:3]},
    )
    return final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Public-platform Video Searching Agent (SSE)")
    p.add_argument("query", help="Natural-language query")
    p.add_argument("--platforms", nargs="*", choices=["youtube", "tiktok", "instagram", "twitter"])
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument("--time-frame",
                   help="e.g. 'past_24h', 'past_week', 'past_month' — see /queries/stream docs")
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    client = MemoriesClient()
    final = run(
        client,
        query=args.query,
        platforms=args.platforms,
        max_results=args.max_results,
        time_frame=args.time_frame,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(final, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
