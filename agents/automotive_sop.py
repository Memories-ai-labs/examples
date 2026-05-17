"""Agent 4 — Automotive Service-Bay SOP Audit.

For each vehicle arrival, verify: was the technician greeting within 60 seconds?
Was the air filter inspected? Mirrors cookbook use case 4.

ReAct loop:
  1. /search for vehicle-arrival events.
  2. For each arrival, ask the VLM about the 60-second window after that
     timestamp — checking greeting, time-to-greet, and air-filter inspection.
  3. Log per-arrival results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent
from common.client import MemoriesError
from common.poll import wait_for_parse


SOP_SCHEMA = (
    '{"technician_greeted": bool, "time_to_greet_sec": int|null, '
    '"air_filter_checked": bool, "tire_pressure_checked": bool, "notes": str}'
)

SYSTEM_PROMPT = (
    "You audit automotive service-bay SOPs from camera footage. "
    "Reply strict JSON only, matching the user-provided schema."
)


def _prompt(start: float, window_end: float) -> str:
    return (
        f"Between {start:.0f}s and {window_end:.0f}s a vehicle just arrived in "
        "the service bay. Answer: did a technician greet the vehicle? How many "
        "seconds elapsed before greeting (null if no greeting)? Was an air "
        "filter inspected? Was tire pressure checked? "
        f"Reply JSON only, matching: {SOP_SCHEMA}"
    )


def run(
    client: MemoriesClient,
    *,
    video_no: str | None,
    upload_path: str | None,
    unique_id: str,
    top_k: int,
    filtering_level: str,
    window_sec: float,
    greet_target_sec: float,
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name="Automotive Service-Bay SOP Audit", verbose=verbose)

    if upload_path:
        agent.thought(f"Need to index {upload_path}.")
        up = agent.timed(
            f"upload_file({upload_path})",
            client.upload_file,
            upload_path,
            unique_id=unique_id,
            camera_model="BAY-CAM",
            tags=["sop-audit", "auto-service"],
        )
        video_no = up["videoNo"]
        agent.timed(
            f"wait_for_parse({video_no})",
            wait_for_parse,
            client,
            video_no,
            on_tick=lambda s: agent.observe(f"indexing status -> {s}"),
        )

    if not video_no:
        raise SystemExit("Either --video-no or --upload must be provided")

    agent.thought("Step 1: detect vehicle arrivals via semantic search.")
    arrivals = agent.timed(
        "search('car drives into service bay', BY_CLIP)",
        client.search,
        "car drives into service bay and parks",
        unique_id=unique_id,
        video_nos=[video_no],
        top_k=top_k,
        filtering_level=filtering_level,
    )
    if not arrivals:
        agent.answer("No vehicle arrivals detected.")
        return {"video_no": video_no, "arrivals": [], "summary": "no_arrivals"}

    try:
        url = client.resolve_media_url(video_no)
    except MemoriesError as e:
        agent.observe(f"no media-URL bridge configured: {e}")
        agent.answer(
            f"{len(arrivals)} arrivals detected. Skipping VLM audit "
            "(set MEMORIES_MEDIA_URL_TEMPLATE to enable).",
            payload={"arrivals": arrivals},
        )
        return {"video_no": video_no, "arrivals": arrivals, "summary": "retrieval_only"}

    audits: list[dict[str, Any]] = []
    for i, a in enumerate(arrivals):
        start = float(a.get("startTime", 0))
        window_end = start + window_sec
        agent.thought(
            f"Arrival {i+1}/{len(arrivals)} @ {start:.0f}s — verify {window_sec:.0f}s window."
        )
        raw = agent.timed(
            f"vlm_complete(window={start:.0f}-{window_end:.0f}s)",
            client.vlm_complete,
            _prompt(start, window_end),
            video_url=url,
            system=SYSTEM_PROMPT,
        )
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            parsed = {"raw": raw, "parse_error": True}
        audits.append({"arrival_t": start, "score": a.get("score"), "result": parsed})

    compliant = sum(
        1 for a in audits
        if isinstance(a["result"], dict)
        and a["result"].get("technician_greeted")
        and (a["result"].get("time_to_greet_sec") or 1e9) <= greet_target_sec
    )
    summary = {
        "video_no": video_no,
        "arrival_count": len(audits),
        "greeting_within_target": compliant,
        "greet_target_sec": greet_target_sec,
        "audits": audits,
    }
    agent.answer(
        f"{compliant}/{len(audits)} arrivals greeted within {greet_target_sec:.0f}s.",
        payload={"greeting_within_target": compliant, "arrival_count": len(audits)},
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Automotive service-bay SOP audit agent")
    p.add_argument("--video-no")
    p.add_argument("--upload")
    p.add_argument("--unique-id", default="auto-service")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--filtering-level", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--window-sec", type=float, default=60.0,
                   help="Seconds after arrival to inspect for greeting / checks")
    p.add_argument("--greet-target-sec", type=float, default=60.0,
                   help="SOP target — greeting must happen within this many seconds")
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    client = MemoriesClient()
    summary = run(
        client,
        video_no=args.video_no,
        upload_path=args.upload,
        unique_id=args.unique_id,
        top_k=args.top_k,
        filtering_level=args.filtering_level,
        window_sec=args.window_sec,
        greet_target_sec=args.greet_target_sec,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
