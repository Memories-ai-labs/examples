"""Agent 2 — Full-Service Restaurant Quality Monitoring.

Builds a timeline of service events from a floor-cam shift via repeated semantic
searches, then computes service-quality metrics: table touches per sitting,
inter-course interval, bounce-rate proxy. Mirrors cookbook use case 2.

ReAct loop:
  1. (Optional) /upload the shift footage and wait for PARSE.
  2. Issue one /search per service-event class — each scans the whole video.
  3. Merge into a sorted event timeline.
  4. Compute aggregate metrics; flag the shift summary.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import MemoriesClient, ReActAgent
from common.poll import wait_for_parse


DEFAULT_QUERIES = [
    ("server_approaches", "server walks up to a table to greet or take an order"),
    ("food_delivery", "server delivers a plate or tray of food to a table"),
    ("drink_refill", "server refills a drink at a table"),
    ("check_presentation", "server presents the bill or check to the table"),
    ("table_cleared", "staff clears empty plates or wipes down a table"),
    ("walkin_no_greet", "customer enters the restaurant, waits, then walks out without being seated"),
]


def _merge_dedupe(events: list[dict[str, Any]], min_gap_sec: float = 5.0) -> list[dict[str, Any]]:
    """Collapse events of the same type whose times are within min_gap_sec."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_type[e["type"]].append(e)
    merged: list[dict[str, Any]] = []
    for t, group in by_type.items():
        group.sort(key=lambda x: x["t_sec"])
        kept: list[dict[str, Any]] = []
        for e in group:
            if kept and (e["t_sec"] - kept[-1]["t_sec"]) < min_gap_sec:
                if e["score"] > kept[-1]["score"]:
                    kept[-1] = e
                continue
            kept.append(e)
        merged.extend(kept)
    merged.sort(key=lambda x: x["t_sec"])
    return merged


def _compute_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    food_times = sorted(e["t_sec"] for e in events if e["type"] == "food_delivery")
    inter_course_gaps = [b - a for a, b in zip(food_times, food_times[1:])]
    approaches = sum(1 for e in events if e["type"] == "server_approaches")
    bounces = sum(1 for e in events if e["type"] == "walkin_no_greet")
    return {
        "food_delivery_count": len(food_times),
        "avg_inter_course_sec": round(sum(inter_course_gaps) / len(inter_course_gaps), 1)
        if inter_course_gaps
        else None,
        "table_touches": approaches,
        "bounce_count": bounces,
    }


def run(
    client: MemoriesClient,
    *,
    video_no: str | None,
    upload_path: str | None,
    unique_id: str,
    top_k: int,
    filtering_level: str,
    queries: list[tuple[str, str]],
    verbose: bool,
) -> dict[str, Any]:
    agent = ReActAgent(name="Restaurant Service-Quality Monitor", verbose=verbose)

    if upload_path:
        agent.thought(f"Need to index {upload_path}.")
        up = agent.timed(
            f"upload_file({upload_path})",
            client.upload_file,
            upload_path,
            unique_id=unique_id,
            camera_model="FLOOR-CAM",
            tags=["service-quality"],
            video_transcription_prompt="Identify table service events: approach, delivery, check.",
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

    agent.thought(
        f"Run {len(queries)} semantic searches in parallel-conceptually to build a service timeline."
    )
    events: list[dict[str, Any]] = []
    for label, q in queries:
        hits = agent.timed(
            f"search('{label}', BY_CLIP, top_k={top_k})",
            client.search,
            q,
            unique_id=unique_id,
            video_nos=[video_no],
            top_k=top_k,
            filtering_level=filtering_level,
        )
        for h in hits:
            events.append({
                "t_sec": float(h.get("startTime", 0)),
                "type": label,
                "score": float(h.get("score", 0.0)),
                "end_sec": float(h.get("endTime", 0)),
            })

    agent.thought(f"Got {len(events)} raw events. Deduping same-type events within 5s.")
    merged = _merge_dedupe(events)
    metrics = _compute_metrics(merged)

    summary = {
        "video_no": video_no,
        "event_count": len(merged),
        "metrics": metrics,
        "timeline": merged,
    }
    agent.answer(
        f"Timeline built: {len(merged)} events, metrics={metrics}",
        payload={"metrics": metrics, "first_5_events": merged[:5]},
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Full-service restaurant quality-monitoring agent")
    p.add_argument("--video-no")
    p.add_argument("--upload")
    p.add_argument("--unique-id", default="restaurant-floor")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--filtering-level", default="medium", choices=["low", "medium", "high"])
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
        queries=DEFAULT_QUERIES,
        verbose=not args.quiet,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
